import random
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.notion import get_wrong_records, save_wrong_answer
from app.rag import load_documents, search
from app.tools import run_code


WEAK_RATE = 0.7  # 약점 키워드를 뽑을 확률
RECENT_LIMIT = 3  # 최근 N문제에 나온 키워드는 다시 안 뽑음
CODE_RATE = 0.4  # 랜덤 출제 때 코드 문제 비율 (실기는 코드 문제 비중이 큼)
ESSAY_RATE = 0.3  # 이론 문제 중 약술형 비율 (최근 실기는 단답형 위주)


llm = ChatOpenAI(model="gpt-5.4-mini")


class Question(BaseModel):
    """LLM이 출제 결과를 이 형식으로 돌려준다."""
    title: str = Field(description="문제 내용을 15자 안팎으로 요약한 제목. 예: static 메서드 업캐스팅 출력")
    question: str = Field(description="문제 본문")
    answer: str = Field(description="정답")
    key_points: list[str] = Field(description="약술형이면 채점용 핵심 단어 2~3개, 아니면 빈 리스트")
    code: str = Field(description="코드 문제면 실행 가능한 전체 코드, 아니면 빈 문자열")

class Review(BaseModel):
    """LLM이 검수 결과를 이 형식으로 돌려준다."""
    ok: bool = Field(description="문제가 없으면 true")
    reason: str = Field(description="탈락 이유. 통과면 빈 문자열")


# run_code가 실패했을 때 돌려주는 문장의 시작 부분
ERROR_PREFIXES = ("컴파일 에러", "컴파일 경고", "실행 에러", "시간 초과", "지원하지 않는 언어")


TYPE_GUIDE = {
    "단답형": "용어나 짧은 값 하나로 답하는 문제. 정답이 하나로 정해져야 한다.",
    "약술형": "1~2문장으로 설명하는 문제. 채점용 핵심 단어 2~3개를 key_points에 넣는다. "
            "핵심 단어는 괄호 없는 짧은 단어(1~2어절)로 쓴다.",
    "코드": "코드의 출력 결과를 묻는 문제. 표준 입력 없이 실행되고, "
            "정의되지 않은 동작(같은 변수를 한 식에서 여러 번 증감 등)은 쓰지 않는다. "
            "Java는 public class Main을 쓴다.",
}


class QuizState(TypedDict):
    # 1. 약점 분석
    keyword: str            # 이번에 출제할 키워드
    category: str           # 키워드의 과목
    past_titles: list[str]  # 이 키워드로 전에 틀린 문제 제목들 (같은 문제 반복 방지)
    recent_keywords: list[str]  # 최근 출제한 키워드 (연속 출제 방지)

    # 2. 출제
    context: str            # RAG로 찾은 개념 자료
    question_type: str      # "단답형" / "약술형" / "코드"
    title: str              # 문제 요약 제목 (노션 제목용)
    question: str           # 문제 본문
    answer: str             # 정답
    key_points: list[str]   # 약술형 채점용 핵심 단어 2~3개
    language: str           # 코드 문제 언어 ("c" / "java" / "python")
    code: str               # 코드 문제의 코드

    # 3. 검수
    review_ok: bool         # 검수 통과 여부
    review_feedback: str    # 탈락 이유 (재출제할 때 LLM에게 전달)
    retry_count: int        # 재출제 횟수

    # 4~5. 답 입력, 채점
    user_answer: str        # 사용자가 입력한 답
    is_correct: bool        # 정답 여부

    # 7. 종료 조건
    solved: int             # 푼 문제 수
    total: int              # 목표 문제 수 (N)
    correct_count: int      # 맞힌 문제 수 (마지막 결과 출력용)


async def analyze_weakness(state: QuizState) -> dict:
    """노션 오답 DB를 보고 이번에 출제할 키워드를 고른다."""
    # 전체 키워드 → 과목 (md 파일에서 바로 읽음, API 호출 없음)
    categories = {d.metadata["keyword"]: d.metadata["category"] for d in load_documents()}
    recent = state["recent_keywords"]

    records = await get_wrong_records()

    # 최근에 나온 키워드는 빼고 고른다 → 같은 키워드가 연속으로 안 나옴
    candidates = [k for k in categories if k not in recent]

    # 1) 코드/이론을 먼저 정한다 → 약점이 이론에 몰려 있어도 코드 문제 비율 유지
    coding = [k for k in candidates if categories[k] == "프로그래밍"]
    theory = [k for k in candidates if categories[k] != "프로그래밍"]
    pool = coding if coding and random.random() < CODE_RATE else theory

    # 2) 그 안에서 약점 키워드를 우선으로 뽑는다
    weak = {k: len(records[k]) for k in pool if k in records}
    if weak and random.random() < WEAK_RATE:
        keyword = random.choices(list(weak), weights=list(weak.values()))[0]
        reason = f"약점 (오답 {weak[keyword]}회)"
    else:
        keyword = random.choice(pool)
        reason = "랜덤"

    print(f"\n[약점 분석] {keyword} - {reason}")
    return {
        "keyword": keyword,
        "category": categories[keyword],
        "past_titles": records.get(keyword, []),
        "recent_keywords": (recent + [keyword])[-RECENT_LIMIT:],
        "retry_count": 0,
        "review_feedback": "",
    }


def _pick_language(keyword: str) -> str:
    """키워드에 언어가 있으면 그 언어, 없으면 3개 중 랜덤."""
    for prefix, language in [("C ", "c"), ("Java", "java"), ("Python", "python")]:
        if keyword.startswith(prefix):
            return language
    return random.choice(["c", "java", "python"])


def generate_question(state: QuizState) -> dict:
    """RAG로 개념 자료를 찾고, 그 자료 안에서 문제를 만든다."""
    keyword = state["keyword"]
    if state["category"] == "프로그래밍":
        qtype = "코드"
    else:
        qtype = "약술형" if random.random() < ESSAY_RATE else "단답형"

    language = _pick_language(keyword) if qtype == "코드" else ""

    context = search(keyword, k=1)[0].page_content

    prompt = f"""정보처리기사 실기 문제를 1개 만들어라.

키워드: {keyword}
유형: {qtype} - {TYPE_GUIDE[qtype]}
{f"언어: {language}" if language else ""}

반드시 아래 자료 내용 안에서만 출제한다.
[자료]
{context}
"""
    if state["past_titles"]:
        titles = "\n".join(f"- {t}" for t in state["past_titles"])
        prompt += f"\n[이미 낸 문제]\n{titles}\n위 문제들과 다른 내용을 묻는 문제를 만들어라.\n"
    if state["review_feedback"]:
        prompt += f"\n[이전 문제가 검수에서 탈락한 이유]\n{state['review_feedback']}\n같은 문제가 생기지 않게 다시 만들어라.\n"

    q = llm.with_structured_output(Question).invoke(prompt)

    question = q.question
    if qtype == "코드":
        # 코드는 code에 따로 두고, 문제 문장은 고정한다 (보여주는 코드 = 실행하는 코드)
        question = f"다음 {language} 코드의 실행 결과를 쓰시오."

    print(f"[출제] {qtype} (재출제 {state['retry_count']}회)")
    return {
        "context": context,
        "question_type": qtype,
        "title": q.title,
        "question": question,
        "answer": q.answer,
        "key_points": q.key_points,
        "language": language,
        "code": q.code if qtype == "코드" else "",  # 이론 문제에 코드가 섞이지 않게
    }


def review_question(state: QuizState) -> dict:
    """문제를 검수한다. 코드 문제는 실제로 실행해서 정답을 확정한다."""
    fail = {"review_ok": False, "retry_count": state["retry_count"] + 1}

    # 코드 문제: LLM 정답 대신 실제 실행 결과를 정답으로 쓴다
    if state["question_type"] == "코드":
        output = run_code.invoke({"language": state["language"], "code": state["code"]})
        if not output.strip() or output.startswith(ERROR_PREFIXES):
            print("[검수] 탈락 - 코드 실행 실패")
            return {**fail, "review_feedback": output[:500] or "출력이 없음"}

        real = output.strip()
        if real != state["answer"].strip():
            print(f"[검수] LLM 정답 {state['answer']!r} → 실행 결과 {real!r}로 교체")
        print("[검수] 통과")
        return {"review_ok": True, "answer": real}

    # 단답형·약술형: 자료와 비교해서 검사
    prompt = f"""정보처리기사 실기 문제를 검수하라. 아래 기준 중 하나라도 어기면 탈락이다.
1. 문제와 정답이 자료 내용과 맞는다.
2. 정답이 하나로 정해진다. (단답형)
3. 약술형이면 채점용 핵심 단어 {state['key_points']}가 자료에 있고, 정답에 들어 있다.

[자료]
{state['context']}

[유형] {state['question_type']}
[문제] {state['question']}
[정답] {state['answer']}
"""
    r = llm.with_structured_output(Review).invoke(prompt)
    if not r.ok:
        print(f"[검수] 탈락 - {r.reason}")
        return {**fail, "review_feedback": r.reason}

    print("[검수] 통과")
    return {"review_ok": True}


def ask_answer(state: QuizState) -> dict:
    """문제를 보여주고 사용자 답을 기다린다. (HITL)"""
    user_answer = interrupt({
        "kind": "question",
        "question": state["question"],
        "code": state["code"],
        "type": state["question_type"],
    })
    return {"user_answer": user_answer.strip()}


def _normalize(text: str) -> str:
    """띄어쓰기·줄바꿈을 한 칸으로 맞추고 소문자로 바꾼다."""
    return " ".join(text.split()).lower()


def _compact(text: str) -> str:
    """띄어쓰기를 전부 없앤다. (예: 경계값 분석 = 경계값분석)"""
    return _normalize(text).replace(" ", "")


def grade_answer(state: QuizState) -> dict:
    """정답 여부를 고정 규칙으로 채점한다."""
    user = state["user_answer"]

    if state["question_type"] == "약술형":
        # 핵심 단어 포함 여부 (띄어쓰기 무시)
        key_points = state["key_points"]
        hits = [k for k in key_points if _compact(k) in _compact(user)]
        is_correct = len(hits) >= min(2, len(key_points))
        print(f"[채점] 핵심 단어 {len(hits)}/{len(key_points)}개 포함: {hits}")
    elif state["question_type"] == "단답형":
        # 띄어쓰기만 무시하고 엄격하게 비교 (같은 뜻인지는 사람이 확인)
        is_correct = _compact(user) == _compact(state["answer"])
    else:
        # 코드: 출력값은 띄어쓰기도 의미가 있어서 한 칸으로만 맞춘다 ("10 20" ≠ "1020")
        is_correct = _normalize(user) == _normalize(state["answer"])

    print(f"[채점] {'정답' if is_correct else '오답'} (정답: {state['answer']})")
    return {
        "is_correct": is_correct,
        "solved": state["solved"] + 1,
        "correct_count": state["correct_count"] + is_correct,
    }


def confirm_answer(state: QuizState) -> dict:
    """오답일 때 같은 뜻으로 썼는지 사람이 확인한다. (HITL)"""
    reply = interrupt({"kind": "confirm", "answer": state["answer"]})
    if reply.strip().lower() == "y":
        print("[확인] 같은 뜻 → 정답 처리")
        return {"is_correct": True, "correct_count": state["correct_count"] + 1}
    return {}


async def save_wrong(state: QuizState) -> dict:
    """틀린 문제를 노션 오답노트에 저장한다."""
    try:
        await save_wrong_answer(state)
        print("[저장] 노션 오답노트에 저장 완료")
    except Exception as e:
        print(f"[저장] 실패 - {e}")  # 저장 실패로 퀴즈가 멈추지 않게
    return {}
