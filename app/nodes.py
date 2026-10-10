import random
import re
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
WEAK_CAP = 3  # 오답 횟수 가중치 상한 (많이 틀린 키워드 하나에만 쏠리지 않게)
LANGUAGES = ["c", "java", "python"]  # 출제할 언어 (안 배운 언어는 빼면 그 언어 키워드도 안 나옴)

# 묻는 방식별 비율. 실기 유형은 단답형·약술형·코드뿐이고, 그 안에서 묻는 방식이 다르다 (최근 실기는 단답형 위주)
THEORY_FORMATS = {"용어 쓰기": 35, "보기에서 고르기": 20, "빈칸 채우기": 15, "순서대로 쓰기": 10, "서술": 20}
CODE_FORMATS = {"실행 결과": 75, "코드 빈칸": 25}
FORMAT_TYPE = {"서술": "약술형", "실행 결과": "코드", "코드 빈칸": "코드"}  # 나머지는 단답형

# 답 입력할 때 보여줄 안내 (답이 여러 개면 쉼표로 구분)
ANSWER_HINT = {
    "보기에서 고르기": "여러 개면 쉼표(,)로 구분",
    "빈칸 채우기": "① 답, ② 답 순서로 쉼표(,)로 구분",
    "순서대로 쓰기": "순서대로 쉼표(,)로 구분",
    "서술": "1~2문장으로 설명",
    "코드 빈칸": "( ① )에 들어갈 코드만",
}


# 같은 코드 문제 6개로 비교했을 때 LLM 정답이 실제 실행 결과와 맞은 수:
# gpt-5.4-mini 1/6, gpt-5.5 5/6, gpt-6.1-sol 5/6, gpt-6-sol 6/6 → 출제는 gpt-6-sol
gen_llm = ChatOpenAI(model="gpt-6-sol")
review_llm = ChatOpenAI(model="gpt-5.4-mini")  # 검수는 빠른 모델로 (코드 문제는 어차피 실행으로 확인)


class Question(BaseModel):
    """LLM이 출제 결과를 이 형식으로 돌려준다."""
    title: str = Field(description="문제 내용을 15자 안팎으로 요약한 제목. 예: static 메서드 업캐스팅 출력")
    question: str = Field(description="문제 본문")
    answer: str = Field(description="정답. 답이 여러 개면 쉼표로 구분")
    key_points: list[str] = Field(description="약술형이면 채점용 핵심 단어 2~3개, 아니면 빈 리스트")
    code: str = Field(description="코드 문제면 실행 가능한 전체 코드, 아니면 빈 문자열")
    blank: str = Field(description="코드 빈칸 문제면 code에서 빈칸으로 만들 짧은 코드 조각, 아니면 빈 문자열")
    explanation: str = Field(description="정답 풀이 2~4문장. 이론은 왜 이 답인지 자료 근거로, 코드는 값이 바뀌는 과정을 따라가며 설명. 코드·식은 `백틱`으로 감싼다")

class Review(BaseModel):
    """LLM이 검수 결과를 이 형식으로 돌려준다."""
    ok: bool = Field(description="문제가 없으면 true")
    reason: str = Field(description="탈락 이유. 통과면 빈 문자열")


# run_code가 실패했을 때 돌려주는 문장의 시작 부분
ERROR_PREFIXES = ("컴파일 에러", "컴파일 경고", "실행 에러", "시간 초과", "지원하지 않는 언어")


# 실기에서 자주 쓰는 문제 형식 (기출 원문이 아니라 묻는 방식만). 매번 하나를 골라 같은 키워드라도 다르게 묻는다.
QUESTION_STYLES = {
    "용어 쓰기": [
        "설명을 읽고 해당하는 용어를 쓰는 문제",
        "사례나 상황을 주고, 그에 해당하는 개념을 쓰는 문제",
    ],
    "서술": [
        "개념의 정의를 설명하는 문제",
        "비슷한 두 개념의 차이를 비교해 설명하는 문제",
        "개념의 특징이나 목적을 설명하는 문제",
        "사례를 주고 어떤 개념인지와 그 이유를 설명하는 문제",
    ],
    "코드": [
        "반복문과 변수 누적을 추적하는 문제",
        "배열·리스트 인덱스를 추적하는 문제",
        "함수 호출 순서와 반환값을 추적하는 문제",
        "조건 분기를 여러 번 거치는 문제",
        "문자열을 다루는 문제",
        "언어의 주요 함수·메서드(문자열 함수 등)를 쓰는 문제",
    ],
}

CODE_RULES = ("코드는 표준 입력 없이 실행되고, 정의되지 않은 동작(같은 변수를 한 식에서 여러 번 증감 등)은 쓰지 않는다. "
              "Java는 public class Main을 쓴다.")

FORMAT_GUIDE = {
    "용어 쓰기": "용어나 짧은 값 하나로 답하는 문제. 정답이 하나로 정해져야 한다.",
    "보기에서 고르기": "<보기>에는 설명 문장이 아니라 용어만 5~7개 나열하고, 조건에 맞는 것을 골라 쓰게 한다. "
                "답이 1개면 '하나를 골라 쓰시오', 2개 이상이면 '모두 골라 쓰시오'라고 쓴다. answer에는 기호(ㄱ, ①) 없이 보기의 용어를 쉼표로 구분해 쓴다.",
    "빈칸 채우기": "문장이나 표에 ( ① ), ( ② ) 같은 빈칸을 1~3개 두고 채우게 한다. "
                "빈칸 답은 괄호 없는 짧은 용어로 하고, answer에는 빈칸 번호 순서대로 쉼표로 구분해 쓴다.",
    "순서대로 쓰기": "자료에서 순서가 있는 항목(강약 순서, 계층, 실행 순서 등)을 순서대로 쓰게 한다. "
                "어떤 순서인지(예: 강한 것부터) 문제에 분명히 쓰고, 문제에 항목을 보여줄 때는 순서를 반드시 섞는다 (섞었다는 말은 쓰지 않는다). "
                "answer에는 순서대로 쉼표로 구분해 쓴다.",
    "서술": "1~2문장으로 설명하는 문제. 채점용 핵심 단어 2~3개를 key_points에 넣는다. "
            "핵심 단어는 괄호 없는 짧은 단어(1~2어절)로 쓰고, answer 문장에 글자 그대로 들어 있는 단어만 고른다.",
    "실행 결과": "코드의 출력 결과를 묻는 문제. " + CODE_RULES,
    "코드 빈칸": "실행 가능한 전체 코드를 code에 쓰고, 그중 빈칸으로 만들 짧은 코드 조각 하나(조건식, 연산자, 함수 호출 등)를 blank에 쓴다. "
              "blank는 짧게(30자 이내) 잡되, code 전체에서 딱 한 번만 나오는 조각이어야 한다. "
              "같은 글자가 다른 곳에도 있으면 조금 더 길게 잡는다 (예: 'n - 1'이 여러 번 나오면 'f(n - 1) * n'). " + CODE_RULES,
}


class QuizState(TypedDict):
    # 1. 약점 분석
    keyword: str            # 이번에 출제할 키워드
    category: str           # 키워드의 과목
    past_titles: list[str]  # 이 키워드로 전에 틀린 문제 제목들 (같은 문제 반복 방지)
    recent_keywords: list[str]  # 최근 출제한 키워드 (연속 출제 방지)
    scope: str              # 출제 범위: "전체" / "코드" / "이론" (화면에서 고름, 없으면 전체)

    # 2. 출제
    context: str            # RAG로 찾은 개념 자료
    question_type: str      # 실기 유형: "단답형" / "약술형" / "코드"
    question_format: str    # 묻는 방식: THEORY_FORMATS 또는 CODE_FORMATS 중 하나
    title: str              # 문제 요약 제목 (노션 제목용)
    question: str           # 문제 본문
    answer: str             # 정답
    key_points: list[str]   # 약술형 채점용 핵심 단어 2~3개
    language: str           # 코드 문제 언어 ("c" / "java" / "python")
    code: str               # 코드 문제의 전체 코드 (실행용)
    blank: str              # 코드 빈칸 문제에서 빈칸으로 만든 코드 조각
    shown_code: str         # 화면에 보여줄 코드 (코드 빈칸이면 빈칸 처리된 코드)
    explanation: str        # 정답 풀이 (채점 후 보여줌)

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
    coding = [k for k in candidates if categories[k] == "프로그래밍" and _language_ok(k)]
    theory = [k for k in candidates if categories[k] != "프로그래밍"]
    scope = state.get("scope", "전체")
    if scope == "코드" and coding:
        pool = coding
    elif scope == "이론":
        pool = theory
    else:
        pool = coding if coding and random.random() < CODE_RATE else theory

    # 2) 그 안에서 약점 키워드를 우선으로 뽑는다
    weak = {k: len(records[k]) for k in pool if k in records}
    if weak and random.random() < WEAK_RATE:
        # 가중치에 상한을 둔다 → 7번 틀린 키워드와 1번 틀린 키워드가 7:1이 아니라 3:1
        weights = [min(n, WEAK_CAP) for n in weak.values()]
        keyword = random.choices(list(weak), weights=weights)[0]
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


LANGUAGE_PREFIX = [("C ", "c"), ("Java", "java"), ("Python", "python")]


def _language_ok(keyword: str) -> bool:
    """출제할 언어(LANGUAGES)에 없는 언어의 키워드는 뺀다. (예: Python을 안 배웠으면 Python 키워드 제외)"""
    return all(lang in LANGUAGES for prefix, lang in LANGUAGE_PREFIX if keyword.startswith(prefix))


def _pick_language(keyword: str) -> str:
    """키워드에 언어가 있으면 그 언어, 없으면 LANGUAGES 중 랜덤."""
    for prefix, language in LANGUAGE_PREFIX:
        if keyword.startswith(prefix):
            return language
    return random.choice(LANGUAGES)


def _pick_focus(context: str) -> str:
    """개념 자료에서 이번 문제의 중심 내용 하나를 고른다. (같은 키워드라도 매번 다른 부분을 묻게)"""
    items = []
    for line in context.splitlines()[1:]:  # 첫 줄(## 제목) 제외
        if line.startswith("- "):
            items.append(line)
        elif line.startswith("  ") and items:
            items[-1] += "\n" + line  # 하위 항목은 위 항목에 붙인다
    items = [i for i in items if not i.startswith(("- 빈도", "- 시험 포인트"))]
    return random.choice(items) if items else ""


def generate_question(state: QuizState) -> dict:
    """RAG로 개념 자료를 찾고, 그 자료 안에서 문제를 만든다."""
    keyword = state["keyword"]
    context = search(keyword, k=1)[0].page_content

    formats = dict(CODE_FORMATS if state["category"] == "프로그래밍" else THEORY_FORMATS)
    if not re.search(r"→|>|순서|계층", context):
        formats.pop("순서대로 쓰기", None)  # 자료에 순서가 없으면 LLM이 순서를 지어낸다
    fmt = random.choices(list(formats), weights=list(formats.values()))[0]
    qtype = FORMAT_TYPE.get(fmt, "단답형")
    is_code = fmt in CODE_FORMATS
    language = _pick_language(keyword) if is_code else ""

    styles = QUESTION_STYLES.get("코드" if is_code else fmt, [])
    style = random.choice(styles) if styles else ""
    focus = _pick_focus(context)

    prompt = f"""정보처리기사 실기 문제를 1개 만들어라.

키워드: {keyword}
유형: {qtype} ({fmt}) - {FORMAT_GUIDE[fmt]}
{f"출제 형식: {style}" if style else ""}
이번 문제의 중심 내용: {focus} (출제 방향일 뿐, 문제 문장에 그대로 쓰지 않는다)
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

    q = gen_llm.with_structured_output(Question).invoke(prompt)

    question = q.question
    if is_code:
        # 코드는 code에 따로 두고, 문제 문장은 고정한다. 코드 빈칸은 검수에서 실행 결과를 넣어 다시 만든다
        question = f"다음 {language} 코드의 실행 결과를 쓰시오."

    print(f"[출제] {qtype} · {fmt}{f' · {style}' if style else ''} (재출제 {state['retry_count']}회)")
    return {
        "context": context,
        "question_type": qtype,
        "question_format": fmt,
        "title": q.title,
        "question": question,
        "answer": q.answer,
        "key_points": q.key_points,
        "language": language,
        "code": q.code if is_code else "",  # 이론 문제에 코드가 섞이지 않게
        "blank": q.blank.strip() if fmt == "코드 빈칸" else "",
        "shown_code": "",  # 코드 문제는 검수에서 정한다
        "explanation": q.explanation,
    }


def review_question(state: QuizState) -> dict:
    """문제를 검수한다. 코드 문제는 실제로 실행해서 정답을 확정한다."""
    fail = {"review_ok": False, "retry_count": state["retry_count"] + 1}

    # 코드 문제: 실제로 실행해서 정답을 확정한다
    if state["question_format"] in CODE_FORMATS:
        output = run_code.invoke({"language": state["language"], "code": state["code"]})
        if not output.strip() or output.startswith(ERROR_PREFIXES):
            print("[검수] 탈락 - 코드 실행 실패")
            return {**fail, "review_feedback": output[:500] or "출력이 없음"}
        real = output.strip()

        if state["question_format"] == "코드 빈칸":
            # 빈칸으로 만들 조각이 코드에 딱 한 번 있어야 빈칸 위치가 하나로 정해진다
            blank = state["blank"]
            if not blank or state["code"].count(blank) != 1:
                print("[검수] 탈락 - 빈칸 위치가 하나로 정해지지 않음")
                return {**fail, "review_feedback": f"blank '{blank}'가 code 안에 정확히 한 번 나와야 한다"}
            print("[검수] 통과")
            return {
                "review_ok": True,
                "answer": blank,
                "question": f"다음 {state['language']} 코드의 실행 결과가 아래와 같을 때, ( ① )에 들어갈 코드를 쓰시오.\n\n[실행 결과]\n{real}",
                "shown_code": state["code"].replace(blank, "( ① )", 1),
            }

        # 실행 결과: LLM 정답 대신 실제 실행 결과를 정답으로 쓴다
        explanation = state["explanation"]
        if real != state["answer"].strip():
            print(f"[검수] LLM 정답 {state['answer']!r} → 실행 결과 {real!r}로 교체")
            explanation = _explain_code(state, real)  # 틀린 정답으로 쓴 풀이라 다시 쓴다
        print("[검수] 통과")
        return {"review_ok": True, "answer": real, "shown_code": state["code"], "explanation": explanation}

    # 약술형: 모범 답안을 그대로 써도 정답이 되도록, 핵심 단어가 정답 문장에 글자 그대로 있는지 코드로 검사
    missing = [k for k in state["key_points"] if _compact(k) not in _compact(state["answer"])]
    if state["question_format"] == "서술" and missing:
        print(f"[검수] 탈락 - 핵심 단어 {missing}가 정답 문장에 없음")
        return {**fail, "review_feedback": f"핵심 단어 {missing}를 정답 문장에 글자 그대로 넣어라"}

    # 이론 문제: 자료와 비교해서 검사
    prompt = f"""정보처리기사 실기 문제를 검수하라. 아래 기준 중 하나라도 어기면 탈락이다.
1. 문제와 정답이 자료 내용과 맞는다.
2. 정답이 하나로 정해진다. 보기에서 고르기·빈칸 채우기·순서대로 쓰기는 정답 목록 전체가 하나로 정해진다.
3. 약술형이면 채점용 핵심 단어 {state['key_points']}가 자료에 있고, 정답에 들어 있다.
4. 풀이가 정답과 맞고, 자료 내용과 어긋나지 않는다.

[자료]
{state['context']}

[유형] {state['question_type']} ({state['question_format']})
[문제] {state['question']}
[정답] {state['answer']}
[풀이] {state['explanation']}
"""
    r = review_llm.with_structured_output(Review).invoke(prompt)
    if not r.ok:
        print(f"[검수] 탈락 - {r.reason}")
        return {**fail, "review_feedback": r.reason}

    print("[검수] 통과")
    return {"review_ok": True}


def _explain_code(state: QuizState, output: str) -> str:
    """LLM 정답이 실행 결과와 다를 때, 실제 실행 결과에 맞춰 풀이를 다시 쓴다."""
    prompt = f"""아래 {state['language']} 코드의 실제 실행 결과는 다음과 같다.
[실행 결과]
{output}

이 결과가 나오는 과정을 2~4문장으로 풀이하라. 값이 바뀌는 과정을 따라가며 설명하고, 코드·식은 `백틱`으로 감싼다.

[코드]
{state['code']}
"""
    return gen_llm.invoke(prompt).content.strip()


def ask_answer(state: QuizState) -> dict:
    """문제를 보여주고 사용자 답을 기다린다. (HITL)"""
    user_answer = interrupt({
        "kind": "question",
        "question": state["question"],
        "code": state["shown_code"],
        "type": state["question_type"],
        "format": state["question_format"],
        "hint": ANSWER_HINT.get(state["question_format"], ""),
    })
    return {"user_answer": user_answer.strip()}


def _normalize(text: str) -> str:
    """띄어쓰기·줄바꿈을 한 칸으로 맞추고 소문자로 바꾼다."""
    return " ".join(text.split()).lower()


def _compact(text: str) -> str:
    """띄어쓰기를 전부 없앤다. (예: 경계값 분석 = 경계값분석)"""
    return _normalize(text).replace(" ", "")


def _split(text: str) -> list[str]:
    """여러 개 답을 쉼표·화살표로 나누고, 앞에 붙은 번호(①, 1., (1))는 뗀다."""
    parts = re.split(r"[,，、\n]|->|→", text)
    return [_compact(re.sub(r"^\s*(?:[①-⑩]|\(\d+\)|\d+[.)])\s*", "", p)) for p in parts if p.strip()]


def grade_answer(state: QuizState) -> dict:
    """정답 여부를 고정 규칙으로 채점한다."""
    user, answer, fmt = state["user_answer"], state["answer"], state["question_format"]

    if fmt == "서술":
        # 핵심 단어 포함 여부 (띄어쓰기 무시)
        key_points = state["key_points"]
        hits = [k for k in key_points if _compact(k) in _compact(user)]
        is_correct = len(hits) >= min(2, len(key_points))
        print(f"[채점] 핵심 단어 {len(hits)}/{len(key_points)}개 포함: {hits}")
    elif fmt == "보기에서 고르기":
        # 고른 것이 모두 맞으면 정답 (순서는 상관없음)
        is_correct = sorted(_split(user)) == sorted(_split(answer))
    elif fmt in ("빈칸 채우기", "순서대로 쓰기"):
        # 빈칸 번호·나열 순서대로 맞아야 정답
        is_correct = _split(user) == _split(answer)
    elif fmt in ("용어 쓰기", "코드 빈칸"):
        # 띄어쓰기만 무시하고 엄격하게 비교 (같은 뜻인지는 사람이 확인)
        is_correct = _compact(user) == _compact(answer)
    else:
        # 실행 결과: 출력값은 띄어쓰기도 의미가 있어서 한 칸으로만 맞춘다 ("10 20" ≠ "1020")
        is_correct = _normalize(user) == _normalize(answer)

    print(f"[채점] {'정답' if is_correct else '오답'} (정답: {state['answer']})")
    print(f"[풀이] {state['explanation']}")
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
