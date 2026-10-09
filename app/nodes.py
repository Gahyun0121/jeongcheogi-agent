import random
from typing import TypedDict
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.notion import get_wrong_counts
from app.rag import load_documents, search

WEAK_RATE = 0.7  # 약점 키워드를 뽑을 확률

llm = ChatOpenAI(model="gpt-5.4-mini")


class Question(BaseModel):
    """LLM이 출제 결과를 이 형식으로 돌려준다."""
    question: str = Field(description="문제 본문")
    answer: str = Field(description="정답")
    key_points: list[str] = Field(description="약술형이면 채점용 핵심 단어 2~3개, 아니면 빈 리스트")
    language: str = Field(description='코드 문제면 "c", "java", "python" 중 하나, 아니면 빈 문자열')
    code: str = Field(description="코드 문제면 실행 가능한 전체 코드, 아니면 빈 문자열")


TYPE_GUIDE = {
    "단답형": "용어나 짧은 값 하나로 답하는 문제. 정답이 하나로 정해져야 한다.",
    "약술형": "1~2문장으로 설명하는 문제. 채점용 핵심 단어 2~3개를 key_points에 넣는다.",
    "코드": "코드의 출력 결과를 묻는 문제. 표준 입력 없이 실행되고, "
            "정의되지 않은 동작(같은 변수를 한 식에서 여러 번 증감 등)은 쓰지 않는다. "
            "Java는 public class Main을 쓴다.",
}


class QuizState(TypedDict):
    # 1. 약점 분석
    keyword: str            # 이번에 출제할 키워드
    category: str           # 키워드의 과목

    # 2. 출제
    context: str            # RAG로 찾은 개념 자료
    question_type: str      # "단답형" / "약술형" / "코드"
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

    counts = await get_wrong_counts()
    weak = {k: c for k, c in counts.items() if k in categories}

    if weak and random.random() < WEAK_RATE:
        keyword = random.choices(list(weak), weights=list(weak.values()))[0]
        reason = f"약점 (오답 {weak[keyword]}회)"
    else:
        keyword = random.choice(list(categories))
        reason = "랜덤"

    print(f"\n[약점 분석] {keyword} - {reason}")
    return {"keyword": keyword, "category": categories[keyword], "retry_count": 0, "review_feedback": ""}


def generate_question(state: QuizState) -> dict:
    """RAG로 개념 자료를 찾고, 그 자료 안에서 문제를 만든다."""
    keyword = state["keyword"]
    qtype = "코드" if state["category"] == "프로그래밍" else random.choice(["단답형", "약술형"])

    context = search(keyword, k=1)[0].page_content

    prompt = f"""정보처리기사 실기 문제를 1개 만들어라.

키워드: {keyword}
유형: {qtype} - {TYPE_GUIDE[qtype]}

반드시 아래 자료 내용 안에서만 출제한다.
[자료]
{context}
"""
    if state["review_feedback"]:
        prompt += f"\n[이전 문제가 검수에서 탈락한 이유]\n{state['review_feedback']}\n같은 문제가 생기지 않게 다시 만들어라.\n"

    q = llm.with_structured_output(Question).invoke(prompt)

    question = q.question
    if qtype == "코드":
        # 화면에 보여주는 코드와 실행하는 코드가 같도록 문제 문장을 직접 만든다
        question = f"다음 {q.language} 코드의 실행 결과를 쓰시오.\n\n{q.code}"

    print(f"[출제] {qtype} (재출제 {state['retry_count']}회)")
    return {
        "context": context,
        "question_type": qtype,
        "question": question,
        "answer": q.answer,
        "key_points": q.key_points,
        "language": q.language.lower(),
        "code": q.code,
    }
