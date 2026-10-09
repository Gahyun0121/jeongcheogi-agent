from typing import TypedDict


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
