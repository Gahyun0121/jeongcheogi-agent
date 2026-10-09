from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.nodes import (
    QuizState,
    analyze_weakness,
    ask_answer,
    confirm_answer,
    generate_question,
    grade_answer,
    review_question,
    save_wrong,
)

MAX_RETRY = 3  # 검수 탈락 허용 횟수 (넘으면 다른 키워드로)


def route_after_review(state: QuizState) -> str:
    """검수 결과에 따라 다음 노드를 고른다."""
    if state["review_ok"]:
        return "ask_answer"
    if state["retry_count"] < MAX_RETRY:
        return "generate_question"
    print(f"[검수] {MAX_RETRY}번 탈락 → 다른 키워드로")
    return "analyze_weakness"


def check_done(state: QuizState) -> str:
    """목표 문제 수를 다 풀었으면 종료한다."""
    if state["solved"] >= state["total"]:
        return END
    return "analyze_weakness"


def route_after_grade(state: QuizState) -> str:
    """맞으면 종료 조건 확인, 틀리면 사람 확인 (코드 문제는 바로 오답 저장)."""
    if state["is_correct"]:
        return check_done(state)
    if state["question_type"] == "코드":
        return "save_wrong"  # 출력값은 정확해야 해서 확인하지 않는다
    return "confirm_answer"


def route_after_confirm(state: QuizState) -> str:
    """사람이 같은 뜻이라고 하면 정답, 아니면 오답 저장."""
    if state["is_correct"]:
        return check_done(state)
    return "save_wrong"



def build_graph():
    graph = StateGraph(QuizState)

    graph.add_node("analyze_weakness", analyze_weakness)
    graph.add_node("generate_question", generate_question)
    graph.add_node("review_question", review_question)
    graph.add_node("ask_answer", ask_answer)
    graph.add_node("grade_answer", grade_answer)
    graph.add_node("save_wrong", save_wrong)
    graph.add_node("confirm_answer", confirm_answer)

    graph.add_edge(START, "analyze_weakness")
    graph.add_edge("analyze_weakness", "generate_question")
    graph.add_edge("generate_question", "review_question")
    graph.add_conditional_edges(
        "review_question", route_after_review,
        ["ask_answer", "generate_question", "analyze_weakness"],
    )
    graph.add_edge("ask_answer", "grade_answer")
    graph.add_conditional_edges(
        "grade_answer", route_after_grade,
        ["confirm_answer", "save_wrong", "analyze_weakness", END],
    )
    graph.add_conditional_edges(
        "confirm_answer", route_after_confirm,
        ["save_wrong", "analyze_weakness", END],
    )
    graph.add_conditional_edges("save_wrong", check_done, ["analyze_weakness", END])

    # interrupt로 멈춘 지점을 저장해야 답 입력 후 이어서 실행할 수 있다
    return graph.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    print(build_graph().get_graph().draw_mermaid())
