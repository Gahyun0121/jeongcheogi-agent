import asyncio

from langgraph.types import Command

from app.graph import build_graph

TOTAL = 5  # 풀 문제 수 (종료 조건)


async def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": "quiz"}}  # checkpointer가 멈춘 지점을 찾는 이름

    result = await graph.ainvoke({
        "solved": 0, 
        "total": TOTAL, 
        "correct_count": 0, 
        "recent_keywords": []
    }, config)

    # interrupt로 멈출 때마다 문제를 보여주고 답을 받아 이어서 실행
    while "__interrupt__" in result:
        info = result["__interrupt__"][0].value

        if info["kind"] == "confirm":
            # 오답일 때: 같은 뜻으로 썼는지 사람이 확인
            answer = input("같은 뜻으로 썼다면 y, 아니면 엔터: ")
        else:
            print(f"\n===== 문제 {result['solved'] + 1}/{TOTAL} ({info['type']}) =====")
            print(info["question"])
            if info["code"]:
                print("\n" + info["code"])
            answer = input("\n답: ")

        result = await graph.ainvoke(Command(resume=answer), config)

    print(f"\n===== 결과: {TOTAL}문제 중 {result['correct_count']}개 정답 =====")


if __name__ == "__main__":
    asyncio.run(main())
