import json
import os
from datetime import date

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

DATA_SOURCE_ID = os.environ["NOTION_DATA_SOURCE_ID"]

# 노션 공식 MCP 서버를 npx로 실행해서 연결
client = MultiServerMCPClient({
    "notion": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@notionhq/notion-mcp-server"],
        "env": {"NOTION_TOKEN": os.environ["NOTION_TOKEN"], "PATH": os.environ["PATH"]},
    }
})


async def _call(tool_name: str, args: dict) -> dict:
    """MCP 도구를 이름으로 찾아 실행하고, 결과 JSON을 dict로 돌려준다."""
    tools = await client.get_tools()
    tool = next(t for t in tools if t.name == tool_name)
    result = await tool.ainvoke(args)
    text = result[0]["text"] if isinstance(result, list) else result
    data = json.loads(text)
    if data.get("object") == "error":
        raise RuntimeError(f"노션 에러: {data.get('message')}")
    return data



async def get_wrong_records() -> dict[str, list[str]]:
    """키워드별로 틀린 문제 제목을 모은다. (목록 길이 = 오답 횟수)"""
    data = await _call("API-query-data-source", {"data_source_id": DATA_SOURCE_ID})
    records = {}
    for page in data["results"]:
        props = page["properties"]
        select = props["키워드"]["select"]
        if select:
            title = "".join(t["plain_text"] for t in props["문제"]["title"])
            records.setdefault(select["name"], []).append(title)
    return records



def _text(value: str) -> list[dict]:
    """노션 텍스트 형식으로 바꾼다. (노션 제한: 2000자)"""
    # 노션 MCP 서버가 "[1, 2]" 같은 JSON 모양 글자를 리스트로 바꿔버려서,
    # 앞에 안 보이는 글자(zero-width space)를 붙여 글자로 유지한다
    if value.strip().startswith(("[", "{")):
        value = "\u200b" + value
    return [{"text": {"content": value[:2000]}}]



async def save_wrong_answer(state: dict) -> None:
    """틀린 문제를 오답 DB에 저장한다."""
    # 페이지 본문: 문제 문장 + (코드 문제면) 코드 블록
    children = [{"type": "paragraph", "paragraph": {"rich_text": _text(state["question"])}}]
    if state["code"]:
        children.append({"type": "code", "code": {"language": state["language"], "rich_text": _text(state["code"])}})

    await _call("API-post-page", {
        "parent": {"data_source_id": DATA_SOURCE_ID},
        "properties": {
            "문제": {"title": _text(state["title"])},
            "키워드": {"select": {"name": state["keyword"]}},
            "과목": {"select": {"name": state["category"]}},
            "유형": {"select": {"name": state["question_type"]}},
            "정답": {"rich_text": _text(state["answer"])},
            "내 답": {"rich_text": _text(state["user_answer"])},
            "날짜": {"date": {"start": date.today().isoformat()}},
        },
        "children": children,
    })



if __name__ == "__main__":
    import asyncio
    print(asyncio.run(get_wrong_records()))
