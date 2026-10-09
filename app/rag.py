import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CONCEPTS_DIR = BASE_DIR / "data" / "concepts"
DB_DIR = BASE_DIR / "data" / "chroma_db"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def load_documents() -> list[Document]:
    """md 파일을 '## 제목' 단위로 나눠 Document 목록을 만든다."""
    docs = []
    for path in sorted(CONCEPTS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")

        # frontmatter에서 category 꺼내기
        _, front, body = text.split("---", 2)
        category = ""
        for line in front.splitlines():
            if line.startswith("category:"):
                category = line.split(":", 1)[1].strip()

        # '## ' 기준으로 나누기 (첫 조각은 '# 제목'이라 버림)
        for section in body.split("\n## ")[1:]:
            keyword = section.splitlines()[0].strip()
            docs.append(
                Document(
                    page_content="## " + section.strip(),
                    metadata={"category": category, "keyword": keyword, "source": path.name},
                )
            )
    return docs


def build_db() -> None:
    """기존 DB를 지우고 새로 저장한다."""
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)  # 다시 실행해도 중복 저장 안 되게
    docs = load_documents()
    Chroma.from_documents(docs, embeddings, persist_directory=str(DB_DIR))
    print(f"저장 완료: {len(docs)}개")


def search(query: str, k: int = 3) -> list[Document]:
    """질문과 가까운 개념 자료 k개를 찾는다."""
    db = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
    return db.similarity_search(query, k=k)


if __name__ == "__main__":
    build_db()
    for i, doc in enumerate(search("페이지 교체"), 1):
        print(i, doc.metadata["keyword"])
