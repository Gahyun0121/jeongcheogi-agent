"""코드 문제의 정답을 확정하려고 실제로 코드를 실행하는 도구."""
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.tools import tool

TIMEOUT = 10  # 무한 루프 방지 (초)


@tool
def run_code(language: str, code: str) -> str:
    """C, Java, Python 코드를 실제로 실행하고 출력값을 돌려준다.
    language: "c", "java", "python" 중 하나.
    Java는 public class Main 안에 main 메서드를 둔다."""
    language = language.lower()

    # 임시 폴더에서 실행하고, 끝나면 자동 삭제
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        if language == "c":
            src = tmp / "main.c"
            src.write_text(code, encoding="utf-8")
            compiled = subprocess.run(
                ["gcc", "-Wall", str(src), "-o", str(tmp / "main")],
                capture_output=True, text=True, timeout=TIMEOUT,
            )

            if compiled.returncode != 0:
                return "컴파일 에러:\n" + compiled.stderr

            if "warning" in compiled.stderr:
                return "컴파일 경고 (답이 모호한 코드):\n" + compiled.stderr
            
            cmd = [str(tmp / "main")]


        elif language == "java":
            src = tmp / "Main.java"
            src.write_text(code, encoding="utf-8")
            cmd = ["java", str(src)]  # javac 없이 바로 실행

        elif language == "python":
            src = tmp / "main.py"
            src.write_text(code, encoding="utf-8")
            cmd = [sys.executable, str(src)]

        else:
            return f"지원하지 않는 언어: {language}"

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return f"시간 초과 ({TIMEOUT}초)"

        if result.returncode != 0:
            return "실행 에러:\n" + result.stderr
        return result.stdout


if __name__ == "__main__":
    print(run_code.invoke({"language": "python", "code": "a = [1, 2, 3, 4, 5]\nprint(a[::-2])"}))
