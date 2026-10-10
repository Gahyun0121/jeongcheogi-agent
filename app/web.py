"""Streamlit 화면: 터미널 대신 브라우저에서 문제를 풀고, 개념 자료를 본다.

실행: uv run streamlit run app/web.py
그래프와 노드는 main.py와 똑같이 쓰고, 입출력만 화면으로 바꾼다.
"""
import asyncio
import contextlib
import html
import io
import random
import sys
import uuid
from datetime import date
from pathlib import Path

import streamlit as st
from langgraph.types import Command

sys.path.append(str(Path(__file__).resolve().parent.parent))  # streamlit run은 프로젝트 루트를 모른다
from app import exam_info  # noqa: E402
from app.graph import build_graph  # noqa: E402
from app.notion import get_wrong_records  # noqa: E402
from app.rag import load_documents  # noqa: E402

TYPE_COLOR = {"단답형": "blue", "약술형": "green", "코드": "orange"}

st.set_page_config(page_title="정처기 실기 약점 맞춤 출제", page_icon="📝", layout="centered")
st.markdown(
    """<style>
    .question {font-size: 1.1rem; line-height: 1.75; font-weight: 500; margin: 0.5rem 0 1rem;}
    .question.prev {font-size: 1rem; font-weight: 400; margin: 0.2rem 0 0.6rem;}

    /* 배경: 아주 연한 파스텔 그라데이션 */
    .stApp {background: linear-gradient(160deg, #F0FAF7 0%, #F1F8FF 55%, #F5F7FF 100%) fixed;}
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #E3F6EF 0%, #E6F1FF 100%);}
    [data-testid="stHeader"] {background: transparent;}

    /* 홈 배너: 민트 → 하늘 → 연한 파랑 (차가운 톤) */
    .hero {padding: 2rem 1.8rem; border-radius: 22px; color: #2E2A3A; margin-bottom: 1.2rem;
           background: linear-gradient(120deg, #BDEFDD 0%, #C4E4FF 55%, #CBD5FF 100%);
           box-shadow: 0 10px 30px rgba(120,160,210,0.22);}
    .hero .eyebrow {font-size: 0.85rem; opacity: 0.7; letter-spacing: 0.04em; font-weight: 600;}
    .hero h1 {color: #2E2A3A; font-size: 1.9rem; margin: 0.3rem 0 0.5rem; padding: 0;}
    .hero p {margin: 0; opacity: 0.8; line-height: 1.6;}
    .hero .chip {display: inline-block; margin-top: 1rem; padding: 0.35rem 0.9rem; border-radius: 999px;
                 background: rgba(255,255,255,0.7); font-weight: 700; font-size: 0.9rem;}

    /* 페이지 위 공지 배너 */
    .notice {padding: 0.6rem 1rem; border-radius: 14px; margin-bottom: 1rem; font-size: 0.92rem;
             background: linear-gradient(90deg, #DDF5EC 0%, #E1EEFF 100%); border: 1px solid rgba(110,170,210,0.35);}

    /* 동작 단계 카드: 단계마다 다른 파스텔 */
    .step {padding: 1rem; border-radius: 16px; height: 100%; border: 1px solid rgba(0,0,0,0.04);}
    .step.s1 {background: #DDF5EC;}
    .step.s2 {background: #DCF0FB;}
    .step.s3 {background: #E0EAFF;}
    .step.s4 {background: #E4E6FB;}
    .step .num {font-size: 0.8rem; font-weight: 700; opacity: 0.6;}
    .step .title {font-weight: 700; margin: 0.2rem 0 0.3rem;}
    .step .desc {font-size: 0.85rem; opacity: 0.8; line-height: 1.5;}
    </style>""",
    unsafe_allow_html=True,
)
ss = st.session_state


# ---------- 그래프 실행 ----------

def run_graph(payload) -> None:
    """그래프를 다음 interrupt까지 실행하고, 노드가 print한 로그를 모아 둔다."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ss.result = asyncio.run(ss.graph.ainvoke(payload, ss.config))
    log = buf.getvalue().strip()

    # 로그를 "직전 문제 채점" 부분과 "다음 문제 출제" 부분으로 나눈다
    grade_log, sep, gen_log = log.partition("[약점 분석]")
    ss.grade_log = grade_log.strip()
    if sep:
        ss.gen_log = (sep + gen_log).strip()


def graph_values() -> dict:
    """checkpointer에 저장된 현재 State."""
    return ss.graph.get_state(ss.config).values


def interrupt_kind() -> str | None:
    """그래프가 멈춘 이유: "question"(답 입력), "confirm"(같은 뜻 확인), None(종료)."""
    if "__interrupt__" not in ss.result:
        return None
    return ss.result["__interrupt__"][0].value["kind"]


def snapshot() -> dict:
    """채점 결과를 보여주려고, 답을 내기 직전의 문제 정보를 저장해 둔다."""
    v = graph_values()
    return {k: v.get(k) for k in ["question", "code", "language", "question_type", "keyword"]}


def start(total: int) -> None:
    ss.graph = build_graph()
    ss.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    ss.total = total
    ss.gen_log = ss.grade_log = ""
    with st.spinner("첫 문제를 만드는 중..."):
        run_graph({"solved": 0, "total": total, "correct_count": 0, "recent_keywords": []})
    ss.phase = "question"
    wrong_counts.clear()  # 개념 정리 페이지의 오답 횟수를 새로 불러오게


def submit_answer(answer: str) -> None:
    ss.prev = snapshot()
    ss.prev["user_answer"] = answer
    before = graph_values()["correct_count"]
    with st.spinner("채점하고 다음 문제를 만드는 중..."):
        run_graph(Command(resume=answer))

    if interrupt_kind() == "confirm":
        ss.phase = "confirm"  # 오답: 같은 뜻인지 사람이 확인
        ss.prev["answer"] = ss.result["__interrupt__"][0].value["answer"]
    else:
        values = graph_values()
        ss.prev["is_correct"] = values["correct_count"] > before
        ss.phase = "feedback"


def confirm(same: bool) -> None:
    first_log = ss.grade_log  # 처음 채점한 기록도 같이 보여주려고 남겨 둔다
    with st.spinner("다음 문제를 만드는 중..."):
        run_graph(Command(resume="y" if same else ""))
    ss.grade_log = f"{first_log}\n{ss.grade_log}".strip()
    ss.prev["is_correct"] = same
    ss.prev["confirmed"] = True
    ss.phase = "feedback"


@st.cache_data(ttl=60, show_spinner="노션에서 오답 기록을 불러오는 중...")
def wrong_counts() -> dict[str, int]:
    """키워드별 오답 횟수 (노션 MCP). 실패하면 빈 dict."""
    try:
        records = asyncio.run(get_wrong_records())
    except Exception:
        return {}
    return {k: len(v) for k, v in records.items()}


# ---------- 공통: 배너, 사이드바 ----------

CHEERS = [
    "오늘도 한 문제만 더! 💪",
    "틀린 문제가 곧 합격 포인트예요 🔑",
    "약점은 노션이 기억해요. 나는 풀기만 🧠",
    "코드 결과값은 직접 손으로 따라가 보기 ✍️",
    "기순교절시논우, 내공유제스스! 🔁",
]


def dday() -> str | None:
    """가장 가까운 실기 시험까지 남은 날 (예: "3회 실기 D-14")."""
    nxt = exam_info.next_practical(date.today())
    if nxt is None:
        return None
    name, start_day, _ = nxt
    d = (start_day - date.today()).days
    return f"{name} 실기 D-{d}" if d > 0 else f"{name} 실기 시험 기간"


def notice_banner() -> None:
    """페이지 맨 위 얇은 공지 배너."""
    text = f"📢 <b>{dday()}</b> · {CHEERS[date.today().toordinal() % len(CHEERS)]}" if dday() else "📢 올해 실기 시험이 끝났어요. 다음 해 일정을 기다려요!"
    st.markdown(f'<div class="notice">{text}</div>', unsafe_allow_html=True)


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📝 정처기 실기 에이전트")
        st.caption("약점을 찾아 맞춤 출제하는 학습 에이전트")
        if dday():
            st.badge(dday(), icon="⏰", color="red")
        st.markdown(f"> {random.choice(CHEERS)}")

        if ss.get("phase") not in (None,):
            v = graph_values()
            st.markdown(f"**이번 풀이** {v.get('correct_count', 0)} / {v.get('solved', 0)} 정답")

        st.divider()
        st.caption("사용 기술")
        st.markdown(
            ":blue-badge[LangGraph] :violet-badge[RAG · Chroma] :green-badge[Notion MCP] "
            ":orange-badge[코드 실행 도구] :red-badge[HITL] :gray-badge[OpenAI]"
        )
        st.divider()
        st.caption("[GitHub 저장소](https://github.com/Gahyun0121/jeongcheogi-agent)")


# ---------- 홈 ----------

STEPS = [
    ("1", "약점 분석", "노션 오답노트에서 많이 틀린 키워드를 골라요"),
    ("2", "출제 (RAG)", "개념 자료를 검색해서 그 안에서만 문제를 내요"),
    ("3", "검수", "자료와 맞는지 검사하고, 코드는 직접 실행해 정답을 확정해요"),
    ("4", "채점 · 오답노트", "틀리면 노션에 저장돼 다음에 더 자주 나와요"),
]


def home_page() -> None:
    chip = f'<span class="chip">⏰ {dday()}</span>' if dday() else ""
    st.markdown(
        f"""<div class="hero">
        <div class="eyebrow">정보처리기사 실기 · 약점 맞춤 출제 에이전트</div>
        <h1>틀린 만큼, 똑똑하게 다시 출제해요</h1>
        <p>문제를 풀수록 약점 키워드를 찾아 노션에 쌓고,<br>그 키워드 위주로 단답형·약술형·코드 문제를 만들어요.</p>
        {chip}
        </div>""",
        unsafe_allow_html=True,
    )

    if st.button("📝 문제 풀러 가기", type="primary", width="stretch"):
        st.switch_page(PAGES["quiz"])

    counts = wrong_counts()
    docs = load_documents()
    c1, c2, c3 = st.columns(3)
    c1.metric("누적 오답", f"{sum(counts.values())}개", border=True)
    c2.metric("약점 키워드", f"{len(counts)}개", border=True)
    c3.metric("개념 키워드", f"{len(docs)}개", border=True)

    st.markdown("#### 이렇게 동작해요")
    for col, (num, title, desc) in zip(st.columns(4), STEPS):
        col.markdown(
            f'<div class="step s{num}"><div class="num">STEP {num}</div>'
            f'<div class="title">{title}</div><div class="desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("##### 🔥 내 약점 TOP 3")
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        if top:
            for i, (k, n) in enumerate(top, 1):
                st.markdown(f"**{i}.** {k} &nbsp; :red-badge[{n}회]")
        else:
            st.caption("아직 오답이 없어요. 문제를 풀면 여기에 쌓여요.")
        if st.button("개념 정리 보기", width="stretch"):
            st.switch_page(PAGES["concept"])

    with right.container(border=True):
        today_doc = docs[date.today().toordinal() % len(docs)]  # 날짜마다 바뀌는 오늘의 개념
        st.markdown("##### 💡 오늘의 개념")
        st.markdown(f"**{today_doc.metadata['keyword']}** &nbsp; :gray-badge[{today_doc.metadata['category']}]")
        lines = [ln for ln in today_doc.page_content.splitlines()[1:] if not ln.startswith("- 빈도")]
        st.markdown("\n".join(lines[:4]))


# ---------- 문제 풀기 화면 조각 ----------

def question_text(text: str, prev: bool = False) -> None:
    """문제 문장을 본문 크기로 보여준다. (마크다운 제목처럼 커지지 않게)"""
    body = html.escape(text).replace("\n", "<br>")
    st.markdown(f'<div class="question{" prev" if prev else ""}">{body}</div>', unsafe_allow_html=True)


def header() -> None:
    v = graph_values()
    solved, correct = v.get("solved", 0), v.get("correct_count", 0)
    current = solved + 1 if ss.phase == "question" else solved
    st.progress(min(solved / ss.total, 1.0))
    c1, c2, c3 = st.columns(3)
    c1.metric("진행", f"{current} / {ss.total}")
    c2.metric("정답", correct)
    c3.metric("오답", solved - correct)


def start_card(title: str = "몇 문제 풀까요?", button: str = "시작하기") -> None:
    with st.container(border=True):
        st.markdown(f"##### {title}")
        total = st.slider("문제 수", min_value=1, max_value=20, value=5, label_visibility="collapsed")
        st.caption("틀린 문제는 노션 오답노트에 저장되고, 다음에 더 자주 나와요.")
        if st.button(button, type="primary", width="stretch"):
            start(total)
            st.rerun()


def question_card() -> None:
    v = graph_values()
    qtype = v["question_type"]
    past = len(v.get("past_titles") or [])
    tag = f":red-background[이전 오답 {past}회]" if past else ":gray-background[처음 보는 키워드]"
    with st.container(border=True):
        st.markdown(f":{TYPE_COLOR[qtype]}-background[{qtype}] &nbsp; **{v['keyword']}** &nbsp; {tag}")
        question_text(v["question"])
        if v.get("code"):
            st.code(v["code"], language=v["language"])

        with st.form("answer_form", clear_on_submit=True, border=False):
            if qtype == "약술형":
                answer = st.text_area("답", placeholder="1~2문장으로 설명하세요 (⌘+Enter로 제출)", height=100)
            else:
                answer = st.text_input("답", placeholder="답을 입력하고 Enter")
            if st.form_submit_button("제출", type="primary", width="stretch"):
                submit_answer(answer)
                st.rerun()

    if ss.gen_log:
        with st.expander("출제 과정 보기 (약점 분석 → 출제 → 검수)"):
            st.code(ss.gen_log, language=None)


def prev_question_box() -> None:
    p = ss.prev
    with st.container(border=True):
        st.caption(f"{p['question_type']} · {p['keyword']}")
        question_text(p["question"], prev=True)
        if p.get("code"):
            st.code(p["code"], language=p["language"])
        st.markdown(f"내 답: `{p['user_answer'] or '(빈 답)'}`")


def confirm_card() -> None:
    p = ss.prev
    prev_question_box()
    st.error(f"**오답**\n\n정답: {p['answer']}")
    st.write("정답과 **같은 뜻**으로 썼나요? (표현만 다르거나 순서만 다른 경우)")
    c1, c2 = st.columns(2)
    if c1.button("같은 뜻이에요 → 정답 처리", width="stretch"):
        confirm(True)
        st.rerun()
    if c2.button("틀렸어요 → 오답노트 저장", type="primary", width="stretch"):
        confirm(False)
        st.rerun()


def feedback_card() -> None:
    p = ss.prev
    prev_question_box()
    if p["is_correct"]:
        st.success("**정답** (같은 뜻으로 인정)" if p.get("confirmed") else "**정답**")
    else:
        st.error("**오답** · 노션 오답노트에 저장했어요")

    if ss.grade_log:
        with st.expander("채점 기록 보기"):
            st.code(ss.grade_log, language=None)

    done = interrupt_kind() is None
    label = "결과 보기" if done else "다음 문제 →"
    if st.button(label, type="primary", width="stretch"):
        ss.phase = "done" if done else "question"
        st.rerun()


def done_card() -> None:
    correct = graph_values()["correct_count"]
    if correct == ss.total:
        st.balloons()
    with st.container(border=True):
        st.markdown("#### 결과")
        st.markdown(f"### {ss.total}문제 중 {correct}개 정답")
        if correct < ss.total:
            st.write("틀린 문제는 노션 오답노트에 저장됐어요. 다음에 시작하면 약점 키워드가 더 자주 나와요.")
        else:
            st.write("모두 맞혔어요! 문제 수를 늘려서 다시 도전해 보세요.")
    start_card("다시 풀어볼까요?", "다시 시작")


def quiz_page() -> None:
    notice_banner()
    st.title("📝 정처기 실기 약점 맞춤 출제")
    st.caption("틀린 키워드를 노션에 쌓고, 약점 위주로 다시 출제해요.")

    with st.sidebar:
        if ss.get("phase") not in (None, "done") and st.button("처음으로", width="stretch"):
            ss.phase = None
            st.rerun()
        with st.expander("채점 기준", expanded=True):
            st.markdown(
                "- **단답형**: 띄어쓰기 무시, 정확히 일치\n"
                "- **약술형**: 핵심 단어 2개 이상 포함\n"
                "- **코드**: 실제 실행 결과와 일치\n\n"
                "오답이면 같은 뜻인지 직접 확인해요. (코드 제외)"
            )

    phase = ss.get("phase")
    if phase is None:
        start_card()
        return

    header()
    if phase == "question":
        question_card()
    elif phase == "confirm":
        confirm_card()
    elif phase == "feedback":
        feedback_card()
    elif phase == "done":
        done_card()


# ---------- 개념 정리 페이지 ----------

def concept_page() -> None:
    notice_banner()
    st.title("📚 개념 정리")
    st.caption("출제에 쓰는 개념 자료예요. 노션 오답노트의 오답 횟수를 같이 보여줘요.")

    counts = wrong_counts()
    docs = load_documents()

    # 내 약점 TOP 5
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    with st.container(border=True):
        st.markdown("##### 🔥 내 약점 TOP 5")
        if top:
            st.markdown(" &nbsp; ".join(f":red-background[{k} · {n}회]" for k, n in top))
        else:
            st.write("아직 오답 기록이 없어요. 문제를 풀면 여기에 쌓여요.")

    c1, c2 = st.columns([3, 1])
    query = c1.text_input("검색", placeholder="키워드 검색 (예: 정규화, LRU)", label_visibility="collapsed")
    weak_only = c2.toggle("약점만")

    categories = list(dict.fromkeys(d.metadata["category"] for d in docs))  # 파일 순서 유지
    for tab, category in zip(st.tabs(categories), categories):
        with tab:
            shown = 0
            for d in docs:
                keyword = d.metadata["keyword"]
                n = counts.get(keyword, 0)
                if d.metadata["category"] != category:
                    continue
                if query and query.lower() not in d.page_content.lower():
                    continue
                if weak_only and n == 0:
                    continue
                badge = f" &nbsp; :red-background[오답 {n}회]" if n else ""
                with st.expander(f"**{keyword}**{badge}"):
                    st.markdown(d.page_content.split("\n", 1)[1])  # 첫 줄(## 제목)은 빼고
                shown += 1
            if shown == 0:
                st.caption("조건에 맞는 키워드가 없어요.")


# ---------- 시험 정보 페이지 ----------

def exam_page() -> None:
    notice_banner()
    st.title("📅 시험 정보")
    st.caption(f"2026년 정보처리기사 · {exam_info.CHECKED_ON} 확인 · 일정은 바뀔 수 있으니 [큐넷]({exam_info.QNET_URL})에서 꼭 확인하세요.")

    today = date.today()
    nxt = exam_info.next_practical(today)
    with st.container(border=True):
        if nxt is None:
            st.markdown("##### 올해 실기 시험이 모두 끝났어요.")
            st.write(f"다음 해 일정은 보통 11~12월에 공고돼요. [큐넷]({exam_info.QNET_URL})에서 확인하세요.")
        else:
            name, start_day, end_day = nxt
            d = (start_day - today).days
            dday = f"D-{d}" if d > 0 else "시험 기간"
            c1, c2 = st.columns([1, 2])
            c1.metric(f"{name} 실기", dday)
            c2.markdown(
                f"**실기 시험 기간** {start_day:%m.%d} ~ {end_day:%m.%d}\n\n"
                "정보처리기사 필답형은 이 기간 중 **하루**에 치러요. 정확한 날짜·장소는 **수험표**에서 확인하세요."
            )

    st.markdown("##### 회차별 일정")
    rows = [
        {
            "회차": r[0], "필기 접수": r[1], "필기 시험": r[2], "필기 발표": r[3],
            "실기 접수": r[4], "실기 시험": f"{r[5]:%m.%d}~{r[6]:%m.%d}", "최종 발표": r[7],
        }
        for r in exam_info.SCHEDULE
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("최종 발표는 1차 / 2차 발표일이에요.")

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("##### ✍️ 실기 시험 방식")
        st.markdown("\n".join(f"- **{k}**: {v}" for k, v in exam_info.PRACTICAL.items()))
    with c2.container(border=True):
        st.markdown("##### 🎒 준비물")
        st.markdown("\n".join(f"- {s}" for s in exam_info.SUPPLIES))
        st.caption("수험자 유의사항은 큐넷 공지에서 꼭 확인하세요.")
        st.markdown("##### 💳 응시료")
        st.markdown("\n".join(f"- **{k}**: {v}" for k, v in exam_info.FEES.items()))


# ---------- 페이지 연결 ----------

PAGES = {
    "home": st.Page(home_page, title="홈", icon="🏠", default=True),
    "quiz": st.Page(quiz_page, title="문제 풀기", icon="📝"),
    "concept": st.Page(concept_page, title="개념 정리", icon="📚"),
    "exam": st.Page(exam_page, title="시험 정보", icon="📅"),
}
page = st.navigation(list(PAGES.values()))
sidebar()
page.run()
