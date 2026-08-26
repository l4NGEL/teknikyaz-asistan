"""TeknikYaz sohbet arayüzü.

Demo mod GPU ve Chroma istemez; canlı mod `dialogue.chat` üzerinden ajan grafını çağırır.

    streamlit run src/serving/chat_app.py
"""
from __future__ import annotations

import uuid

import streamlit as st

from src.serving.demo_catalog import DEMO_EXAMPLES, lookup_demo

st.set_page_config(page_title="TeknikYaz", page_icon="📄", layout="wide")

BUCKET_COLOR = {"yüksek": "#16a34a", "orta": "#d97706", "düşük": "#dc2626"}


def _init_state() -> None:
    st.session_state.setdefault("mode", "demo")
    st.session_state.setdefault("session_id", f"ui-{uuid.uuid4().hex[:8]}")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending", None)


def _reset() -> None:
    st.session_state.session_id = f"ui-{uuid.uuid4().hex[:8]}"
    st.session_state.messages = []
    st.session_state.pending = None


def _run_live(message: str) -> dict:
    from src.nlp_tasks.dialogue import chat as run_chat

    result = run_chat(st.session_state.session_id, message)
    return {
        "answer": result.get("answer") or "",
        "sources": result.get("sources") or [],
        "confidence": result.get("confidence"),
        "confidence_bucket": result.get("confidence_bucket"),
        "needs_human_review": bool(result.get("needs_human_review")),
        "review_reason": result.get("review_reason"),
        "draft_answer": result.get("draft_answer"),
        "agent_trace": result.get("agent_trace") or [],
        "demo": False,
    }


def _answer(message: str) -> dict:
    if st.session_state.mode == "demo":
        return lookup_demo(message)
    try:
        return _run_live(message)
    except Exception as exc:
        return {
            "answer": (
                "Canlı mod model veya vektör indeksi olmadan çalışmadı. "
                "Kenardaki Demo moda geçin. "
                f"Ayrıntı: {exc}"
            ),
            "sources": [],
            "confidence": None,
            "confidence_bucket": None,
            "needs_human_review": False,
            "review_reason": None,
            "draft_answer": None,
            "agent_trace": [],
            "demo": False,
        }


def _render_trace(trace: list[str]) -> None:
    if not trace:
        return
    st.caption("Ajan izi: " + " → ".join(trace))


def _render_assistant(payload: dict) -> None:
    if payload.get("needs_human_review"):
        st.warning("İnsan onayı gerekli" + (f" — {payload.get('review_reason')}" if payload.get("review_reason") else ""))
    st.markdown(payload.get("answer") or "")

    bucket = payload.get("confidence_bucket")
    conf = payload.get("confidence")
    if bucket or conf is not None:
        color = BUCKET_COLOR.get(bucket or "", "#64748b")
        label = bucket or "—"
        score = f"{conf:.2f}" if isinstance(conf, int | float) else "—"
        st.markdown(
            f"<span style='background:{color};color:white;padding:2px 8px;"
            f"border-radius:999px;font-size:0.85rem'>güven: {label} ({score})</span>",
            unsafe_allow_html=True,
        )
        if isinstance(conf, int | float):
            st.progress(min(max(float(conf), 0.0), 1.0))

    _render_trace(list(payload.get("agent_trace") or []))

    sources = payload.get("sources") or []
    if sources:
        with st.expander("Kaynaklar"):
            for src in sources:
                title = src.get("title") or "kaynak"
                url = src.get("url") or ""
                st.markdown(f"- [{title}]({url})" if url else f"- {title}")

    if payload.get("draft_answer"):
        with st.expander("Onay bekleyen taslak"):
            st.write(payload["draft_answer"])


_init_state()

with st.sidebar:
    st.header("TeknikYaz")
    st.caption("Hiyerarşik hibrit RAG · LangGraph · insan kapısı")
    mode = st.radio(
        "Mod",
        options=("demo", "live"),
        format_func=lambda v: "Demo (GPU yok)" if v == "demo" else "Canlı (ajan grafı)",
        key="mode",
    )
    if mode == "demo":
        st.info("Sabit turlar. Router + güven + HITL görünür; model inmez.")
    else:
        st.warning("Chroma indeksi ve yerel LLM gerekir. Yoksa otomatik Demo'ya düşmez; hata mesajı gelir.")
    st.caption(f"Oturum `{st.session_state.session_id}`")
    if st.button("Sohbeti sıfırla"):
        _reset()
        st.rerun()
    st.subheader("Örnek sorular")
    for example in DEMO_EXAMPLES:
        if st.button(example, key=f"ex-{example}"):
            st.session_state.pending = example
            st.rerun()

st.title("Teknik dokümantasyon asistanı")
st.caption("Cevap, kaynak, güven kovası ve gerekirse insan onayı aynı turda görünür.")

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        if item["role"] == "user":
            st.markdown(item["content"])
        else:
            _render_assistant(item["payload"])

prompt = st.chat_input("README, API, kurulum veya mimari hakkında sorun")
if st.session_state.pending:
    prompt = st.session_state.pending
    st.session_state.pending = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    payload = _answer(prompt)
    st.session_state.messages.append({"role": "assistant", "payload": payload})
    with st.chat_message("assistant"):
        _render_assistant(payload)
