"""Çok ajanlı orkestrasyon: yönlendirici → retrieval → yazar → eleştirmen → konsensüs → insan kapısı.

LangGraph varsa StateGraph olarak derlenir; yoksa aynı düğümler sırayla çalışır.
Düğümler bağımlılık enjekte edilebilir (retrieve/generate/groundedness) — CI ve birim
testleri GPU/model indirmeden grafı doğrular.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from src.agents.conflict import detect_conflicts
from src.agents.consensus import consensus_answer, conservative_rewrite
from src.agents.router import route
from src.config import AGENT_CONFIG
from src.xai.confidence import combine_confidence, lexical_groundedness, should_escalate

try:
    from langgraph.graph import END, START, StateGraph

    HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover - CI langgraph kurulu olacak
    HAS_LANGGRAPH = False
    END = "END"
    START = "START"
    StateGraph = None  # type: ignore[misc, assignment]


class AgentState(TypedDict, total=False):
    question: str
    route: str
    contexts: list[dict]
    writer_answer: str
    critic_answer: str
    conflicts: list[dict]
    groundedness: float
    confidence: float
    confidence_bucket: str
    needs_human_review: bool
    review_reason: str | None
    answer: str
    draft_answer: str | None
    sources: list[dict]
    agent_trace: list[str]


@dataclass
class AgentDeps:
    retrieve_fn: Callable[..., list[dict]]
    generate_fn: Callable[..., str]
    groundedness_fn: Callable[..., float]
    flag_unsupported_fn: Callable[..., list[dict]] = field(default_factory=lambda: (lambda *_a, **_k: []))
    confidence_threshold: float = AGENT_CONFIG.confidence_threshold


def default_deps() -> AgentDeps:
    from src.rag.rag_pipeline import generate
    from src.rag.retriever import retrieve
    from src.xai.hallucination_check import flag_unsupported_sentences, groundedness_score

    def _generate(messages: list[dict], **kwargs: Any) -> str:
        return generate(messages, **kwargs)

    def _retrieve(question: str, final_k: int = 5, **kwargs: Any) -> list[dict]:
        return retrieve(question, final_k=final_k, **kwargs)

    def _groundedness(answer: str, contexts: list[str]) -> float:
        if AGENT_CONFIG.use_nli_groundedness:
            return groundedness_score(answer, contexts)
        return lexical_groundedness(answer, contexts)

    return AgentDeps(
        retrieve_fn=_retrieve,
        generate_fn=_generate,
        groundedness_fn=_groundedness,
        flag_unsupported_fn=flag_unsupported_sentences,
    )


def _trace(state: AgentState, node: str) -> list[str]:
    return list(state.get("agent_trace") or []) + [node]


def _context_texts(contexts: list[dict]) -> list[str]:
    return [(c.get("parent_text") or c.get("text") or "") for c in contexts]


def _sources(contexts: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out = []
    for ctx in contexts:
        item = {"title": ctx.get("doc_title") or "", "url": ctx.get("doc_url") or ""}
        key = (item["title"], item["url"])
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def node_router(state: AgentState, deps: AgentDeps) -> dict:
    decision = route(state.get("question") or "")
    return {"route": decision, "agent_trace": _trace(state, f"router:{decision}")}


def node_refuse(state: AgentState, deps: AgentDeps) -> dict:
    return {
        "answer": "Bu soru teknik dokümantasyon kapsamının dışında. README, API, kurulum veya mimari belgeler hakkında sorun.",
        "draft_answer": None,
        "needs_human_review": False,
        "review_reason": None,
        "confidence": 1.0,
        "confidence_bucket": "yüksek",
        "contexts": [],
        "sources": [],
        "conflicts": [],
        "groundedness": 1.0,
        "agent_trace": _trace(state, "refuse"),
    }


def node_clarify(state: AgentState, deps: AgentDeps) -> dict:
    return {
        "answer": "Soruyu biraz açabilir misiniz? Hangi belge türünü (README, API, kurulum, mimari) kastediyorsunuz?",
        "draft_answer": None,
        "needs_human_review": False,
        "review_reason": None,
        "confidence": 1.0,
        "confidence_bucket": "yüksek",
        "contexts": [],
        "sources": [],
        "conflicts": [],
        "groundedness": 1.0,
        "agent_trace": _trace(state, "clarify"),
    }


def node_retrieve(state: AgentState, deps: AgentDeps) -> dict:
    hits = deps.retrieve_fn(state["question"], final_k=5)
    return {"contexts": hits, "agent_trace": _trace(state, f"retrieve:{len(hits)}")}


def node_writer(state: AgentState, deps: AgentDeps) -> dict:
    from src.rag.rag_pipeline import build_prompt

    contexts = state.get("contexts") or []
    if not contexts:
        return {
            "writer_answer": "Bu bilgi elimdeki şablonlarda yok.",
            "agent_trace": _trace(state, "writer:empty"),
        }
    messages = build_prompt(state["question"], contexts)
    answer = deps.generate_fn(messages)
    return {"writer_answer": answer, "agent_trace": _trace(state, "writer")}


def node_critic(state: AgentState, deps: AgentDeps) -> dict:
    contexts = state.get("contexts") or []
    writer = state.get("writer_answer") or ""
    texts = _context_texts(contexts)
    groundedness = deps.groundedness_fn(writer, texts) if writer and texts else 0.0
    unsupported = deps.flag_unsupported_fn(writer, texts) if writer and texts else []
    critic = conservative_rewrite(writer, unsupported)
    conflicts = detect_conflicts(contexts)
    return {
        "critic_answer": critic,
        "groundedness": groundedness,
        "conflicts": conflicts,
        "agent_trace": _trace(state, f"critic:g={groundedness:.2f},conflicts={len(conflicts)}"),
    }


def node_consensus(state: AgentState, deps: AgentDeps) -> dict:
    from src.xai.confidence import confidence_bucket

    contexts = state.get("contexts") or []
    conflicts = state.get("conflicts") or []
    groundedness = float(state.get("groundedness") or 0.0)
    chosen = consensus_answer(
        state.get("writer_answer") or "",
        state.get("critic_answer") or "",
        groundedness,
        n_conflicts=len(conflicts),
    )
    confidence = combine_confidence(groundedness, contexts, n_conflicts=len(conflicts))
    escalate, reason = should_escalate(confidence, n_conflicts=len(conflicts), threshold=deps.confidence_threshold)
    public = chosen
    draft = None
    if escalate:
        draft = chosen
        public = (
            "Bu yanıtın güven skoru düşük olduğu için insan onayına yönlendirildi. "
            f"Gerekçe: {reason}"
        )
    return {
        "answer": public,
        "draft_answer": draft,
        "confidence": confidence,
        "confidence_bucket": confidence_bucket(confidence),
        "needs_human_review": escalate,
        "review_reason": reason,
        "sources": _sources(contexts),
        "agent_trace": _trace(state, "consensus" + (":human" if escalate else "")),
    }


def run_sequential(state: AgentState, deps: AgentDeps) -> AgentState:
    merged: AgentState = dict(state)
    merged.update(node_router(merged, deps))
    decision = merged.get("route")
    if decision == "refuse":
        merged.update(node_refuse(merged, deps))
        return merged
    if decision == "clarify":
        merged.update(node_clarify(merged, deps))
        return merged
    merged.update(node_retrieve(merged, deps))
    merged.update(node_writer(merged, deps))
    merged.update(node_critic(merged, deps))
    merged.update(node_consensus(merged, deps))
    return merged


def _bind(fn, deps: AgentDeps):
    def _node(state: AgentState) -> dict:
        return fn(state, deps)

    return _node


def build_langgraph(deps: AgentDeps):
    if not HAS_LANGGRAPH:
        raise ImportError("langgraph kurulu değil")
    graph = StateGraph(AgentState)
    graph.add_node("router", _bind(node_router, deps))
    graph.add_node("refuse", _bind(node_refuse, deps))
    graph.add_node("clarify", _bind(node_clarify, deps))
    graph.add_node("retrieve", _bind(node_retrieve, deps))
    graph.add_node("writer", _bind(node_writer, deps))
    graph.add_node("critic", _bind(node_critic, deps))
    graph.add_node("consensus", _bind(node_consensus, deps))
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        lambda s: s.get("route") or "retrieve",
        {"retrieve": "retrieve", "refuse": "refuse", "clarify": "clarify"},
    )
    graph.add_edge("retrieve", "writer")
    graph.add_edge("writer", "critic")
    graph.add_edge("critic", "consensus")
    graph.add_edge("consensus", END)
    graph.add_edge("refuse", END)
    graph.add_edge("clarify", END)
    return graph.compile()


def run_agents(
    question: str,
    deps: AgentDeps | None = None,
    use_langgraph: bool | None = None,
) -> dict:
    deps = deps or default_deps()
    use_lg = AGENT_CONFIG.use_langgraph if use_langgraph is None else use_langgraph
    initial: AgentState = {"question": question, "agent_trace": []}
    if use_lg and HAS_LANGGRAPH:
        app = build_langgraph(deps)
        return dict(app.invoke(initial))
    return dict(run_sequential(initial, deps))
