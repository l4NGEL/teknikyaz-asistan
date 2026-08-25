from src.agents.conflict import detect_conflicts
from src.agents.consensus import consensus_answer
from src.agents.graph import AgentDeps, build_langgraph, run_agents, run_sequential
from src.agents.router import route


def test_router_refuses_off_domain_and_greetings():
    assert route("hava durumu yarın nasıl") == "refuse"
    assert route("merhaba") == "refuse"
    assert route("ok") == "clarify"
    assert route("Bir README'de hangi bölümler olmalı?") == "retrieve"


def test_detects_license_conflict():
    contexts = [
        {"doc_title": "A", "text": "Bu proje MIT lisansı altındadır."},
        {"doc_title": "B", "text": "Lisans Apache License 2.0 olarak belirtilmiştir."},
    ]
    conflicts = detect_conflicts(contexts)
    assert any(c["type"] == "license" for c in conflicts)


def test_consensus_prefers_critic_when_conflicts():
    chosen = consensus_answer("yazar metni", "eleştirmen metni", groundedness=0.9, n_conflicts=1)
    assert chosen == "eleştirmen metni"


def _deps(**overrides):
    def retrieve_fn(question, final_k=5, **_k):
        return [
            {
                "doc_title": "README Şablonu",
                "doc_url": "https://example.com/readme",
                "text": "README kurulum, kullanım ve lisans bölümlerini içerir.",
                "parent_text": "README kurulum, kullanım ve lisans bölümlerini içerir.",
                "rerank_score": 0.92,
            }
        ][:final_k]

    def generate_fn(messages, **_k):
        return "README kurulum, kullanım ve lisans bölümlerini içermelidir."

    def groundedness_fn(answer, contexts):
        return 0.85

    kwargs = dict(
        retrieve_fn=retrieve_fn,
        generate_fn=generate_fn,
        groundedness_fn=groundedness_fn,
        flag_unsupported_fn=lambda *_a, **_k: [],
        confidence_threshold=0.45,
    )
    kwargs.update(overrides)
    return AgentDeps(**kwargs)


def test_sequential_graph_returns_grounded_answer_without_human_gate():
    result = run_sequential({"question": "Bir README'de hangi bölümler olmalı?", "agent_trace": []}, _deps())
    assert result["route"] == "retrieve"
    assert result["needs_human_review"] is False
    assert "README" in result["answer"]
    assert result["agent_trace"][0].startswith("router:")


def test_low_confidence_escalates_to_human():
    deps = _deps(groundedness_fn=lambda *_a, **_k: 0.1, confidence_threshold=0.45)
    result = run_agents("Bir README'de hangi bölümler olmalı?", deps=deps, use_langgraph=False)
    assert result["needs_human_review"] is True
    assert result["draft_answer"]
    assert "insan onayına" in result["answer"]


def test_langgraph_matches_sequential_when_available():
    try:
        app = build_langgraph(_deps())
    except ImportError:
        return
    question = "Bir README'de hangi bölümler olmalı?"
    sequential = run_sequential({"question": question, "agent_trace": []}, _deps())
    graphed = dict(app.invoke({"question": question, "agent_trace": []}))
    assert graphed["answer"] == sequential["answer"]
    assert graphed["needs_human_review"] == sequential["needs_human_review"]
