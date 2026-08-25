"""Diyalog yönetimi: çok turlu konuşma durumunu tutan, RAG destekli bir sohbet yöneticisi.

Gerçek diyalog sistemlerinin temel zorluğu "referans çözümleme"dir: kullanıcı "Peki onun
menzili nedir?" dediğinde "onun"un neyi işaret ettiğini anlamak gerekir. Burada bunu basit
ama etkili bir teknikle çözüyoruz: her yeni soruyu, önceki tur(lar)ın özetiyle birlikte
"bağımsız soru"ya (standalone query) dönüştürüp RAG'a öyle veriyoruz.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class DialogueSession:
    session_id: str
    history: list[Turn] = field(default_factory=list)
    max_history_turns: int = 6

    def add_user_turn(self, text: str) -> None:
        self.history.append(Turn("user", text))

    def add_assistant_turn(self, text: str) -> None:
        self.history.append(Turn("assistant", text))
        self._trim()

    def _trim(self) -> None:
        # Bağlam penceresini şişirmemek için son N turu tut.
        if len(self.history) > self.max_history_turns * 2:
            self.history = self.history[-self.max_history_turns * 2:]

    def history_as_text(self) -> str:
        return "\n".join(f"{t.role}: {t.content}" for t in self.history)


def rewrite_standalone_query(session: DialogueSession, new_question: str) -> str:
    """Önceki turları kullanarak soruyu bağımsız hale getirir.

    Basit sezgisel kural: geçmiş boşsa soru zaten bağımsızdır. Doluysa, son
    kullanıcı/asistan turunu bağlam olarak soruya iliştiriyoruz — tam bir coreference
    resolution modeli yerine, LLM'in kendi bağlam penceresinde bunu çözmesine izin
    veren ucuz ve pratik bir yöntem.
    """
    if not session.history:
        return new_question
    last_turns = session.history[-2:]
    context = " ".join(t.content for t in last_turns)
    return f"[Önceki bağlam: {context}]\nSoru: {new_question}"


_sessions: dict[str, DialogueSession] = {}


def get_or_create_session(session_id: str) -> DialogueSession:
    if session_id not in _sessions:
        _sessions[session_id] = DialogueSession(session_id=session_id)
    return _sessions[session_id]


def chat(session_id: str, user_message: str) -> dict:
    """Tek bir diyalog turunu işler: çok ajanlı grafı çağırır, geçmişi günceller."""
    from src.agents.graph import run_agents

    session = get_or_create_session(session_id)
    standalone_query = rewrite_standalone_query(session, user_message)
    session.add_user_turn(user_message)

    result = run_agents(standalone_query)
    public_answer = result.get("answer") or ""
    session.add_assistant_turn(public_answer)

    contexts = result.get("contexts") or []
    return {
        "session_id": session_id,
        "answer": public_answer,
        "sources": result.get("sources", []),
        "context_texts": [c.get("parent_text") or c.get("text") or "" for c in contexts],
        "turn_count": len(session.history) // 2,
        "confidence": result.get("confidence"),
        "confidence_bucket": result.get("confidence_bucket"),
        "needs_human_review": bool(result.get("needs_human_review")),
        "review_reason": result.get("review_reason"),
        "draft_answer": result.get("draft_answer"),
        "agent_trace": result.get("agent_trace") or [],
        "conflicts": result.get("conflicts") or [],
    }
