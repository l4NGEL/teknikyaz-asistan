"""Yazar ve eleştirmen çıktılarından tek bir cevap seçer.

Kural: groundedness yüksekse yazarın cevabı; düşükse veya çelişki varsa
eleştirmenin (daha muhafazakâr) metni. İkisi de boşsa reddet.
"""
from __future__ import annotations


def consensus_answer(
    writer_answer: str,
    critic_answer: str,
    groundedness: float,
    n_conflicts: int,
    groundedness_keep_writer: float = 0.5,
) -> str:
    writer = (writer_answer or "").strip()
    critic = (critic_answer or "").strip()
    if n_conflicts > 0 and critic:
        return critic
    if groundedness >= groundedness_keep_writer and writer:
        return writer
    return critic or writer or "Bu bilgi elimdeki şablonlarda yok."


def conservative_rewrite(answer: str, unsupported: list[dict] | None = None) -> str:
    """Desteksiz cümleleri düşürür; hiç cümle kalmazsa dürüst ret."""
    if not unsupported:
        return answer
    drop = {item["sentence"] for item in unsupported if item.get("sentence")}
    kept = [s.strip() for s in answer.split(".") if s.strip() and not any(s.strip() in d or d in s for d in drop)]
    if not kept:
        return "Bu bilgi elimdeki şablonlarda yok."
    text = ". ".join(kept)
    if not text.endswith("."):
        text += "."
    return text
