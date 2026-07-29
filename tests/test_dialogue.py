from src.nlp_tasks.dialogue import DialogueSession, rewrite_standalone_query


def test_empty_history_returns_question_unchanged():
    session = DialogueSession(session_id="s1")
    question = "Bayraktar TB2'nin menzili nedir?"
    assert rewrite_standalone_query(session, question) == question


def test_history_is_injected_into_rewritten_query():
    session = DialogueSession(session_id="s1")
    session.add_user_turn("Bayraktar TB2 nedir?")
    session.add_assistant_turn("Bayraktar TB2, Baykar tarafından geliştirilen bir İHA'dır.")

    rewritten = rewrite_standalone_query(session, "Peki menzili ne kadar?")

    assert "Bayraktar TB2" in rewritten
    assert "Peki menzili ne kadar?" in rewritten


def test_history_trims_to_max_turns():
    session = DialogueSession(session_id="s1", max_history_turns=2)
    for i in range(10):
        session.add_user_turn(f"soru {i}")
        session.add_assistant_turn(f"cevap {i}")

    assert len(session.history) == session.max_history_turns * 2
