from src.nlp_tasks.dialogue import DialogueSession, rewrite_standalone_query


def test_empty_history_returns_question_unchanged():
    session = DialogueSession(session_id="s1")
    question = "İyi bir README nasıl yapılandırılır?"
    assert rewrite_standalone_query(session, question) == question


def test_history_is_injected_into_rewritten_query():
    session = DialogueSession(session_id="s1")
    session.add_user_turn("İyi bir README nasıl yapılandırılır?")
    session.add_assistant_turn("README; özellikler, kurulum, kullanım ve lisans bölümlerini içermelidir.")

    rewritten = rewrite_standalone_query(session, "Peki kurulum bölümüne ne yazmalıyım?")

    assert "README" in rewritten
    assert "Peki kurulum bölümüne ne yazmalıyım?" in rewritten


def test_history_trims_to_max_turns():
    session = DialogueSession(session_id="s1", max_history_turns=2)
    for i in range(10):
        session.add_user_turn(f"soru {i}")
        session.add_assistant_turn(f"cevap {i}")

    assert len(session.history) == session.max_history_turns * 2
