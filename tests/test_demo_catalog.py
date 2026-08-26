from src.serving.demo_catalog import lookup_demo


def test_demo_readme_is_high_confidence_retrieve():
    payload = lookup_demo("Bir README'de hangi bölümler olmalı?")
    assert payload["needs_human_review"] is False
    assert payload["confidence_bucket"] == "yüksek"
    assert payload["sources"][0]["title"] == "README Şablonu"
    assert "router:retrieve" in payload["agent_trace"]


def test_demo_greeting_is_welcome_not_refuse():
    payload = lookup_demo("selam")
    assert "router:greet" in payload["agent_trace"]
    assert "Merhaba" in payload["answer"]
    assert payload["sources"] == []


def test_demo_thanks_is_acknowledged():
    payload = lookup_demo("teşekkürler")
    assert "thanks" in payload["agent_trace"]
    assert "Rica ederim" in payload["answer"]
    payload = lookup_demo("hava durumu yarın nasıl")
    assert "router:refuse" in payload["agent_trace"]
    assert payload["sources"] == []


def test_demo_unknown_topic_escalates_to_human():
    payload = lookup_demo("Drone sürü zekası belgesi nasıl yazılır?")
    assert payload["needs_human_review"] is True
    assert payload["draft_answer"]
    assert "consensus:human" in payload["agent_trace"]


def test_demo_short_question_asks_to_clarify():
    payload = lookup_demo("ok")
    assert "router:clarify" in payload["agent_trace"]
