import pytest
from pydantic import ValidationError

from src.serving.schemas import ChatRequest, SummarizeRequest


def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="")


def test_chat_request_accepts_valid_message():
    req = ChatRequest(session_id="s1", message="İyi bir API dokümantasyonu nasıl yazılır?")
    assert req.session_id == "s1"


def test_summarize_request_rejects_invalid_method():
    with pytest.raises(ValidationError):
        SummarizeRequest(text="metin", method="invalid-method")


def test_chat_response_defaults_human_review_to_false():
    from src.serving.schemas import ChatResponse

    resp = ChatResponse(session_id="s1", answer="ok", sources=[], turn_count=1)
    assert resp.needs_human_review is False
    assert resp.agent_trace == []
