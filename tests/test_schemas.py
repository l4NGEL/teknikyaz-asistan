import pytest
from pydantic import ValidationError

from src.serving.schemas import ChatRequest, SummarizeRequest


def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="")


def test_chat_request_accepts_valid_message():
    req = ChatRequest(session_id="s1", message="Bayraktar TB2 nedir?")
    assert req.session_id == "s1"


def test_summarize_request_rejects_invalid_method():
    with pytest.raises(ValidationError):
        SummarizeRequest(text="metin", method="invalid-method")


def test_summarize_request_defaults_to_extractive():
    req = SummarizeRequest(text="metin")
    assert req.method == "extractive"
