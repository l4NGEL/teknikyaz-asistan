from unittest.mock import patch

from src.optimize.vllm_benchmark import percentile
from src.optimize.vllm_client import VLLMError, generate_vllm


def test_percentile_p95():
    values = [float(i) for i in range(1, 21)]
    p95 = percentile(values, 95)
    assert 18 <= p95 <= 20


def test_generate_vllm_reads_openai_payload():
    payload = {"choices": [{"message": {"content": "  README kurulum  "}}]}
    with patch("src.optimize.vllm_client.chat_completions", return_value=payload):
        assert generate_vllm("soru") == "README kurulum"


def test_generate_vllm_wraps_http_errors():
    import requests

    with patch("src.optimize.vllm_client.requests.post", side_effect=requests.ConnectionError("down")):
        try:
            generate_vllm("soru")
        except VLLMError as exc:
            assert "ulaşılamadı" in str(exc)
        else:
            raise AssertionError("VLLMError bekleniyordu")
