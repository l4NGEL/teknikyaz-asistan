"""vLLM OpenAI-uyumlu HTTP istemcisi.

vLLM paketini (büyük, GPU) zorunlu bağımlılık yapmıyoruz: serving ayrı bir
süreci dinler (`docker compose --profile gpu up vllm`). Bu, Diagnis ilanındaki
"production inference: vLLM" maddesinin bu projedeki karşılığıdır.
"""
from __future__ import annotations

from typing import Any

import requests

from src.config import VLLM_CONFIG


class VLLMError(RuntimeError):
    pass


def chat_completions(
    messages: list[dict],
    *,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    base_url = (base_url or VLLM_CONFIG.base_url).rstrip("/")
    model = model or VLLM_CONFIG.model
    timeout_s = VLLM_CONFIG.timeout_s if timeout_s is None else timeout_s
    url = f"{base_url}/chat/completions"
    try:
        response = requests.post(
            url,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=timeout_s,
        )
    except requests.RequestException as exc:
        raise VLLMError(f"vLLM'e ulaşılamadı ({url}): {exc}") from exc
    if response.status_code >= 400:
        raise VLLMError(f"vLLM HTTP {response.status_code}: {response.text[:500]}")
    return response.json()


def generate_vllm(prompt: str, **kwargs: Any) -> str:
    payload = chat_completions(
        [
            {"role": "system", "content": "Kısa ve doğru Türkçe cevap ver."},
            {"role": "user", "content": prompt},
        ],
        **kwargs,
    )
    try:
        return payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError) as exc:
        raise VLLMError(f"beklenmeyen vLLM yanıtı: {payload!r}") from exc
