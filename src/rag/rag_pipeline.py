"""Uçtan uca RAG: retrieve -> kaynaklı prompt oluştur -> LLM ile üret -> kaynakları döndür.

`model_path` verilirse (örn. QLoRA fine-tune + DPO sonrası adapter birleştirilmiş model)
o kullanılır; verilmezse config'teki temel model. Bu sayede aynı pipeline, projenin
4. ve 5. aşamalarından önce/sonra karşılaştırmalı olarak (LLMOps A/B test) çalıştırılabilir.
"""
from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MODEL_CONFIG
from src.rag.retriever import retrieve

SYSTEM_PROMPT = """Sen Türkçe savunma ve havacılık teknik dokümanları konusunda uzman bir
asistansın. SADECE sana verilen bağlamdaki bilgileri kullanarak cevap ver. Bağlamda cevap
yoksa "Bu bilgi elimdeki dokümanlarda yok." de. Cevabının sonunda kullandığın kaynakların
başlıklarını [Kaynak: ...] formatında belirt."""


@lru_cache(maxsize=4)
def _load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    return tokenizer, model


def build_prompt(question: str, contexts: list[dict]) -> list[dict]:
    context_block = "\n\n".join(
        f"[{c['doc_title']}]\n{c['text']}" for c in contexts
    )
    user_content = f"Bağlam:\n{context_block}\n\nSoru: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate(messages: list[dict], model_path: str | None = None, max_new_tokens: int = 300) -> str:
    model_path = model_path or MODEL_CONFIG.base_llm
    tokenizer, model = _load_model(model_path)

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def answer(question: str, model_path: str | None = None, top_k: int = 5, rerank: bool = True) -> dict:
    contexts = retrieve(question, final_k=top_k, rerank=rerank)
    if not contexts:
        return {
            "answer": "Bu bilgi elimdeki dokümanlarda yok.",
            "sources": [],
            "contexts": [],
        }

    messages = build_prompt(question, contexts)
    response = generate(messages, model_path=model_path)

    return {
        "answer": response,
        "sources": [{"title": c["doc_title"], "url": c["doc_url"]} for c in contexts],
        "contexts": contexts,
    }
