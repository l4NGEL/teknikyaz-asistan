"""MCP sunucusu gerçek protokol üzerinden (in-memory transport) test edilir."""
import anyio
import pytest
from mcp.client import Client

from src.governance.acl import load_principals
from src.serving.mcp_server import NOT_FOUND, build_server

DOCS = {
    "README Şablonu": [{"text": "# Proje Adı kurulum", "metadata": {"chunk_index": 0}}],
    "Güvenlik Politikası Şablonu": [{"text": "yönetici parolası: password=Gizli123", "metadata": {"chunk_index": 0}}],
}


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(args)


def fake_retrieve(question, top_k, principal):
    hits = [
        {"chunk_id": "README Şablonu::0", "doc_title": "README Şablonu",
         "text": "iletişim: destek@ornek.com", "fusion_score": 0.91},
    ]
    return hits[:top_k], 2


def make(user):
    audit = Recorder()
    server = build_server(load_principals()[user], retrieve_fn=fake_retrieve, documents=lambda: DOCS, audit=audit)
    return server, audit


def run(coro_fn):
    return anyio.run(coro_fn)


def test_tools_are_listed_read_only():
    server, _ = make("musteri")

    async def go():
        async with Client(server) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in run(go)}
    assert set(tools) == {"search_docs", "get_document", "whoami"}
    assert all(t.annotations.read_only_hint for t in tools.values())
    # Kimlik bir araç parametresi değil: model kendini admin ilan edemez.
    assert all("user" not in (t.input_schema.get("properties") or {}) for t in tools.values())


def test_search_masks_pii_and_audits():
    server, audit = make("musteri")

    async def go():
        async with Client(server) as client:
            return await client.call_tool("search_docs", {"query": "readme", "top_k": 3})

    result = run(go)
    assert not result.is_error
    payload = result.structured_content["result"]
    assert payload[0]["doc_title"] == "README Şablonu"
    assert "[EMAIL]" in payload[0]["snippet"] and "destek@" not in payload[0]["snippet"]
    assert audit.calls == [("musteri", "mcp.search_docs", "readme", ["README Şablonu::0"], 2)]


@pytest.mark.parametrize("title", ["Güvenlik Politikası Şablonu", "Hiç Olmayan Belge"])
def test_forbidden_and_missing_documents_look_identical(title):
    server, audit = make("musteri")

    async def go():
        async with Client(server) as client:
            return await client.call_tool("get_document", {"doc_title": title})

    result = run(go)
    assert result.is_error
    assert result.content[0].text.endswith(NOT_FOUND)
    assert audit.calls[-1][-1] == 1  # reddedilen erişim de kayda geçer


def test_authorized_document_is_returned_masked():
    server, _ = make("guvenlik")

    async def go():
        async with Client(server) as client:
            return await client.call_tool("get_document", {"doc_title": "Güvenlik Politikası Şablonu"})

    result = run(go)
    assert not result.is_error
    doc = result.structured_content
    assert doc["classification"] == "restricted"
    assert "Gizli123" not in doc["text"] and "password=[SECRET]" in doc["text"]


def test_whoami_reports_session_identity():
    server, _ = make("gelistirici")

    async def go():
        async with Client(server) as client:
            return await client.call_tool("whoami", {})

    assert run(go).structured_content == {"user": "gelistirici", "clearance": "internal", "groups": ["engineering"]}
