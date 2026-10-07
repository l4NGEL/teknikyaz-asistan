import json

import pytest

from src.governance import audit
from src.governance.acl import AccessPolicy, Principal, default_policy, load_principals, resolve_principal
from src.governance.pii import mask, redact, valid_iban, valid_luhn, valid_tckn
from src.llmops import tracing

# --- ACL ---------------------------------------------------------------------

ALL_TITLES = [
    "README Şablonu",
    "SSS — Faturalandırma",
    "API Referansı: oturum_dogrula",
    "Kurulum Kılavuzu — Bildirim Servisi",
    "Mimari Doküman — Kimlik Doğrulama ve Yetkilendirme",
    "Güvenlik Politikası Şablonu",
    "Tamamen Yeni Bir Belge",
]


def test_policy_classifies_by_title():
    policy = default_policy()
    assert policy.classify("README Şablonu").classification == "public"
    assert policy.classify("API Referansı: oturum_dogrula").classification == "internal"
    assert policy.classify("Kurulum Kılavuzu Şablonu").classification == "public"
    assert policy.classify("Kurulum Kılavuzu — Bildirim Servisi").classification == "internal"
    assert policy.classify("Mimari Doküman — Kimlik Doğrulama ve Yetkilendirme").classification == "restricted"


def test_unknown_document_is_denied_by_default():
    policy = default_policy()
    access = policy.classify("Tamamen Yeni Bir Belge")
    assert access.classification == "restricted" and access.groups == {"admin"}
    guvenlik = load_principals()["guvenlik"]
    assert not policy.can_read(guvenlik, "Tamamen Yeni Bir Belge")
    assert policy.can_read(load_principals()["admin"], "Tamamen Yeni Bir Belge")


@pytest.mark.parametrize(
    "user, readable",
    [
        ("musteri", {"README Şablonu", "SSS — Faturalandırma"}),
        ("gelistirici", {"README Şablonu", "SSS — Faturalandırma", "API Referansı: oturum_dogrula",
                         "Kurulum Kılavuzu — Bildirim Servisi"}),
        ("guvenlik", set(ALL_TITLES) - {"Tamamen Yeni Bir Belge"}),
    ],
)
def test_principal_visibility_matrix(user, readable):
    policy = default_policy()
    principal = load_principals()[user]
    assert {t for t in ALL_TITLES if policy.can_read(principal, t)} == readable


def test_clearance_alone_is_not_enough_without_group():
    policy = default_policy()
    high_clearance_wrong_group = Principal("x", "restricted", frozenset({"engineering"}))
    assert not policy.can_read(high_clearance_wrong_group, "Güvenlik Politikası Şablonu")
    assert policy.can_read(high_clearance_wrong_group, "API Referansı: puan_ekle")


def test_unknown_user_fails_closed():
    p = resolve_principal("admin ")  # boşluk: birebir eşleşmez → yükseltilmez
    assert p.clearance == "public" and not p.groups
    assert resolve_principal(None).user_id == "anonymous"


def test_filter_hits_counts_denials():
    hits = [{"chunk_id": f"{t}::0", "doc_title": t} for t in ALL_TITLES]
    allowed, denied = default_policy().filter_hits(load_principals()["musteri"], hits)
    assert [h["doc_title"] for h in allowed] == ["README Şablonu", "SSS — Faturalandırma"]
    assert denied == len(ALL_TITLES) - 2


def test_policy_rejects_unknown_level():
    with pytest.raises(ValueError):
        AccessPolicy(["public"], {"classification": "public"}, [{"title_regex": ".", "classification": "gizli"}])


# --- PII ---------------------------------------------------------------------

VALID_TCKN = "10000000146"
VALID_IBAN = "TR33 0006 1005 1978 6457 8413 26"
VALID_CARD = "4111 1111 1111 1111"


def test_checksum_validators():
    assert valid_tckn(VALID_TCKN) and not valid_tckn("12345678901")
    assert valid_iban(VALID_IBAN) and not valid_iban("TR33 0006 1005 1978 6457 8413 27")
    assert valid_luhn(VALID_CARD) and not valid_luhn("4111 1111 1111 1112")


def test_redact_masks_turkish_pii():
    text = (f"TCKN {VALID_TCKN}, IBAN {VALID_IBAN}, kart {VALID_CARD}, "
            "tel 0532 123 45 67, e-posta ayse.kaya@ornek.com.tr, ip 10.0.0.12")
    result = redact(text)
    assert dict(result.counts) == {"TCKN": 1, "IBAN": 1, "CARD": 1, "PHONE": 1, "EMAIL": 1, "IP": 1}
    for raw in (VALID_TCKN, "8413 26", "1111 1111", "123 45 67", "ayse.kaya", "10.0.0.12"):
        assert raw not in result.text


def test_redact_keeps_invalid_numbers():
    # Checksum tutmayan 11 haneli sayı (sipariş no gibi) maskelenmemeli.
    assert mask("Sipariş 12345678901 hazır") == "Sipariş 12345678901 hazır"
    assert mask("sürüm 1.2.3") == "sürüm 1.2.3"


def test_redact_secrets():
    text = (
        "anahtar sk-ant-api03-abcdefghijklmnopqrstuvwxyz012345 ve ghp_" + "a" * 36
        + " password=hunter2 token: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.c2lnbmF0dXJlLXZhbHVl"
    )
    out = redact(text)
    assert "hunter2" not in out.text and "sk-ant" not in out.text and "ghp_" not in out.text
    assert "password=[SECRET]" in out.text
    assert out.counts["API_KEY"] == 2


# --- Denetim kaydı -----------------------------------------------------------

def test_audit_chain_detects_tampering(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit.log_access("musteri", "search", f"TCKN'm {VALID_TCKN}", ["README Şablonu::0"], 3, path=path)
    audit.log_access("gelistirici", "search", "kurulum", ["Kurulum Kılavuzu — Form Motoru::0"], 0, path=path)
    audit.log_access("musteri", "chat", "merhaba", [], 0, path=path)
    assert audit.verify(path) == (True, None)

    rows = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    assert VALID_TCKN not in rows[0] and first["pii_masked"] == {"TCKN": 1}

    first["denied_count"] = 0  # biri izini örtmeye çalışıyor
    rows[0] = json.dumps(first, ensure_ascii=False)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert audit.verify(path) == (False, 1)


def test_audit_chain_detects_deleted_line(tmp_path):
    path = tmp_path / "audit.jsonl"
    for i in range(3):
        audit.log_access("u", "search", f"soru {i}", [], 0, path=path)
    rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([rows[0], rows[2]]) + "\n", encoding="utf-8")
    assert audit.verify(path) == (False, 2)


# --- Tracing -----------------------------------------------------------------

def test_spans_nest_and_export_masked(tmp_path):
    path = tmp_path / "traces.jsonl"
    with tracing.span("request", path=path, question=f"IBAN {VALID_IBAN}"):
        with tracing.span("retrieve"):
            with tracing.span("rerank", pairs=12):
                pass
        tracing.annotate(answer_len=42)

    [trace] = tracing.load_traces(path)
    assert trace["name"] == "request" and trace["parent_id"] is None
    assert "[IBAN]" in trace["attributes"]["question"]
    assert trace["attributes"]["answer_len"] == 42
    retrieve = trace["children"][0]
    assert retrieve["parent_id"] == trace["span_id"]
    assert retrieve["children"][0]["trace_id"] == trace["trace_id"]
    summary = tracing.stage_latency_summary([trace])
    assert set(summary) == {"request", "retrieve", "rerank"}


def test_span_records_error_and_reraises(tmp_path):
    path = tmp_path / "traces.jsonl"
    with pytest.raises(RuntimeError):
        with tracing.span("generate", path=path):
            raise RuntimeError("GPU OOM")
    [trace] = tracing.load_traces(path)
    assert trace["status"] == "error" and "GPU OOM" in trace["attributes"]["error"]
