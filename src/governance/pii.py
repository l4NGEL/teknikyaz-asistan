"""PII ve secret maskeleme: loglara, denetim kaydına ve dış araçlara giden metin için.

Desen tek başına yanlış pozitif üretir (11 haneli her sayı TCKN değildir), o yüzden
TCKN, IBAN ve kart numarası checksum ile doğrulanır; yalnızca geçerli olanlar maskelenir.
Secret desenleri (API anahtarı, JWT, özel anahtar bloğu) her zaman maskelenir: bunların
bir log satırında görünmesinin meşru bir nedeni yok.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field


def valid_tckn(digits: str) -> bool:
    if len(digits) != 11 or not digits.isdigit() or digits[0] == "0":
        return False
    d = [int(c) for c in digits]
    tenth = ((d[0] + d[2] + d[4] + d[6] + d[8]) * 7 - (d[1] + d[3] + d[5] + d[7])) % 10
    return d[9] == tenth and d[10] == sum(d[:10]) % 10


def valid_iban(iban: str) -> bool:
    s = re.sub(r"\s+", "", iban).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", s):
        return False
    if s.startswith("TR") and len(s) != 26:
        return False
    rearranged = s[4:] + s[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def valid_luhn(number: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


# (etiket, desen, doğrulayıcı). Sıra önemli: secret'lar ve IBAN, içlerindeki rakam
# dizilerini telefon/TCKN desenleri yakalamadan önce maskelenir.
_RULES: list[tuple[str, re.Pattern, Callable[[str], bool] | None]] = [
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"), None),
    ("JWT", re.compile(r"\beyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}\b"), None),
    ("API_KEY", re.compile(
        r"\b(?:sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
        r"|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35})\b"
    ), None),
    ("SECRET", re.compile(r"(?i)\b(password|passwd|parola|şifre|secret|api[_-]?key|token)\s*[:=]\s*\S+"), None),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), None),
    ("IBAN", re.compile(r"\bTR\d{2}(?:\s?[0-9A-Z]{4}){5}\s?[0-9A-Z]{2}\b", re.IGNORECASE), valid_iban),
    ("CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), valid_luhn),
    ("TCKN", re.compile(r"\b\d{11}\b"), valid_tckn),
    ("PHONE", re.compile(r"(?<!\d)(?:\+90[\s-]?|0)?5\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"), None),
    ("IP", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), None),
]


@dataclass
class RedactionResult:
    text: str
    counts: Counter = field(default_factory=Counter)

    @property
    def found(self) -> bool:
        return bool(self.counts)


def redact(text: str) -> RedactionResult:
    if not text:
        return RedactionResult(text or "")
    counts: Counter = Counter()
    for label, pattern, validator in _RULES:
        def _sub(match: re.Match, label=label, validator=validator) -> str:
            value = match.group(0)
            if validator is not None and not validator(re.sub(r"[\s-]", "", value)):
                return value
            counts[label] += 1
            if label == "SECRET":  # anahtar adını koru, yalnızca değeri maskele
                return f"{match.group(1)}=[SECRET]"
            return f"[{label}]"

        text = pattern.sub(_sub, text)
    return RedactionResult(text, counts)


def mask(text: str) -> str:
    return redact(text).text
