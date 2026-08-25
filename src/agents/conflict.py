"""Kaynaklar arası çelişki tespiti.

Aynı soruya dönen pasajlarda lisans, sürüm veya sayısal iddia ayrışırsa
konsensüs düğümü ve insan-onayı kapısı bunu görür. NLI olmadan da çalışır;
üretimde groundedness ile birlikte kullanılır.
"""
from __future__ import annotations

import re

_LICENSE_RE = re.compile(r"\b(MIT|Apache(?: License)?(?: 2\.0)?|GPL(?:-?\d)?|BSD)\b", re.I)
_VERSION_RE = re.compile(r"\bv?(\d+\.\d+(?:\.\d+)?)\b", re.I)
_MS_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*ms\b", re.I)


def _licenses(text: str) -> set[str]:
    found: set[str] = set()
    for match in _LICENSE_RE.finditer(text or ""):
        token = match.group(1).upper()
        if token.startswith("APACHE"):
            found.add("APACHE")
        elif token.startswith("GPL"):
            found.add("GPL")
        else:
            found.add(token)
    return found


def detect_conflicts(contexts: list[dict]) -> list[dict]:
    """Bağlam pasajları arasında lisans / sürüm / gecikme çelişkisi arar."""
    if len(contexts) < 2:
        return []

    conflicts: list[dict] = []
    all_licenses: set[str] = set()
    version_by_title: dict[str, str] = {}
    latencies: list[float] = []

    for ctx in contexts:
        title = ctx.get("doc_title") or ""
        text = f"{title} {ctx.get('parent_text') or ctx.get('text') or ''}"
        all_licenses |= _licenses(text)
        versions = _VERSION_RE.findall(text)
        if versions:
            version_by_title[title] = versions[0]
        for match in _MS_RE.finditer(text):
            try:
                latencies.append(float(match.group(1).replace(",", ".")))
            except ValueError:
                pass

    if len(all_licenses) >= 2:
        conflicts.append({"type": "license", "detail": f"farklı lisanslar: {sorted(all_licenses)}"})

    unique_versions = set(version_by_title.values())
    if len(version_by_title) >= 2 and len(unique_versions) >= 2:
        conflicts.append({"type": "version", "detail": f"farklı sürüm numaraları: {version_by_title}"})

    if latencies and min(latencies) > 0 and max(latencies) >= 10 * min(latencies):
        conflicts.append(
            {
                "type": "numeric",
                "detail": f"ms cinsinden değerler bir büyüklük mertebesinden fazla ayrışıyor: {latencies}",
            }
        )

    return conflicts
