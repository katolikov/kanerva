"""Secret detection applied to everything before it reaches the store."""

from __future__ import annotations

import math
import re
from collections import Counter

PATTERNS: dict[str, re.Pattern[str]] = {
    "private-key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github-token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "anthropic-key": re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}"),
    "openai-key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),
    "slack-token": re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    "stripe-key": re.compile(r"\b[sr]k_live_[A-Za-z0-9]{16,}"),
    "google-api-key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "npm-token": re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    "url-credentials": re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:[^\s:/@]+@[^\s]+"),
    "auth-header": re.compile(r"(?i)\b(?:authorization\s*[:=]\s*[\"']?(?:bearer|basic|token)\s+|bearer\s+)[A-Za-z0-9._~+/\-]{16,}=*"),
}

_ASSIGNMENT = re.compile(
    r"(?i)\b[\w\-]*(?:pass(?:word|wd)?|secret|token|api[_-]?key|auth[_-]?key|access[_-]?key|private[_-]?key|credential)s?[\w\-]*"
    r"[\"']?\s*[:=]\s*[\"']?([^\s\"',;]{8,})"
)


def shannon_entropy(value: str) -> float:
    counts = Counter(value)
    return -sum(c / len(value) * math.log2(c / len(value)) for c in counts.values())


def find_secrets(text: str) -> list[tuple[str, int, int]]:
    """Return (kind, start, end) spans of probable secrets."""
    spans: list[tuple[str, int, int]] = []
    for kind, pattern in PATTERNS.items():
        spans.extend((kind, m.start(), m.end()) for m in pattern.finditer(text))
    for m in _ASSIGNMENT.finditer(text):
        value = m.group(1)
        placeholder = value.startswith(("$", "{", "<", "process.env", "os.environ")) or value.isdigit()
        if shannon_entropy(value) >= 2.5 and not placeholder:
            spans.append(("assigned-secret", m.start(1), m.end(1)))
    return spans


def redact(text: str) -> tuple[str, list[str]]:
    """Replace secret spans with typed placeholders; returns the text and the kinds found."""
    spans = sorted(find_secrets(text), key=lambda s: s[1])
    kinds: list[str] = []
    out: list[str] = []
    cursor = 0
    for kind, start, end in spans:
        if start < cursor:
            continue
        out.append(text[cursor:start])
        out.append(f"[REDACTED:{kind}]")
        cursor = end
        kinds.append(kind)
    out.append(text[cursor:])
    return "".join(out), kinds
