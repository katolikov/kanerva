"""Tokenisation helpers shared by the lexical index, selection and feedback."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Callable, Iterable

STOPWORDS = frozenset(
    """a about above after again against all am an and any are aren as at be because been before
    being below between both but by can could did didn do does doesn doing don down during each
    few for from further had hadn has hasn have haven having he her here hers herself him himself
    his how i if in into is isn it its itself just ll me might more most mustn my myself no nor
    not now of off on once only or other our ours ourselves out over own re s same shan she should
    shouldn so some such t than that the their theirs them themselves then there these they this
    those through to too under until up us ve very was wasn we were weren what when where which
    while who whom why will with won would wouldn you your yours yourself yourselves""".split()
    + """\u0430 \u0431\u0435\u0437 \u0431\u043e\u043b\u0435\u0435 \u0431\u044b \u0431\u044b\u043b \u0431\u044b\u043b\u0430 \u0431\u044b\u043b\u0438 \u0431\u044b\u043b\u043e \u0431\u044b\u0442\u044c \u0432 \u0432\u0430\u043c \u0432\u0430\u043c\u0438 \u0432\u0430\u0441 \u0432\u0435\u0434\u044c \u0432\u0435\u0441\u044c \u0432\u043e \u0432\u043e\u0442 \u0432\u0441\u0435 \u0432\u0441\u0435\u0433\u043e \u0432\u0441\u0435\u0439
    \u0432\u0441\u0435\u043c \u0432\u0441\u0435\u043c\u0438 \u0432\u0441\u0435\u0445 \u0432\u0441\u044e \u0432\u0441\u044f \u0432\u0441\u0451 \u0432\u044b \u0433\u0434\u0435 \u0434\u0430 \u0434\u0430\u0436\u0435 \u0434\u043b\u044f \u0434\u043e \u0434\u0440\u0443\u0433\u0430\u044f \u0434\u0440\u0443\u0433\u0438\u0435 \u0434\u0440\u0443\u0433\u0438\u043c \u0434\u0440\u0443\u0433\u0438\u0445 \u0434\u0440\u0443\u0433\u043e\u0435 \u0434\u0440\u0443\u0433\u043e\u0439
    \u0435\u0433\u043e \u0435\u0435 \u0435\u0439 \u0435\u043c\u0443 \u0435\u0441\u043b\u0438 \u0435\u0441\u0442\u044c \u0435\u0449\u0435 \u0435\u0449\u0451 \u0435\u0451 \u0435\u044e \u0436\u0435 \u0437\u0430 \u0437\u0430\u0447\u0435\u043c \u0437\u0434\u0435\u0441\u044c \u0438 \u0438\u0437 \u0438\u043b\u0438 \u0438\u043c \u0438\u043c\u0438 \u0438\u0445 \u043a \u043a\u0430\u043a \u043a\u0430\u043a\u0430\u044f \u043a\u0430\u043a\u0438\u0435
    \u043a\u0430\u043a\u0438\u043c \u043a\u0430\u043a\u0438\u043c\u0438 \u043a\u0430\u043a\u0438\u0445 \u043a\u0430\u043a\u0438\u043c \u043a\u0430\u043a\u043e\u0432 \u043a\u0430\u043a\u043e\u0432\u0430 \u043a\u0430\u043a\u043e\u0432\u043e \u043a\u0430\u043a\u043e\u0433\u043e \u043a\u0430\u043a\u043e\u0435 \u043a\u0430\u043a\u043e\u0439 \u043a\u0430\u043a\u043e\u043c \u043a\u0430\u043a\u043e\u043c\u0443 \u043a\u0430\u043a\u0443\u044e \u043a\u043e \u043a\u043e\u0433\u0434\u0430
    \u043a\u043e\u0433\u043e \u043a\u043e\u0439 \u043a\u043e\u043c \u043a\u043e\u043c\u0443 \u043a\u043e\u0442\u043e\u0440\u0430\u044f \u043a\u043e\u0442\u043e\u0440\u043e\u0433\u043e \u043a\u043e\u0442\u043e\u0440\u043e\u0435 \u043a\u043e\u0442\u043e\u0440\u043e\u0439 \u043a\u043e\u0442\u043e\u0440\u043e\u043c \u043a\u043e\u0442\u043e\u0440\u043e\u043c\u0443 \u043a\u043e\u0442\u043e\u0440\u043e\u044e \u043a\u043e\u0442\u043e\u0440\u0443\u044e \u043a\u043e\u0442\u043e\u0440\u044b\u0435
    \u043a\u043e\u0442\u043e\u0440\u044b\u0439 \u043a\u043e\u0442\u043e\u0440\u044b\u043c \u043a\u043e\u0442\u043e\u0440\u044b\u043c\u0438 \u043a\u043e\u0442\u043e\u0440\u044b\u0445 \u043a\u0442\u043e \u043a\u0443\u0434\u0430 \u043b\u0438 \u043b\u0438\u0431\u043e \u043b\u0438\u0448\u044c \u043c\u0435\u043d\u044f \u043c\u043d\u0435 \u043c\u043d\u0435 \u043c\u043d\u043e\u0439 \u043c\u043d\u043e\u044e \u043c\u043e\u0433 \u043c\u043e\u0433\u043b\u0430 \u043c\u043e\u0433\u043b\u0438
    \u043c\u043e\u0433\u0443 \u043c\u043e\u0433\u0443\u0442 \u043c\u043e\u0436\u0435\u0442 \u043c\u043e\u0439 \u043c\u043e\u0435\u0433\u043e \u043c\u043e\u0435\u0439 \u043c\u043e\u0435\u043c \u043c\u043e\u0435\u043c\u0443 \u043c\u043e\u0435\u044e \u043c\u043e\u0438\u043c \u043c\u043e\u0438\u043c\u0438 \u043c\u043e\u0438\u0445 \u043c\u043e\u0447\u044c \u043c\u043e\u044e \u043c\u043e\u044f \u043c\u043e\u0451 \u043c\u043e\u0451\u043c \u043c\u044b \u043d\u0430
    \u043d\u0430\u0434 \u043d\u0430\u0434\u043e \u043d\u0430\u0448 \u043d\u0430\u0448\u0430 \u043d\u0430\u0448\u0435 \u043d\u0430\u0448\u0435\u0433\u043e \u043d\u0430\u0448\u0435\u0439 \u043d\u0430\u0448\u0435\u043c \u043d\u0430\u0448\u0435\u043c\u0443 \u043d\u0430\u0448\u0438\u043c \u043d\u0430\u0448\u0438\u043c\u0438 \u043d\u0430\u0448\u0438\u0445 \u043d\u0430\u0448\u0443 \u043d\u0430\u0448\u0438 \u043d\u0430\u043c \u043d\u0430\u043c\u0438 \u043d\u0430\u0441 \u043d\u0435
    \u043d\u0435\u0433\u043e \u043d\u0435\u0435 \u043d\u0435\u043b\u044c\u0437\u044f \u043d\u0435\u043c \u043d\u0435\u0442 \u043d\u0435\u0451 \u043d\u0435\u0439 \u043d\u0435\u043c\u0443 \u043d\u0435\u044e \u043d\u0438\u043c \u043d\u0438\u043c\u0438 \u043d\u0438\u0445 \u043d\u0438 \u043d\u043e \u043d\u0443 \u043d\u0443\u0436\u043d\u043e \u043d\u0451\u043c \u043e \u043e\u0431 \u043e\u0431\u043e \u043e\u0434\u043d\u0430\u043a\u043e \u043e\u043d
    \u043e\u043d\u0430 \u043e\u043d\u0438 \u043e\u043d\u043e \u043e\u0442 \u043e\u0442\u043a\u0443\u0434\u0430 \u043e\u0447\u0435\u043d\u044c \u043f\u0435\u0440\u0435\u0434 \u043f\u043e \u043f\u043e\u0434 \u043f\u043e\u044d\u0442\u043e\u043c\u0443 \u043f\u043e\u0442\u043e\u043c\u0443 \u043f\u043e\u0447\u0435\u043c\u0443 \u043f\u0440\u0438 \u043f\u0440\u043e \u0441 \u0441\u0430\u043c\u043e \u0441\u0430\u043c\u0430 \u0441\u0430\u043c\u0438 \u0441\u0430\u043c\u0438\u043c
    \u0441\u0430\u043c\u0438\u0445 \u0441\u0430\u043c\u043e\u0433\u043e \u0441\u0430\u043c\u043e\u0439 \u0441\u0430\u043c\u043e\u043c \u0441\u0430\u043c\u043e\u043c\u0443 \u0441\u0430\u043c\u0443\u044e \u0441\u0430\u043c\u044b\u0439 \u0441\u0432\u043e\u0435 \u0441\u0432\u043e\u0435\u0433\u043e \u0441\u0432\u043e\u0435\u0439 \u0441\u0432\u043e\u0435\u043c \u0441\u0432\u043e\u0435\u043c\u0443 \u0441\u0432\u043e\u0435\u044e \u0441\u0432\u043e\u0438\u043c \u0441\u0432\u043e\u0438\u043c\u0438
    \u0441\u0432\u043e\u0438\u0445 \u0441\u0432\u043e\u0439 \u0441\u0432\u043e\u044e \u0441\u0432\u043e\u044f \u0441\u0432\u043e\u0451 \u0441\u0432\u043e\u0451\u043c \u0441\u0432\u043e\u0438 \u0441\u0435\u0431\u0435 \u0441\u0435\u0431\u044f \u0441\u043a\u043e\u043b\u044c\u043a\u043e \u0441\u043e \u0441\u043e\u0431\u043e\u0439 \u0441\u043e\u0431\u043e\u044e \u0442\u0430 \u0442\u0430\u043a \u0442\u0430\u043a\u0436\u0435 \u0442\u0430\u043a\u0430\u044f \u0442\u0430\u043a\u0438\u0435
    \u0442\u0430\u043a\u0438\u043c \u0442\u0430\u043a\u0438\u043c\u0438 \u0442\u0430\u043a\u0438\u0445 \u0442\u0430\u043a\u043e\u0433\u043e \u0442\u0430\u043a\u043e\u0435 \u0442\u0430\u043a\u043e\u0439 \u0442\u0430\u043a\u043e\u043c \u0442\u0430\u043a\u043e\u043c\u0443 \u0442\u0430\u043a\u0443\u044e \u0442\u0430\u043c \u0442\u0432\u043e\u0435 \u0442\u0432\u043e\u0435\u0433\u043e \u0442\u0432\u043e\u0435\u0439 \u0442\u0432\u043e\u0435\u043c \u0442\u0432\u043e\u0435\u043c\u0443
    \u0442\u0432\u043e\u0438\u043c \u0442\u0432\u043e\u0438\u043c\u0438 \u0442\u0432\u043e\u0438\u0445 \u0442\u0432\u043e\u0439 \u0442\u0432\u043e\u044e \u0442\u0432\u043e\u044f \u0442\u0432\u043e\u0451 \u0442\u0432\u043e\u0451\u043c \u0442\u0432\u043e\u0438 \u0442\u0435 \u0442\u0435\u0431\u0435 \u0442\u0435\u0431\u044f \u0442\u0435\u043c \u0442\u0435\u043c\u0438 \u0442\u0435\u043f\u0435\u0440\u044c \u0442\u0435\u0445 \u0442\u043e \u0442\u043e\u0431\u043e\u0439
    \u0442\u043e\u0431\u043e\u044e \u0442\u043e\u0433\u0434\u0430 \u0442\u043e\u0433\u043e \u0442\u043e\u0439 \u0442\u043e\u043b\u044c\u043a\u043e \u0442\u043e\u043c \u0442\u043e\u043c\u0443 \u0442\u043e\u0442 \u0442\u043e\u0436\u0435 \u0442\u0443 \u0442\u044b \u0442\u0443\u0442 \u0443 \u0443\u0436 \u0443\u0436\u0435 \u0445\u043e\u0442\u044f \u0447\u0435\u0433\u043e \u0447\u0435\u043c \u0447\u0435\u043c\u0443 \u0447\u0435\u0440\u0435\u0437 \u0447\u0442\u043e
    \u0447\u0442\u043e\u0431\u044b \u0447\u044c\u0435 \u0447\u044c\u0435\u0433\u043e \u0447\u044c\u0435\u0439 \u0447\u044c\u0438 \u0447\u044c\u0438\u0445 \u0447\u044c\u044e \u0447\u044c\u044f \u0447\u0451\u043c \u0447\u044c\u0451 \u044d\u0442\u0430 \u044d\u0442\u0438 \u044d\u0442\u0438\u043c \u044d\u0442\u0438\u043c\u0438 \u044d\u0442\u0438\u0445 \u044d\u0442\u043e \u044d\u0442\u043e\u0433\u043e \u044d\u0442\u043e\u0439 \u044d\u0442\u043e\u043c
    \u044d\u0442\u043e\u043c\u0443 \u044d\u0442\u043e\u0442 \u044d\u0442\u0443 \u044f""".split()
)

_WORD = re.compile(r"\w+")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]{2,}")
_WIDE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_RUSSIAN = re.compile(r"[\u0430-\u044f\u0451]+")
_RU_RV = re.compile(r"[^\u0430\u0435\u0438\u043e\u0443\u044b\u044d\u044e\u044f]*[\u0430\u0435\u0438\u043e\u0443\u044b\u044d\u044e\u044f]")
_RU_R2 = re.compile(r"(?:[^\u0430\u0435\u0438\u043e\u0443\u044b\u044d\u044e\u044f]*[\u0430\u0435\u0438\u043e\u0443\u044b\u044d\u044e\u044f]+[^\u0430\u0435\u0438\u043e\u0443\u044b\u044d\u044e\u044f]){2}")

# The endings of the Snowball Russian stemmer. Groups named _AFTER_A count only after \u0430 or \u044f.
_RU_GERUND_AFTER_A = frozenset("\u0432 \u0432\u0448\u0438 \u0432\u0448\u0438\u0441\u044c".split())
_RU_GERUND = frozenset("\u0438\u0432 \u0438\u0432\u0448\u0438 \u0438\u0432\u0448\u0438\u0441\u044c \u044b\u0432 \u044b\u0432\u0448\u0438 \u044b\u0432\u0448\u0438\u0441\u044c".split())
_RU_REFLEXIVE = frozenset("\u0441\u044f \u0441\u044c".split())
_RU_ADJECTIVE = frozenset("\u0435\u0435 \u0438\u0435 \u044b\u0435 \u043e\u0435 \u0438\u043c\u0438 \u044b\u043c\u0438 \u0435\u0439 \u0438\u0439 \u044b\u0439 \u043e\u0439 \u0435\u043c \u0438\u043c \u044b\u043c \u043e\u043c \u0435\u0433\u043e \u043e\u0433\u043e \u0435\u043c\u0443 \u043e\u043c\u0443 \u0438\u0445 \u044b\u0445 \u0443\u044e \u044e\u044e \u0430\u044f \u044f\u044f \u043e\u044e \u0435\u044e".split())
_RU_PARTICIPLE_AFTER_A = frozenset("\u0435\u043c \u043d\u043d \u0432\u0448 \u044e\u0449 \u0449".split())
_RU_PARTICIPLE = frozenset("\u0438\u0432\u0448 \u044b\u0432\u0448 \u0443\u044e\u0449".split())
_RU_VERB_AFTER_A = frozenset("\u043b\u0430 \u043d\u0430 \u0435\u0442\u0435 \u0439\u0442\u0435 \u043b\u0438 \u0439 \u043b \u0435\u043c \u043d \u043b\u043e \u043d\u043e \u0435\u0442 \u044e\u0442 \u043d\u044b \u0442\u044c \u0435\u0448\u044c \u043d\u043d\u043e".split())
_RU_VERB = frozenset(
    "\u0438\u043b\u0430 \u044b\u043b\u0430 \u0435\u043d\u0430 \u0435\u0439\u0442\u0435 \u0443\u0439\u0442\u0435 \u0438\u0442\u0435 \u0438\u043b\u0438 \u044b\u043b\u0438 \u0435\u0439 \u0443\u0439 \u0438\u043b \u044b\u043b \u0438\u043c \u044b\u043c \u0435\u043d \u0438\u043b\u043e \u044b\u043b\u043e \u0435\u043d\u043e \u044f\u0442 \u0443\u0435\u0442 \u0443\u044e\u0442 \u0438\u0442 \u044b\u0442 \u0435\u043d\u044b \u0438\u0442\u044c \u044b\u0442\u044c \u0438\u0448\u044c \u0443\u044e \u044e".split()
)
_RU_NOUN = frozenset(
    "\u0430 \u0435\u0432 \u043e\u0432 \u0438\u0435 \u044c\u0435 \u0435 \u0438\u044f\u043c\u0438 \u044f\u043c\u0438 \u0430\u043c\u0438 \u0435\u0438 \u0438\u0438 \u0438 \u0438\u0435\u0439 \u0435\u0439 \u043e\u0439 \u0438\u0439 \u0439 \u0438\u044f\u043c \u044f\u043c \u0438\u0435\u043c \u0435\u043c \u0430\u043c \u043e\u043c \u043e \u0443 \u0430\u0445 \u0438\u044f\u0445 \u044f\u0445 \u044b \u044c \u0438\u044e \u044c\u044e \u044e \u0438\u044f \u044c\u044f \u044f".split()
)
_RU_SUPERLATIVE = frozenset("\u0435\u0439\u0448 \u0435\u0439\u0448\u0435".split())
_RU_DERIVATIONAL = frozenset("\u043e\u0441\u0442 \u043e\u0441\u0442\u044c".split())
_RU_LONGEST_ENDING = 6


def split_identifier(word: str) -> list[str]:
    """Split ``parseHTTPResponse_v2`` into its lower-case word parts."""
    parts: list[str] = []
    for chunk in word.split("_"):
        if chunk:
            parts.extend(p.lower() for p in _CAMEL.split(chunk) if p)
    return parts


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens, with code identifiers also split into their parts."""
    out: list[str] = []
    for word in _WORD.findall(text):
        lower = word.lower()
        out.append(lower)
        parts = split_identifier(word)
        if len(parts) > 1:
            out.extend(parts)
        for run in _CJK.findall(word):  # no spaces between words in these scripts, so index character pairs
            out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


def content_terms(text: str) -> list[str]:
    """Distinct non-stopword tokens in first-seen order."""
    seen: dict[str, None] = {}
    for token in tokenize(text):
        if token not in STOPWORDS and len(token) > 1:
            seen.setdefault(token, None)
    return list(seen)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s and s.strip()]


def approx_tokens(text: str) -> int:
    """Dependency-free token estimate, calibrated on cl100k_base over English, code, Russian, Chinese and hashes."""
    if not text:
        return 0
    words = _WORD.findall(text)
    punctuation = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    wide = len(_WIDE.findall(text))
    other = len(_NON_ASCII.findall(text)) - wide
    ascii_chars = len(text) - wide - other
    hard = sum(len(w) for w in words if len(w) >= 12 and sum(c.isdigit() for c in w) >= 3)
    estimate = 0.75 * ascii_chars / 4 + 0.25 * (1.3 * len(words) + punctuation) + other / 2.2 + wide * 1.1 + hard * 0.37
    return max(1, math.ceil(estimate))


def tiktoken_counter(encoding: str = "cl100k_base") -> Callable[[str], int]:
    import tiktoken

    enc = tiktoken.get_encoding(encoding)
    return lambda text: len(enc.encode(text, disallowed_special=()))


def _without_ending(word: str, start: int, endings: frozenset[str], after_a: frozenset[str] = frozenset()) -> str | None:
    """``word`` without its longest ending from these groups inside ``word[start:]``, or ``None``.

    Only the longest ending is tried: one from ``after_a`` that does not follow а or я means ``None``.
    """
    for size in range(min(len(word) - start, _RU_LONGEST_ENDING), 0, -1):
        ending = word[-size:]
        if ending in endings:
            return word[:-size]
        if ending in after_a:
            return word[:-size] if size < len(word) - start and word[-size - 1] in "ая" else None
    return None


@lru_cache(maxsize=1 << 16)
def russian_stem(word: str) -> str:
    """The Snowball Russian stem of a lower-case word. The result is not always a fixed point of this function."""
    word = word.replace("ё", "е")
    found = _RU_RV.match(word)
    rv = found.end() if found else len(word)
    found = _RU_R2.match(word)
    r2 = found.end() if found else len(word)
    stem = _without_ending(word, rv, _RU_GERUND, _RU_GERUND_AFTER_A)
    if stem is None:
        stem = _without_ending(word, rv, _RU_REFLEXIVE) or word
        adjective = _without_ending(stem, rv, _RU_ADJECTIVE)
        if adjective is not None:
            stem = _without_ending(adjective, rv, _RU_PARTICIPLE, _RU_PARTICIPLE_AFTER_A) or adjective
        else:
            stem = _without_ending(stem, rv, _RU_VERB, _RU_VERB_AFTER_A) or _without_ending(stem, rv, _RU_NOUN) or stem
    if stem[rv:].endswith("и"):
        stem = stem[:-1]
    stem = _without_ending(stem, r2, _RU_DERIVATIONAL) or stem
    plain = _without_ending(stem, rv, _RU_SUPERLATIVE)
    stem = plain or stem
    if stem[rv:].endswith("нн") or (plain is None and stem[rv:].endswith("ь")):
        stem = stem[:-1]
    return stem


def search_text(text: str) -> str:
    """Text as seen by FTS5: the original, split identifier parts and the stems of Russian words."""
    tokens: list[str] = []
    for token in tokenize(text):
        tokens.append(token)
        if _RUSSIAN.fullmatch(token) and russian_stem(token) != token:  # FTS5's porter stems English only
            tokens.append(russian_stem(token))
    return " ".join(tokens)


def fts_query(terms: Iterable[str]) -> str:
    """An OR over ``terms``, plus the stems of Russian words, which is how ``search_text`` indexes them."""
    terms = list(terms)
    stems = dict.fromkeys(russian_stem(t) for t in terms if _RUSSIAN.fullmatch(t))
    quoted = ['"' + t.replace('"', '""') + '"' for t in terms + [s for s in stems if s not in terms]]
    return " OR ".join(quoted)


def light_stem(token: str) -> str:
    """Conservative suffix stripping so that facet matching tolerates inflection. Russian words get their full stem."""
    if _RUSSIAN.fullmatch(token):
        return russian_stem(token)
    if len(token) > 4:
        if token.endswith("ies"):
            return token[:-3] + "y"
        if token.endswith("ing") and len(token) > 5:
            return token[:-3]
        if token.endswith("ed") and len(token) > 5:
            return token[:-2]
        if token.endswith("es") and len(token) > 5:
            return token[:-2]
        if token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
    return token


def query_snippet(text: str, stems: list[str], limit: int, count: Callable[[str], int]) -> str:
    """The sentences of ``text`` that cover the most query stems, in original order, within ``limit`` tokens."""
    sentences = split_sentences(text)
    wanted = set(stems)
    scored = []
    for index, sentence in enumerate(sentences):
        have = {light_stem(t) for t in tokenize(sentence)}
        scored.append((len(have & wanted), -index, index))
    chosen: list[int] = []
    used = 0
    covered: set[str] = set()
    for _, _, index in sorted(scored, reverse=True):
        cost = count(sentences[index])
        if used + cost > limit:
            continue
        chosen.append(index)
        used += cost
        covered |= {light_stem(t) for t in tokenize(sentences[index])} & wanted
    if not chosen:
        words = text.split()
        return " ".join(words[: max(limit // 2, 1)]) + " ..."
    return " ... ".join(sentences[i] for i in sorted(chosen))


def query_window(
    text: str, terms: list[str], weights: list[float], limit: int, count: Callable[[str], int]
) -> tuple[str, float, float]:
    """The run of consecutive sentences within ``limit`` tokens that covers the most query weight.

    Returns the excerpt, the weight it covers and the weight the whole text covers. ``terms`` are stems.
    """
    sentences = split_sentences(text)
    weight = dict(zip(terms, weights))
    have = [{light_stem(t) for t in tokenize(s)} & weight.keys() for s in sentences]
    total = sum(weight[t] for t in set().union(*have)) if have else 0.0
    costs = [count(s) for s in sentences]
    best, best_span = -1.0, (0, 0)
    for start in range(len(sentences)):
        covered: set[str] = set()
        used = 0
        for end in range(start, len(sentences)):
            if used + costs[end] > limit:
                break
            used += costs[end]
            covered |= have[end]
            value = sum(weight[t] for t in covered)
            if value > best:
                best, best_span = value, (start, end + 1)
    if best < 0:
        words = text.split()
        return " ".join(words[: max(limit // 2, 1)]) + " ...", 0.0, total
    start, end = best_span
    excerpt = " ".join(sentences[start:end])
    if start > 0:
        excerpt = "... " + excerpt
    if end < len(sentences):
        excerpt += " ..."
    return excerpt, best, total
