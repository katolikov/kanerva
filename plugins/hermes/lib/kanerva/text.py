"""Tokenisation helpers shared by the lexical index, selection and feedback."""

from __future__ import annotations

import math
import re
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
)

_WORD = re.compile(r"\w+")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]{2,}")
_WIDE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


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


def search_text(text: str) -> str:
    """Text as seen by FTS5: the original plus split identifier parts."""
    return " ".join(tokenize(text))


def fts_query(terms: Iterable[str]) -> str:
    quoted = ['"' + t.replace('"', '""') + '"' for t in terms]
    return " OR ".join(quoted)


def light_stem(token: str) -> str:
    """Conservative suffix stripping so that facet matching tolerates inflection."""
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
