"""Relevance: P(item is needed | query) from retrieval, item, neighbourhood and query-item signals.

The shipped models are gradient-boosted trees, stored as plain arrays so that scoring needs only NumPy. They are
fitted on development data by ``experiments/fit_relevance.py`` and shipped in ``relevance_model.json``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .text import STOPWORDS, content_terms, light_stem, tokenize

_ENTITY = re.compile(r"\b[A-Z][A-Za-z0-9_\-]+")
_NUMBER = re.compile(r"\d+(?:[.,:/]\d+)*")
_RU_MONTH = (
    r"(январ[ьяе]|феврал[ьяе]|март[ае]?|апрел[ьяе]|ма[йяе]|июн[ьяе]|июл[ьяе]|август[ае]?|"
    r"сентябр[ьяе]|октябр[ьяе]|ноябр[ьяе]|декабр[ьяе])"
)
_DATE = re.compile(
    r"\b(19|20)\d\d\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d|"
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|yesterday|today|tomorrow|week|weekend)\b|"
    rf"\b(19|20)\d\dг|\b\d{{1,2}}\s+{_RU_MONTH}\b|\b{_RU_MONTH}\s+\d|"
    r"\b(понедельник\w*|вторник\w*|сред[ау]|четверг\w*|пятниц\w*|суббот\w*|воскресень\w*|вчера|сегодня|завтра|"
    r"недел(я|и|е|ю|ь|ям|ях|ями)|на выходных|в выходные)\b",
    re.I,
)
_TEMPORAL = re.compile(
    r"\b(when|date|day|days|week|weeks|month|months|year|years|ago|last|first|recent|recently|latest|"
    r"before|after|since|until|how long|earlier|later|currently|now)\b|"
    r"\b(когда|дат[аыеу]|день|дня|дню|дни|дней|днях|недел(я|и|е|ю|ь|ям|ях|ями)|месяц(а|у|е|ы|ев|ах)?|год(а|у|ы|ов|ах)?|"
    r"лет|назад|последн\w+|перв(ый|ая|ое|ые|ого|ой|ую|ым|ых|ом)|недавн\w+|новейш\w+|раньше|ранее|позже|позднее|после|"
    r"с тех пор|как долго|сейчас|теперь)\b",
    re.I,
)
_HOW_MANY = re.compile(
    r"\bhow (many|much|long|often|old|far)\b|\bwhat (number|time|percentage)\b|"
    r"\bсколько\b|\bкак (долго|часто|давно|далеко)\b|\bкакой (номер|процент)\b|\bкакое (число|количество|время)\b",
    re.I,
)
_WHEN = re.compile(
    r"\bwhen\b|\bwhat (date|day|year|month|time)\b|"
    r"\bкогда\b|\bв как(ом|ой|ую) (году|месяце|день|дату)\b|\bв котором (году|часу)\b|"
    r"\bкак(ой|ая|ое|ого) (год|года|дата|день|месяц|число|числа)\b",
    re.I,
)
_FIRST_PERSON = re.compile(
    r"\b(I|my|me|mine|myself)\b|\bI'(m|ve|d|ll)\b|"
    r"\b(?i:я|меня|мне|мной|мною|мой|моя|моё|мое|мои|моего|моей|моему|моим|моём|моем|моих|моими|мою)\b"
)
_SPEAKER = re.compile(r"^\s*([A-Za-z][\w .'-]{0,30}?):\s")

FEATURES = (
    "bias",
    "lex_rank",
    "dense_rank",
    "lex_score",
    "dense_score",
    "term_cover",
    "rank_agreement",
    "lex_top",
    "dense_top",
    "log_tokens",
    "short",
    "content_density",
    "is_question",
    "has_number",
    "role_user",
    "role_assistant",
    "first_person_user",
    "speaker_in_query",
    "entity_overlap",
    "number_overlap",
    "neighbour_dense",
    "neighbour2_dense",
    "session_dense",
    "session_hits",
    "dense_gap",
    "recency",
    "temporal_recency",
    "when_date",
    "count_number",
    "activation",
    "rerank_score",
    "rerank_rank",
    "rerank_top",
)
RERANK_FEATURES = ("rerank_score", "rerank_rank", "rerank_top")


@dataclass
class ItemView:
    """What the feature map needs to know about one stored item."""

    text: str
    tokens: int
    created_at: float
    session: str | None


def _speaker(text: str) -> str | None:
    m = _SPEAKER.match(text)
    return m.group(1).strip().lower() if m else None


def _entities(text: str) -> set[str]:
    """Capitalised words that are not stopwords: names, products, places, identifiers."""
    return {e.lower() for e in _ENTITY.findall(text)} - STOPWORDS - {"user", "assistant"}


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


def feature_matrix(
    query: str,
    items: list[ItemView],
    lex_rank: np.ndarray,
    dense_rank: np.ndarray,
    lex_score: np.ndarray,
    dense_score: np.ndarray,
    term_cover: np.ndarray,
    activation: np.ndarray,
    neighbour_dense: np.ndarray,
    neighbour2_dense: np.ndarray,
    session_dense: np.ndarray,
    rerank_rank: np.ndarray | None = None,
    rerank_score: np.ndarray | None = None,
    kappa: float = 10.0,
) -> np.ndarray:
    """One row per candidate, columns in ``FEATURES`` order. Missing ranks are ``inf``."""
    n = len(items)
    x = np.zeros((n, len(FEATURES)))
    if n == 0:
        return x
    lex_f = 1.0 / (1.0 + lex_rank / kappa)
    dense_f = 1.0 / (1.0 + dense_rank / kappa)
    q_entities = _entities(query)
    q_numbers = _numbers(query)
    q_first = bool(_FIRST_PERSON.search(query))
    q_temporal = bool(_TEMPORAL.search(query))
    q_when = bool(_WHEN.search(query))
    q_count = bool(_HOW_MANY.search(query))
    q_lower = query.lower()
    created = np.array([it.created_at for it in items])
    order = np.argsort(np.argsort(-created, kind="stable"), kind="stable")
    recency = 1.0 / (1.0 + order / kappa)
    sessions: dict[str | None, int] = {}
    for it in items:
        sessions[it.session] = sessions.get(it.session, 0) + 1
    top_dense = float(dense_score.max()) if n else 0.0
    for idx, it in enumerate(items):
        text = it.text
        terms = content_terms(text)
        speaker = _speaker(text)
        has_number = bool(_NUMBER.search(text))
        has_date = bool(_DATE.search(text))
        row = x[idx]
        row[9] = math.log1p(it.tokens) / math.log(1000.0)
        row[10] = float(it.tokens <= 12)
        row[11] = min(len(terms) / max(it.tokens, 1), 1.0)
        row[12] = float("?" in text[-200:])
        row[13] = float(has_number)
        row[14] = float(speaker == "user")
        row[15] = float(speaker == "assistant")
        row[16] = float(q_first and speaker == "user")
        row[17] = float(bool(speaker) and speaker not in ("user", "assistant") and speaker in q_lower)
        if q_entities:
            row[18] = len(q_entities & _entities(text)) / len(q_entities)
        if q_numbers:
            row[19] = len(q_numbers & _numbers(text)) / len(q_numbers)
        row[23] = math.log1p(sessions[it.session] - 1) / math.log(50.0) if it.session is not None else 0.0
        row[27] = float(q_when and has_date)
        row[28] = float(q_count and has_number)
    x[:, 0] = 1.0
    x[:, 1] = lex_f
    x[:, 2] = dense_f
    x[:, 3] = lex_score / (lex_score.max() or 1.0)
    x[:, 4] = np.clip(dense_score, 0.0, 1.0)
    x[:, 5] = term_cover
    x[:, 6] = lex_f * dense_f
    x[:, 7] = 1.0 / (1.0 + lex_rank)
    x[:, 8] = 1.0 / (1.0 + dense_rank)
    x[:, 20] = np.clip(neighbour_dense, 0.0, 1.0)
    x[:, 21] = np.clip(neighbour2_dense, 0.0, 1.0)
    x[:, 22] = np.clip(session_dense, 0.0, 1.0)
    x[:, 24] = np.clip(dense_score, 0.0, 1.0) - max(top_dense, 0.0)
    x[:, 25] = recency
    x[:, 26] = recency * float(q_temporal)
    x[:, 29] = activation
    if rerank_rank is not None and rerank_score is not None:
        x[:, 30] = rerank_score
        x[:, 31] = 1.0 / (1.0 + rerank_rank / kappa)
        x[:, 32] = 1.0 / (1.0 + rerank_rank)
    return x


@dataclass
class RelevanceModel:
    """Logistic model over standardised features; columns not in ``use`` get weight zero."""

    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    intercept: float
    use: tuple[str, ...]

    def predict(self, x: np.ndarray) -> np.ndarray:
        cols = [FEATURES.index(f) for f in self.use]
        z = (x[:, cols] - self.mean) / self.scale
        return 1.0 / (1.0 + np.exp(-(z @ self.weights + self.intercept)))

    def to_json(self) -> dict:
        return {"use": list(self.use), "weights": self.weights.tolist(), "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "intercept": self.intercept}

    @classmethod
    def from_json(cls, data: dict) -> "RelevanceModel":
        return cls(np.asarray(data["weights"]), np.asarray(data["mean"]), np.asarray(data["scale"]),
                   float(data["intercept"]), tuple(data["use"]))


@dataclass
class TreeModel:
    """Gradient-boosted trees, stored as padded node arrays and evaluated level by level.

    ``feature``, ``threshold``, ``left``, ``right`` and ``value`` have shape (trees, max_nodes). A leaf points to
    itself, so every row reaches its leaf after ``depth`` steps. Samples go left when ``x <= threshold``.
    """

    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray
    baseline: float
    depth: int
    use: tuple[str, ...]

    def predict(self, x: np.ndarray) -> np.ndarray:
        cols = [FEATURES.index(f) for f in self.use]
        z = x[:, cols]
        n = z.shape[0]
        trees = np.arange(self.feature.shape[0])[:, None]
        node = np.zeros((self.feature.shape[0], n), dtype=np.int64)
        rows = np.arange(n)[None, :]
        for _ in range(self.depth):
            f = self.feature[trees, node]
            go_left = z[rows, f] <= self.threshold[trees, node]
            node = np.where(go_left, self.left[trees, node], self.right[trees, node])
        raw = self.baseline + self.value[trees, node].sum(axis=0)
        return 1.0 / (1.0 + np.exp(-raw))

    def to_json(self) -> dict:
        return {"kind": "trees", "use": list(self.use), "baseline": self.baseline, "depth": self.depth,
                **{k: getattr(self, k).tolist() for k in ("feature", "threshold", "left", "right", "value")}}

    @classmethod
    def from_json(cls, data: dict) -> "TreeModel":
        return cls(np.asarray(data["feature"], dtype=np.int64), np.asarray(data["threshold"], dtype=float),
                   np.asarray(data["left"], dtype=np.int64), np.asarray(data["right"], dtype=np.int64),
                   np.asarray(data["value"], dtype=float), float(data["baseline"]), int(data["depth"]),
                   tuple(data["use"]))


MODEL_PATH = Path(__file__).with_name("relevance_model.json")


@lru_cache(maxsize=4)
def load_model(name: str) -> RelevanceModel | TreeModel | None:
    """The shipped model ``name`` (for example ``dense``, ``lexical`` or ``rerank``), or ``None`` if absent.

    ``file.json#name`` reads another model file, which is how an experiment compares against earlier models.
    """
    path, _, name = name.rpartition("#")
    source = Path(path) if path else MODEL_PATH
    if not source.exists():
        return None
    models = json.loads(source.read_text())
    if name not in models:
        return None
    data = models[name]
    return TreeModel.from_json(data) if data.get("kind") == "trees" else RelevanceModel.from_json(data)


def stems(text: str) -> set[str]:
    return {light_stem(t) for t in tokenize(text)}
