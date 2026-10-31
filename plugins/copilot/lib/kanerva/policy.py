"""Retrieval configuration and item feature computation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

FEATURES = (
    "bias",
    "lex_rank",
    "dense_rank",
    "lex_score",
    "dense_score",
    "term_cover",
    "activation",
    "usefulness",
    "rank_agreement",
    "full_cover",
    "lex_top",
    "dense_top",
    "rerank_top",
    "rerank_rank",
    "rerank_score",
)

DEFAULT_THETA = (0.0048, -0.0876, -0.0289, 0.0381, -0.006, -0.0223, 0.0026, 0.0, 0.1523, 0.0938, 0.2418, 0.1369, 0.0, 0.0, 0.0)
LEXICAL_THETA = (-0.021, -0.0316, 0.0, 0.0407, 0.0, 0.0239, 0.0227, 0.0, 0.0, 0.07, 0.3406, 0.0, 0.0, 0.0, 0.0)
RERANK_THETA = (0.0013, -0.0339, 0.0062, 0.0303, -0.0071, -0.0156, 0.0027, 0.0, 0.0066, 0.0336, 0.0993, 0.0534, 0.485, -0.0877, 0.0243)


@dataclass(frozen=True)
class RetrievalConfig:
    candidates: int = 200
    neighbour_top: int = 50  # the items next to this many best fused candidates join the pool
    neighbour_reach: int = 1
    rank_kappa: float = 10.0
    item_overhead: int = 14  # the rendered prefix '[#id date] ' costs 11 to 15 tokens

    use_dense: bool = True
    use_reranker: bool = False
    rerank_pool: int = 50

    modular: float = 0.0  # eta: weight of the additive relevance term
    query_facet: float = 0.5  # omega_0: facet weight of the whole query
    gamma: float = 0.0  # weight of the facility-location term; no effect on dev, so off by default
    cost_exponents: tuple[float, ...] = (1.0, 0.5)
    min_weight: float = 0.0
    floor: float = 0.002  # weight floor proportional to rank signals; lets the tail fill spare budget

    snippet_threshold: int = 0  # items above this many tokens are injected as query-biased snippets
    snippet_tokens: int = 40
    excerpt_threshold: int = 0  # items above this many tokens may also be chosen as an excerpt of excerpt_tokens;
    # 160 is the tested setting
    excerpt_tokens: int = 128
    newest_first: bool = True  # list chosen items newest first, so a reader meets the latest version of a fact first

    use_validity: bool = True
    drop_superseded: bool = True  # exact version chains: old versions leave unless the query asks for history
    use_supersession: bool = True
    supersession_threshold: float = 0.8
    supersession_strength: float = 0.8
    supersession_drop: float = 0.97  # an unkeyed item with a newer twin this similar is removed outright
    hashed_drop: float = 0.85  # the same for hashed words, which rate two versions of one fact lower than a sentence model
    soft_supersede_keyed: bool = False  # keyed items have exact version chains, so similarity is not evidence
    hash_twins: bool = False  # compare twins by hashed words even when items have vectors
    word_twins: bool = True  # compare them both ways, since a sentence model hears little difference between FOO = 3 and FOO = 4
    twin_exact: bool = True  # a twin must also hold the query's words whole: dictAddRaw says nothing about dictAdd
    supersede_events: bool = False  # two similar events are two events; only facts can replace each other
    twin_needs_query: bool = True  # a newer twin replaces an older item only if it covers the query at least as well
    stale_factor: float = 0.05

    # "trees": the calibrated relevance model of kanerva.relevance; "linear": a linear prior over FEATURES.
    relevance: str = "trees"
    model: str = ""  # relevance model name; empty picks rerank, dense, static or lexical from the setup

    use_bandit: bool = True
    learn: bool = True
    item_lift: bool = False  # scale each item by how often it helped against how often the model expected it to
    lift_prior: float = 5.0  # expected uses at which an item's own record weighs as much as the model
    bandit_beta: float = 0.0
    bandit_ridge: float = 16.0
    theta0: tuple[float, ...] = DEFAULT_THETA

    def with_(self, **changes) -> "RetrievalConfig":
        return replace(self, **changes)


@dataclass
class Candidates:
    """Per-candidate retrieval signals; missing ranks are ``None``-coded as infinity."""

    ids: list[int] = field(default_factory=list)
    lex_rank: np.ndarray | None = None
    dense_rank: np.ndarray | None = None
    lex_score: np.ndarray | None = None
    dense_score: np.ndarray | None = None
    term_cover: np.ndarray | None = None
    activation: np.ndarray | None = None
    usefulness: np.ndarray | None = None
    lift: np.ndarray | None = None
    validity: np.ndarray | None = None
    rerank_rank: np.ndarray | None = None
    rerank_score: np.ndarray | None = None
    relevance: np.ndarray | None = None  # rows of kanerva.relevance.FEATURES


def subset(c: Candidates, keep: np.ndarray) -> Candidates:
    """The candidates selected by a boolean mask."""
    def pick(values: np.ndarray | None) -> np.ndarray | None:
        return None if values is None else values[keep]

    return Candidates(
        ids=[i for i, k in zip(c.ids, keep) if k],
        lex_rank=pick(c.lex_rank),
        dense_rank=pick(c.dense_rank),
        lex_score=pick(c.lex_score),
        dense_score=pick(c.dense_score),
        term_cover=pick(c.term_cover),
        activation=pick(c.activation),
        usefulness=pick(c.usefulness),
        lift=pick(c.lift),
        validity=pick(c.validity),
        rerank_rank=pick(c.rerank_rank),
        rerank_score=pick(c.rerank_score),
        relevance=pick(c.relevance),
    )


QUERY_SIGNALS = ("cover", "terms", "weight", "lex_spread", "dense", "dense_gap")
LEXICAL_SIGNALS = QUERY_SIGNALS[:4]


def query_signals(c: Candidates, weights: np.ndarray | None, terms: int) -> np.ndarray:
    """What the result list says about the whole query, in ``QUERY_SIGNALS`` order: is any stored item about this?"""
    if not c.ids or weights is None:
        return np.array([0.0, np.log1p(terms), 0.0, 0.0, 0.0, 0.0])
    scores = np.sort(c.lex_score[c.lex_score > 0])[::-1][:50]
    spread = float(scores.std() / scores.mean()) if scores.size > 1 else 0.0
    dense = np.sort(c.dense_score)[::-1][:50]
    return np.array([
        float(c.term_cover.max()), np.log1p(terms), float(weights.max()), spread, float(dense[0]), float(dense[0] - dense.mean()),
    ])


@dataclass(frozen=True)
class QueryGate:
    """Logistic model over ``query_signals``: below ``threshold`` the store is taken to have nothing on this query."""

    use: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    intercept: float
    threshold: float

    def score(self, signals: np.ndarray) -> np.ndarray:
        return (signals - np.asarray(self.mean)) / np.asarray(self.scale) @ np.asarray(self.weights) + self.intercept


QUERY_GATES = {
    "lexical": QueryGate(
        LEXICAL_SIGNALS,
        (0.35406, 1.8615, 0.26058, 0.43963),
        (0.28348, 0.35278, 0.25658, 0.62576),
        (1.5323, 0.56487, 1.069, 0.30833),
        -1.1494, -2.316,
    ),
    "full": QueryGate(
        QUERY_SIGNALS,
        (0.35425, 1.8615, 0.35096, 0.43377, 0.42102, 0.13968),
        (0.28332, 0.35278, 0.26691, 0.61765, 0.21383, 0.095266),
        (0.85609, 0.40221, 0.697, 0.20268, 2.1096, 0.28618),
        -1.61, -1.6036,
    ),
}

BANDIT_FEATURES = ("bias", "prior", "usefulness")
BANDIT_THETA = (0.0, 1.0, 0.0)


def bandit_features(prior: np.ndarray, c: Candidates) -> np.ndarray:
    """What the online learner sees when the relevance model supplies the prior.

    The bias absorbs the base rate of the reward, including false positives from a noisy signal. It is zeroed when
    scoring items, since a constant added to every item would push the density rule toward short items.
    """
    return np.column_stack([np.ones(len(prior)), prior, c.usefulness - 0.5])


def feature_matrix(c: Candidates, cfg: RetrievalConfig) -> np.ndarray:
    n = len(c.ids)
    lex = c.lex_score / (c.lex_score.max() or 1.0) if n else c.lex_score
    lex_rank = 1.0 / (1.0 + c.lex_rank / cfg.rank_kappa)
    dense_rank = 1.0 / (1.0 + c.dense_rank / cfg.rank_kappa)
    rerank_rank = np.full(n, np.inf) if c.rerank_rank is None else c.rerank_rank
    rerank_score = np.zeros(n) if c.rerank_score is None else c.rerank_score
    x = np.column_stack(
        [
            np.ones(n),
            lex_rank,
            dense_rank,
            lex,
            np.clip(c.dense_score, 0.0, 1.0),
            c.term_cover,
            c.activation,
            c.usefulness - 0.5,
            lex_rank * dense_rank,
            c.term_cover**2,
            1.0 / (1.0 + c.lex_rank),
            1.0 / (1.0 + c.dense_rank),
            1.0 / (1.0 + rerank_rank),
            1.0 / (1.0 + rerank_rank / cfg.rank_kappa),
            rerank_score,
        ]
    )
    return x
