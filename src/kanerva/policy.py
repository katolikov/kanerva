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
    candidates: int = 100
    neighbour_top: int = 0  # the items next to this many best fused candidates join the pool
    neighbour_reach: int = 1
    rank_kappa: float = 10.0
    item_overhead: int = 14  # the rendered prefix '[#id date] ' costs 11 to 15 tokens

    use_dense: bool = True
    use_reranker: bool = False
    rerank_pool: int = 50

    modular: float = 0.0  # eta: weight of the additive relevance term
    query_facet: float = 0.5  # omega_0: facet weight of the whole query
    gamma: float = 0.05  # weight of the facility-location term
    cost_exponents: tuple[float, ...] = (1.0, 0.5)
    min_weight: float = 0.0
    floor: float = 0.002  # weight floor proportional to rank signals; lets the tail fill spare budget

    snippet_threshold: int = 0  # items above this many tokens are injected as query-biased snippets
    snippet_tokens: int = 40
    excerpt_threshold: int = 0  # items above this many tokens may also be chosen as an excerpt of excerpt_tokens
    excerpt_tokens: int = 96

    use_validity: bool = True
    drop_superseded: bool = True  # exact version chains: old versions leave unless the query asks for history
    use_supersession: bool = True
    supersession_threshold: float = 0.8
    supersession_strength: float = 0.8
    supersession_drop: float = 0.97  # an unkeyed item with a newer twin this similar is removed outright
    soft_supersede_keyed: bool = False  # keyed items have exact version chains, so similarity is not evidence
    stale_factor: float = 0.05

    # "trees": the calibrated relevance model of kanerva.relevance; "linear": the round-1 linear prior over FEATURES.
    relevance: str = "trees"

    use_bandit: bool = True
    learn: bool = True
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
        validity=pick(c.validity),
        rerank_rank=pick(c.rerank_rank),
        rerank_score=pick(c.rerank_score),
        relevance=pick(c.relevance),
    )


BANDIT_FEATURES = ("prior", "usefulness")
BANDIT_THETA = (1.0, 0.0)


def bandit_features(prior: np.ndarray, c: Candidates) -> np.ndarray:
    """What the online learner sees when the relevance model supplies the prior: its probability and past use."""
    return np.column_stack([prior, c.usefulness - 0.5])


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
