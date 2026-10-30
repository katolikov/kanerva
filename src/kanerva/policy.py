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
)

DEFAULT_THETA = (0.0045, -0.0863, -0.0309, 0.0345, -0.0048, -0.0215, 0.0032, 0.0, 0.1559, 0.0934, 0.2436, 0.1367)


@dataclass(frozen=True)
class RetrievalConfig:
    candidates: int = 100
    rank_kappa: float = 10.0
    item_overhead: int = 8

    modular: float = 0.5  # eta: weight of the additive relevance term
    query_facet: float = 0.5  # omega_0: facet weight of the whole query
    gamma: float = 0.2  # weight of the facility-location term
    cost_exponents: tuple[float, ...] = (1.0, 0.5, 0.0)
    min_weight: float = 0.0
    floor: float = 0.0  # weight floor proportional to rank signals; lets the tail fill spare budget

    use_validity: bool = True
    use_supersession: bool = True
    supersession_threshold: float = 0.8
    supersession_strength: float = 0.8
    stale_factor: float = 0.05

    use_bandit: bool = True
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


def feature_matrix(c: Candidates, cfg: RetrievalConfig) -> np.ndarray:
    n = len(c.ids)
    lex = c.lex_score / (c.lex_score.max() or 1.0) if n else c.lex_score
    lex_rank = 1.0 / (1.0 + c.lex_rank / cfg.rank_kappa)
    dense_rank = 1.0 / (1.0 + c.dense_rank / cfg.rank_kappa)
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
        ]
    )
    return x * c.validity[:, None]
