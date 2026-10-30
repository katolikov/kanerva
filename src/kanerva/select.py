"""Budgeted monotone submodular selection.

The objective over candidate items is

    F(S) = (1 - gamma) * (R(S) + eta * sum_{i in S} w_i) + gamma * C(S)

where R is a weighted probabilistic coverage of query facets and C is a facility-location
coverage of the candidate pool (C uses fixed prior weights, so learning only touches R and w).
``budgeted_select`` runs cost-density greedy variants, adds the best singleton, and returns the
highest-value feasible set, which keeps the (1 - 1/e) / 2 worst-case guarantee of Theorem 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SelectionProblem:
    """All arrays are indexed by candidate; ``n`` candidates and ``f`` query facets."""

    weights: np.ndarray  # shape (n,), values in [0, 1]: P(item is relevant)
    costs: np.ndarray  # shape (n,), token costs > 0
    affinity: np.ndarray  # shape (n, f), values in [0, 1]
    facet_weights: np.ndarray  # shape (f,), non-negative, sums to 1
    similarity: np.ndarray | None = None  # shape (n, n), values in [0, 1]
    prior_weights: np.ndarray | None = None  # shape (n,): fixed importance for C
    gamma: float = 0.0
    modular: float = 0.0  # weight of the additive term eta * sum(w_i)


def _objective_state(p: SelectionProblem) -> tuple[np.ndarray, np.ndarray | None]:
    miss = np.ones(p.affinity.shape[1])
    best = None if p.similarity is None else np.zeros(p.weights.shape[0])
    return miss, best


def value(p: SelectionProblem, chosen: list[int]) -> float:
    miss, best = _objective_state(p)
    for i in chosen:
        miss = miss * (1.0 - p.weights[i] * p.affinity[i])
    r = float(p.facet_weights @ (1.0 - miss)) + p.modular * float(sum(p.weights[i] for i in chosen))
    if p.similarity is None or p.gamma == 0.0 or not chosen:
        return (1.0 - p.gamma) * r
    best = p.similarity[chosen].max(axis=0)
    total = float(p.prior_weights.sum()) or 1.0
    return (1.0 - p.gamma) * r + p.gamma * float(p.prior_weights @ best) / total


def marginal_gains(p: SelectionProblem, miss: np.ndarray, best: np.ndarray | None) -> np.ndarray:
    """Vectorised F(S + i) - F(S) for every candidate i."""
    factor = 1.0 - p.weights[:, None] * p.affinity
    gains = (1.0 - p.gamma) * ((miss * (1.0 - factor)) @ p.facet_weights + p.modular * p.weights)
    if best is not None and p.gamma > 0.0:
        total = float(p.prior_weights.sum()) or 1.0
        lift = np.clip(p.similarity - best, 0.0, None)
        gains = gains + p.gamma * (lift @ p.prior_weights) / total
    return gains


def _greedy(p: SelectionProblem, budget: float, cost_exponent: float) -> list[int]:
    alive = p.costs <= budget
    miss, best = _objective_state(p)
    chosen: list[int] = []
    spent = 0.0
    scale = np.power(p.costs, cost_exponent)
    while alive.any():
        gains = marginal_gains(p, miss, best)
        density = np.where(alive, gains / scale, -np.inf)
        i = int(np.argmax(density))
        if gains[i] <= 1e-12:
            break
        alive[i] = False
        if spent + p.costs[i] > budget:
            continue
        chosen.append(i)
        spent += float(p.costs[i])
        miss = miss * (1.0 - p.weights[i] * p.affinity[i])
        if best is not None:
            best = np.maximum(best, p.similarity[i])
    return chosen


def budgeted_select(
    p: SelectionProblem, budget: float, cost_exponents: tuple[float, ...] = (1.0,)
) -> list[int]:
    """Return the best feasible set among density greedy variants and the best singleton.

    ``cost_exponents`` must contain 1.0 for Theorem 1 to apply. Other exponents can only raise the
    returned value, since the result is the maximiser of F over all runs.
    """
    n = p.weights.shape[0]
    if n == 0 or budget <= 0:
        return []
    runs = [_greedy(p, budget, rho) for rho in cost_exponents]
    feasible = np.flatnonzero(p.costs <= budget)
    if feasible.size:
        singles = marginal_gains(p, *_objective_state(p))
        runs.append([int(feasible[np.argmax(singles[feasible])])])
    return max(runs, key=lambda chosen: value(p, chosen))


def optimality_certificate(p: SelectionProblem, chosen: list[int], budget: float) -> float:
    """Upper bound on OPT: F(S) plus the fractional-knapsack value of marginal gains at S."""
    miss, best = _objective_state(p)
    for i in chosen:
        miss = miss * (1.0 - p.weights[i] * p.affinity[i])
        if best is not None:
            best = np.maximum(best, p.similarity[i])
    gains = marginal_gains(p, miss, best)
    gains[chosen] = 0.0
    order = np.argsort(-gains / p.costs)
    remaining = float(budget)
    bound = value(p, chosen)
    for i in order:
        if gains[i] <= 0 or remaining <= 0:
            break
        take = min(1.0, remaining / float(p.costs[i]))
        bound += take * float(gains[i])
        remaining -= take * float(p.costs[i])
    return bound
