"""E4: checks of Theorems 1 and 2 and of Propositions 3 and 8 on synthetic instances with known truth."""

from __future__ import annotations

import json
import math
import time

import numpy as np

from experiments.harness import RESULTS, git_commit
from kanerva.bandit import LinearBandit
from kanerva.select import SelectionProblem, budgeted_select, optimality_certificate, value

SEEDS = tuple(range(8))
D = 8
ITEMS = 40
BUDGET = 30.0
HORIZON = 4000
CHECKPOINTS = (250, 500, 1000, 2000, 4000)


def instance(rng: np.random.Generator):
    theta = rng.normal(size=D)
    theta /= np.linalg.norm(theta)
    return 0.5 * theta


def round_items(rng, theta):
    x = rng.normal(size=(ITEMS, D))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    mu = np.clip(0.5 + x @ theta, 0.0, 1.0)
    return np.column_stack([x, np.ones(ITEMS)]) / math.sqrt(2.0), mu


def problem(weights, costs, affinity, facets) -> SelectionProblem:
    return SelectionProblem(weights, costs, affinity, facets, modular=0.5)


def run(seed: int, policy: str) -> list[float]:
    """Cumulative 0.5(1-1/e)-regret against the per-round greedy optimum on the true weights."""
    rng = np.random.default_rng(seed)
    theta = instance(rng)
    alpha = 1.0
    hazards = np.array([0.002, 0.02, 0.1])
    observed = np.zeros((3, 2))
    observed[:, 1] = 1.0
    bandit = LinearBandit(np.zeros(D + 1), ridge=20.0, beta=0.5 if policy != "greedy" else 0.0)
    regret = []
    total = 0.0
    for t in range(HORIZON):
        x, mu = round_items(rng, theta)
        classes = rng.integers(0, 3, size=ITEMS)
        ages = rng.exponential(30.0, size=ITEMS)
        rho = np.exp(-hazards[classes] * ages)
        valid = rng.random(ITEMS) < rho
        costs = rng.integers(2, 10, size=ITEMS).astype(float)
        affinity = (rng.random((ITEMS, 5)) < 0.3).astype(float)
        affinity[:, 0] = 1.0
        facets = np.full(5, 0.2)
        true = rho * mu
        best = value(problem(true, costs, affinity, facets), budgeted_select(problem(true, costs, affinity, facets), BUDGET))
        scores = bandit.score(x)
        if policy == "known":
            weights = rho * scores
        elif policy == "estimated":
            rates = (observed[:, 0] + 1.0) / (observed[:, 1] + 10.0)
            weights = np.exp(-rates[classes] * ages) * scores
        else:
            weights = scores
        chosen = budgeted_select(problem(weights, costs, affinity, facets), BUDGET)
        got = value(problem(true, costs, affinity, facets), chosen)
        total += alpha * best - got
        regret.append(total)
        if chosen:
            idx = np.array(chosen)
            keep = idx[valid[idx]] if policy != "blind" else idx
            reward = (rng.random(len(keep)) < mu[keep]).astype(float)
            if policy == "blind":
                reward = reward * valid[keep]
            bandit.update(x[keep], reward)
            for i in idx:
                observed[classes[i], 0] += 0 if valid[i] else 1
                observed[classes[i], 1] += ages[i]
    return [regret[c - 1] for c in CHECKPOINTS]


def selection_gap(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(200):
        n = 60
        weights = rng.uniform(size=n)
        costs = rng.integers(1, 15, size=n).astype(float)
        affinity = (rng.random((n, 8)) < 0.2).astype(float)
        facets = rng.dirichlet(np.ones(8))
        p = problem(weights, costs, affinity, facets)
        chosen = budgeted_select(p, 40.0, (1.0, 0.5, 0.0))
        ratios.append(value(p, chosen) / optimality_certificate(p, chosen, 40.0))
    return ratios


def main() -> None:
    start = time.time()
    curves: dict[str, list[list[float]]] = {}
    for policy in ("known", "estimated", "blind"):
        curves[policy] = [run(seed, policy) for seed in SEEDS]
        print(policy, np.mean(curves[policy], axis=0).round(1), flush=True)
    slopes = {}
    for policy, values in curves.items():
        y = np.log(np.maximum(np.mean(values, axis=0), 1e-9))
        slopes[policy] = float(np.polyfit(np.log(CHECKPOINTS), y, 1)[0])
    ratios = [r for s in SEEDS for r in selection_gap(s)]
    out = {
        "meta": {"experiment": "synthetic", "seeds": SEEDS, "d": D + 1, "items": ITEMS, "budget": BUDGET, "horizon": HORIZON,
                 "commit": git_commit(), "wall_seconds": round(time.time() - start, 1)},
        "checkpoints": CHECKPOINTS,
        "regret": curves,
        "loglog_slope": slopes,
        "certificate_ratio": {"mean": float(np.mean(ratios)), "p05": float(np.percentile(ratios, 5)), "min": float(np.min(ratios)), "n": len(ratios)},
    }
    (RESULTS / "synthetic.json").write_text(json.dumps(out, indent=1))
    print(slopes, out["certificate_ratio"])


if __name__ == "__main__":
    main()
