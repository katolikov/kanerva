"""Horizon at which the upper bound of Theorem 2 drops below the trivial bound T, for two feature dimensions."""

from __future__ import annotations

import json
import math

from experiments.harness import RESULTS, git_commit


def bound(T: float, d: int, K: int = 40, R: float = 0.5, sigma: float = 1.0, delta: float = 0.05) -> float:
    lam = float(K)
    beta = R * math.sqrt(d * math.log(4 * (1 + K * T / lam) / delta)) + math.sqrt(lam) * sigma
    return 2 * beta * math.sqrt(2 * d * K * T * math.log(1 + K * T / (d * lam))) + K * math.sqrt(2 * T * math.log(4 / delta))


def horizon(d: int) -> float:
    T = 1.0
    while bound(T, d) >= T:
        T *= 1.05
    return T


if __name__ == "__main__":
    out = {"meta": {"commit": git_commit(), "K": 40, "R": 0.5, "Sigma": 1.0, "delta": 0.05, "L": 1.0},
           "horizon": {str(d): horizon(d) for d in (3, 12, 15)}}
    (RESULTS / "regret_scale.json").write_text(json.dumps(out, indent=1))
    print(out["horizon"])
