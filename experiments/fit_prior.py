"""Fit the policy prior theta0 by weighted ridge regression on the dev splits only."""

from __future__ import annotations

import json

import numpy as np

from experiments.embed_cache import TextCache
from experiments.harness import RESULTS, apply_write, new_memory
from experiments.registry import load
from kanerva import RetrievalConfig
from kanerva.policy import FEATURES


def collect(dataset: str, cfg: RetrievalConfig, keyed: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = TextCache(dataset)
    xs, ys, ws = [], [], []
    for stream in load(dataset):
        if stream.split != "dev":
            continue
        mem = new_memory(cache, cfg)
        id_of: dict[str, int] = {}
        cursor = 0
        for query in stream.queries:
            while cursor < query.pos:
                apply_write(mem, cache, stream.writes[cursor], id_of, keyed)
                cursor += 1
            gold = {id_of[t] for t in query.gold if t in id_of}
            if not gold:
                continue
            prepared = mem.prepare(query.question, query.now, None, cfg, cache.vector(query.question))
            if prepared.features is None:
                continue
            xs.append(prepared.features)
            ys.append(np.array([float(i in gold) for i in prepared.candidates.ids]))
            ws.append(np.full(len(ys[-1]), 1.0 / len(ys[-1])))
        mem.close()
    return np.vstack(xs), np.concatenate(ys), np.concatenate(ws)


def fit(cfg: RetrievalConfig, ridge: float = 1e-3) -> np.ndarray:
    parts = [collect(name, cfg, name == "repomem") for name in ("lme", "locomo", "repomem")]
    xtx, xty = 0.0, 0.0
    for x, y, w in parts:
        w = w / w.sum()
        xtx = xtx + (x * w[:, None]).T @ x
        xty = xty + (x * w[:, None]).T @ y
    d = xtx.shape[0]
    return np.linalg.solve(xtx + ridge * np.eye(d) / 3.0, xty)


if __name__ == "__main__":
    theta = fit(RetrievalConfig(use_validity=False, use_bandit=False))
    lexical = fit(RetrievalConfig(use_validity=False, use_bandit=False, use_dense=False, gamma=0.0))
    print("DEFAULT_THETA =", tuple(np.round(theta, 4).tolist()))
    print("LEXICAL_THETA =", tuple(np.round(lexical, 4).tolist()))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fitted_prior.json").write_text(
        json.dumps({"features": list(FEATURES), "theta": theta.tolist(), "lexical": lexical.tolist()}, indent=1)
    )
