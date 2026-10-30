"""Coordinate search for the selection hyperparameters on the dev splits only."""

from __future__ import annotations

import json
import time

import numpy as np

from experiments.harness import BUDGETS, RESULTS, aggregate, run_dataset
from experiments.registry import load
from kanerva import RetrievalConfig

GRID = {
    "modular": (0.0, 0.5, 1.0, 2.0, 4.0),
    "gamma": (0.0, 0.05, 0.1, 0.2),
    "cost_exponents": ((1.0,), (1.0, 0.5), (1.0, 0.5, 0.0), (1.0, 0.75, 0.5, 0.25, 0.0)),
    "floor": (0.0, 0.002, 0.01, 0.05),
    "query_facet": (0.25, 0.5, 0.75, 0.9),
}


def objective(streams: dict, cfg: RetrievalConfig) -> tuple[float, dict]:
    parts = {}
    for name, dev in streams.items():
        agg = aggregate(run_dataset(dev, name, cfg, keyed=name == "repomem", selected={"kanerva"}))
        parts[name] = float(np.mean([agg[f"kanerva@{b}"]["recall"] for b in BUDGETS]))
    return float(np.mean(list(parts.values()))), parts


def main() -> None:
    streams = {name: [s for s in load(name) if s.split == "dev"] for name in ("lme", "locomo", "repomem")}
    cfg = RetrievalConfig()
    best, parts = objective(streams, cfg)
    print("start", round(best, 4), parts, flush=True)
    log = []
    for _ in range(1):
        for param, values in GRID.items():
            for v in values:
                if getattr(cfg, param) == v:
                    continue
                trial = cfg.with_(**{param: v})
                start = time.time()
                score, trial_parts = objective(streams, trial)
                print(param, v, round(score, 4), {k: round(x, 4) for k, x in trial_parts.items()}, f"{time.time() - start:.0f}s", flush=True)
                log.append({"param": param, "value": v, "score": score, "parts": trial_parts})
                if score > best + 1e-4:
                    best, cfg = score, trial
    chosen = {k: getattr(cfg, k) for k in GRID}
    print("best", best, chosen)
    (RESULTS / "tuning.json").write_text(json.dumps({"best": best, "chosen": chosen, "log": log}, indent=1))


if __name__ == "__main__":
    main()
