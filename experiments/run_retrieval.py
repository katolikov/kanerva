"""E1: budgeted retrieval on the test splits for all baselines, the method and its ablations."""

from __future__ import annotations

import sys
import time

from experiments.harness import BUDGETS, run_dataset, save, table
from experiments.registry import load
from kanerva import RetrievalConfig

RUNS = {
    "lme": {"keyed": False},
    "locomo": {"keyed": False},
    "repomem-keyed": {"dataset": "repomem", "keyed": True},
    "repomem-plain": {"dataset": "repomem", "keyed": False},
}


def run(name: str, split: str = "test") -> None:
    spec = RUNS[name]
    dataset = spec.get("dataset", name)
    streams = [s for s in load(dataset) if s.split == split]
    start = time.time()
    records = run_dataset(streams, dataset, RetrievalConfig(), keyed=spec["keyed"], fixed_k=(3, 5, 10, 20))
    meta = {"experiment": "retrieval", "dataset": name, "split": split, "streams": len(streams), "budgets": BUDGETS,
            "seeds": "deterministic retrieval; RepoMem has 3 seeded streams per repository",
            "wall_seconds": round(time.time() - start, 1)}
    save(f"retrieval_{name}", records, meta)
    print(f"{name}: {len(records)} records in {time.time() - start:.0f}s\n{table(records)}", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or RUNS:
        run(name)
