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
    "repomem-holdout": {"dataset": "repomem-holdout", "keyed": True, "split": "holdout"},
}


def run_small_embedder() -> None:
    """The method and its rivals with the static model2vec embedder that the plugin can use."""
    for name, dataset, keyed in (("lme", "lme", False), ("locomo", "locomo", False), ("repomem-plain", "repomem", False)):
        streams = [s for s in load(dataset) if s.split == "test"]
        records = run_dataset(streams, f"{dataset}-m2v", RetrievalConfig(), keyed=keyed, selected={"kanerva", "hybrid", "dense"})
        save(f"m2v_{name}", records, {"experiment": "small-embedder", "dataset": name, "model": "minishlab/potion-base-8M"})


def run(name: str) -> None:
    spec = RUNS[name]
    split = spec.get("split", "test")
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
    for name in sys.argv[1:] or [*RUNS, "m2v"]:
        run_small_embedder() if name == "m2v" else run(name)
