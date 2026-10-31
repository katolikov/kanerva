"""Near-twin detection with the static embedder: its own vectors against hashing vectors, as lexical mode uses.

Thresholds are picked on the repositories of the fitting data, with a rule set before the run: among the settings
whose mean recall is not below the current method, the one that injects the fewest stale facts at 1024 tokens. Every
RepoMem repository has been seen by now, so all of these are development numbers, not a test.
Writes results/dev_static_twins.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from experiments.final_test import compare, streams_for
from experiments.harness import BUDGETS, RESULTS, git_commit, run_dataset
from experiments.new_repos_test import stale
from kanerva import RetrievalConfig

CURRENT = RetrievalConfig(model="static")
GRID = ((0.8, 0.97), (0.7, 0.9), (0.6, 0.9), (0.6, 0.8), (0.5, 0.8), (0.5, 0.7), (0.4, 0.7))  # versions of a fact sit near 0.8
TUNE = {"dataset": "repomem", "cache": "repomem-m2v", "split": None, "keyed": False}
DATASETS = {
    "repomem-holdout": {"dataset": "repomem-holdout", "cache": "repomem-holdout-m2v", "split": "holdout", "keyed": False},
    "repomem-fresh": {"dataset": "repomem-fresh", "cache": "repomem-fresh-m2v", "split": "fresh", "keyed": False},
    "repomem-new": {"dataset": "repomem-new", "cache": "repomem-new-m2v", "split": "new", "keyed": False},
    "beam": {"dataset": "beam", "cache": "beam-m2v", "split": "test", "keyed": False},
    "lme": {"dataset": "lme", "cache": "lme-m2v", "split": None, "keyed": False},
    "locomo": {"dataset": "locomo", "cache": "locomo-m2v", "split": None, "keyed": False},
}


def records(spec: dict, cfg: RetrievalConfig, method: str) -> list[dict]:
    from experiments.registry import load

    streams = load(spec["dataset"]) if spec["split"] is None else streams_for(spec)
    rows = run_dataset(streams, spec["cache"], cfg, keyed=spec["keyed"], selected={"kanerva"})
    return [r | {"method": method} for r in rows]


def mean_recall(rows: list[dict]) -> float:
    return float(np.mean([r["recall"] for r in rows]))


def per_budget(rows: list[dict]) -> dict:
    return {str(b): float(np.mean([r["recall"] for r in rows if r["budget"] == b])) for b in BUDGETS}


def row(rows: list[dict], method: str) -> dict:
    changed = stale(rows).get(method, {})
    return {"recall": mean_recall(rows), "budgets": per_budget(rows), "stale_any": changed.get("stale_any"),
            "stale_error": changed.get("stale_error"), "changed_recall": changed.get("recall")}


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    baseline = records(TUNE, CURRENT, "vectors")
    tuning = {"vectors": row(baseline, "vectors")}
    print("tuning vectors", tuning["vectors"], flush=True)
    candidates = {}
    for tau, drop in GRID:
        cfg = CURRENT.with_(hash_twins=True, supersession_threshold=tau, supersession_drop=drop)
        name = f"hashing-{tau}-{drop}"
        tuning[name] = row(records(TUNE, cfg, name), name)
        candidates[name] = cfg
        print("tuning", name, tuning[name], flush=True)
    floor = tuning["vectors"]["recall"]
    allowed = [n for n in candidates if tuning[n]["recall"] >= floor]
    if not allowed:
        print("no setting keeps recall on the tuning data", flush=True)
        report = {"meta": {"experiment": "near-twin vectors, static embedder", "commit": git_commit()}, "tuning": tuning, "chosen": None}
        (RESULTS / "dev_static_twins.json").write_text(json.dumps(report, indent=1))
        return
    chosen = min(allowed, key=lambda n: tuning[n]["stale_any"])
    print("chosen", chosen, flush=True)
    rng = np.random.default_rng(0)
    results: dict = {}
    for name, spec in DATASETS.items():
        before = records(spec, CURRENT, "vectors")
        after = records(spec, candidates[chosen], "hashing")
        results[name] = {"vectors": row(before, "vectors"), "hashing": row(after, "hashing")}
        if name.startswith("repomem") or name == "beam":
            results[name]["test"] = compare(before + after, name, "hashing", "vectors", rng)["mean"]
        print(name, json.dumps(results[name])[:600], flush=True)
    meta = {"experiment": "near-twin vectors, static embedder", "commit": git_commit(), "rule": "lowest stale injection with no loss of recall on the tuning data"}
    (RESULTS / "dev_static_twins.json").write_text(json.dumps({"meta": meta, "tuning": tuning, "chosen": chosen, "results": results}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
