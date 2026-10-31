"""Near-twin detection with the static embedder: its own vectors against hashing vectors, as lexical mode uses.

A setting would ship only if, on each of the four groups of RepoMem repositories, it lowers stale injection at 1024
tokens and keeps mean recall. Every repository has been seen by now, so these are development numbers, not a test.
Writes results/dev_static_twins.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from experiments.final_test import per_query, streams_for
from experiments.harness import BUDGETS, RESULTS, git_commit, run_dataset
from experiments.new_repos_test import stale
from experiments.registry import load
from experiments.stats import paired
from kanerva import RetrievalConfig

CURRENT = RetrievalConfig(model="static")
GRID = ((0.7, 0.9), (0.7, 0.85), (0.65, 0.85), (0.6, 0.9), (0.6, 0.85))  # versions of a fact sit near 0.8
GROUPS = {
    "repomem": {"dataset": "repomem", "cache": "repomem-m2v", "split": None},
    "repomem-holdout": {"dataset": "repomem-holdout", "cache": "repomem-holdout-m2v", "split": "holdout"},
    "repomem-fresh": {"dataset": "repomem-fresh", "cache": "repomem-fresh-m2v", "split": "fresh"},
    "repomem-new": {"dataset": "repomem-new", "cache": "repomem-new-m2v", "split": "new"},
}
CONVERSATIONS = {"dataset": "beam", "cache": "beam-m2v", "split": "test"}


def records(spec: dict, cfg: RetrievalConfig, method: str) -> list[dict]:
    streams = load(spec["dataset"]) if spec["split"] is None else streams_for(spec)
    rows = run_dataset(streams, spec["cache"], cfg, keyed=False, selected={"kanerva"})
    return [r | {"method": method} for r in rows]


def row(rows: list[dict], method: str) -> dict:
    changed = stale(rows).get(method, {})
    budgets = {str(b): float(np.mean([r["recall"] for r in rows if r["budget"] == b])) for b in BUDGETS}
    return {"recall": float(np.mean([r["recall"] for r in rows])), "budgets": budgets, "stale_any": changed.get("stale_any"),
            "stale_error": changed.get("stale_error"), "changed_recall": changed.get("recall")}


def test(before: list[dict], after: list[dict], rng) -> dict:
    return paired(per_query(after, "hashing", "recall", BUDGETS), per_query(before, "vectors", "recall", BUDGETS), rng, "repomem")


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    rng = np.random.default_rng(0)
    current = {name: records(spec, CURRENT, "vectors") for name, spec in GROUPS.items()}
    table: dict = {"vectors": {name: row(rows, "vectors") for name, rows in current.items()}}
    print("vectors", json.dumps(table["vectors"])[:400], flush=True)
    for tau, drop in GRID:
        cfg = CURRENT.with_(hash_twins=True, supersession_threshold=tau, supersession_drop=drop)
        name = f"hashing-{tau}-{drop}"
        table[name] = {}
        for group, spec in GROUPS.items():
            rows = records(spec, cfg, "hashing")
            table[name][group] = row(rows, "hashing") | {"test": test(current[group], rows, rng)}
        print(name, {g: (round(v["recall"], 4), round(v["stale_any"], 3)) for g, v in table[name].items()}, flush=True)
    base = table["vectors"]
    passing = [n for n in table if n != "vectors" and all(
        table[n][g]["stale_any"] < base[g]["stale_any"] and table[n][g]["recall"] >= base[g]["recall"] for g in GROUPS)]
    chosen = min(passing, key=lambda n: sum(table[n][g]["stale_any"] for g in GROUPS)) if passing else None
    print("passing", passing, "chosen", chosen, flush=True)
    tau, drop = GRID[0]
    cfg = CURRENT.with_(hash_twins=True, supersession_threshold=tau, supersession_drop=drop)
    conversations = {"vectors": row(records(CONVERSATIONS, CURRENT, "vectors"), "vectors"), "hashing": row(records(CONVERSATIONS, cfg, "hashing"), "hashing")}
    print("beam", conversations["vectors"]["recall"], conversations["hashing"]["recall"], flush=True)
    meta = {"experiment": "near-twin vectors, static embedder", "commit": git_commit(), "rule": "lower stale injection and no loss of recall on all four groups"}
    report = {"meta": meta, "chosen": chosen, "passing": passing, "table": table, "beam": conversations}
    (RESULTS / "dev_static_twins.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
