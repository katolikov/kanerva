"""The second held-out test: eight RepoMem repositories, for the rule that only facts can replace each other.

``kanerva`` is the shipped method and ``kanerva-events`` the same method with the earlier rule, under which similar
events also replaced each other. Writes results/fresh_repomem-new*.json, new_tests.json and new_tracks.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from experiments.final_test import BASELINES, by_track, compare, run, score_pairs
from experiments.harness import RESULTS, git_commit
from kanerva import RetrievalConfig

KEYED = {"dataset": "repomem-new", "cache": "repomem-new", "split": "new", "keyed": True}
PLAIN = {**KEYED, "keyed": False}
STATIC = {"dataset": "repomem-new", "cache": "repomem-new-m2v", "split": "new", "keyed": False}
METHODS = BASELINES | {"kanerva", "kanerva-events", "kanerva-ce", "kanerva-lex", "kanerva-lex-events", "kanerva-r1", "-supersession"}
PAIRS = (
    ("kanerva", "kanerva-events"), ("kanerva", "hybrid"), ("kanerva", "rerank"), ("kanerva", "hybrid-live"), ("kanerva", "kanerva-r1"),
    ("kanerva-ce", "rerank"), ("kanerva-lex", "kanerva-lex-events"), ("kanerva-lex", "bm25"),
)


def stale(records: list[dict]) -> dict:
    changed = [r for r in records if r["budget"] == 1024 and r["category"].endswith("-changed")]
    return {
        m: {k: float(np.mean([r[k] for r in changed if r["method"] == m])) for k in ("stale_any", "stale_error", "recall")}
        for m in sorted({r["method"] for r in changed})
    }


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    score_pairs("repomem-new", KEYED)
    outputs = {"repomem-new": run("repomem-new", KEYED, METHODS, RetrievalConfig()),
               "repomem-new-plain": run("repomem-new-plain", PLAIN, METHODS, RetrievalConfig())}
    os.environ["KANERVA_CE_CACHE"] = "repomem-new-m2v"
    outputs["repomem-new-static"] = run("repomem-new-static", STATIC, {"bm25", "dense", "hybrid", "kanerva"}, RetrievalConfig(model="static"))
    rng = np.random.default_rng(0)
    report: dict = {"meta": {"experiment": "second held-out test", "commit": git_commit()}}
    for name, records in outputs.items():
        have = {r["method"] for r in records}
        report[name] = {f"{a} vs {b}": compare(records, name, a, b, rng) for a, b in PAIRS if a in have and b in have}
        report[name]["stale_on_changed_facts"] = stale(records)
    (RESULTS / "new_tests.json").write_text(json.dumps(report, indent=1))
    tracks = {"meta": {"experiment": "RepoMem tracks", "commit": git_commit()},
              "results": {name: by_track(name) for name in ("repomem-new", "repomem-new-plain")}}
    (RESULTS / "new_tracks.json").write_text(json.dumps(tracks, indent=1))
    print("commit", git_commit())


if __name__ == "__main__":
    sys.exit(main())
