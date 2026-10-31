"""Two flaws of the RepoMem facts, and how much they move the third test's numbers.

Some changed facts only moved to another file and kept their value, so injecting the older note does no harm. And the
dependency extractor reads console-script entries such as ``= httpie.manager.__main__:main`` as version constraints. The
dataset stays as it was, so that all tests remain comparable. This rescores the stored rows of the third test without those
queries. Writes results/data_flaws.json.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections import Counter

import numpy as np

from experiments.answer_test import _VALUE, code_tokens
from experiments.harness import BUDGETS, RESULTS, git_commit
from experiments.registry import load
from experiments.stats import paired

SETS = ("repomem-final-plain", "repomem-final")
METHODS = ("kanerva", "kanerva-lex", "kanerva-0.4", "kanerva-lex-0.4", "hybrid", "hybrid-dedup", "bm25")
PAIRS = (("kanerva", "hybrid"), ("kanerva-lex", "bm25"))
STALE_BUDGET = 1024


def harmless(query) -> bool:
    """A changed fact whose older versions all hold the same value. A location's value is its file, so a move counts there."""
    pattern = _VALUE[query.category.split("-")[1]]
    found = {m.group(1) for m in map(pattern.search, query.stale) if m}
    return bool(found) and all(code_tokens(v) == code_tokens(query.answer) for v in found)


def bogus(query) -> bool:
    """A dependency fact whose value holds no version number."""
    return query.category.startswith("fact-dependency") and not re.search(r"\d", query.answer)


def flawed_queries() -> tuple[set, set, dict]:
    moved, wrong, counts = set(), set(), Counter()
    for stream in load("repomem-final"):
        for query in stream.queries:
            if not query.category.startswith("fact-"):
                continue
            kind = query.category.split("-")[1]
            counts[f"{kind}/facts"] += 1
            if query.category.endswith("-changed"):
                counts[f"{kind}/changed"] += 1
                if harmless(query):
                    moved.add((stream.name, query.qid))
                    counts[f"{kind}/moved"] += 1
            if bogus(query):
                wrong.add((stream.name, query.qid))
                counts[f"{kind}/bogus"] += 1
    return moved, wrong, dict(counts)


def stale_rates(rows: list[dict], skip: set) -> dict:
    out = {}
    for method in METHODS:
        now = [r for r in rows if r["method"] == method and int(r["budget"]) == STALE_BUDGET and r["category"].endswith("-changed")
               and (r["stream"], r["qid"]) not in skip]
        out[method] = round(float(np.mean([float(r["stale_any"]) for r in now])), 4) if now else None
    return out


def repository(row: dict) -> str:
    return row["stream"].rsplit("-", 1)[0]


def lead(rows: list[dict], a: str, b: str, skip: set, rng) -> dict:
    """The mean recall difference over budgets, by repository, with the flawed queries left out."""
    values: dict = {a: {}, b: {}}
    for r in rows:
        if r["method"] in values and (r["stream"], r["qid"]) not in skip:
            values[r["method"]].setdefault((r["stream"], r["qid"]), []).append(float(r["recall"]))
    left = {key: float(np.mean(v)) for key, v in values[a].items()}
    right = {key: float(np.mean(v)) for key, v in values[b].items()}
    test = paired(left, right, rng, "repomem")
    return {k: test[k] for k in ("diff", "cluster_ci", "units_positive", "p", "n")}


def main() -> None:
    moved, wrong, counts = flawed_queries()
    rng = np.random.default_rng(0)
    report: dict = {"meta": {"experiment": "flaws of the RepoMem facts", "commit": git_commit(), "budget": STALE_BUDGET, "budgets": BUDGETS}, "counts": counts,
                    "moved": len(moved), "bogus": len(wrong)}
    for name in SETS:
        rows = [r for r in json.load(gzip.open(RESULTS / "raw" / f"fresh_{name}.json.gz", "rt")) if r["method"] in {*METHODS}]
        report[name] = {
            "stale_all": stale_rates(rows, set()), "stale_value_changed": stale_rates(rows, moved),
            "leads": {f"{a} vs {b}": {"all": lead(rows, a, b, set(), rng), "clean": lead(rows, a, b, moved | wrong, rng)} for a, b in PAIRS},
        }
    (RESULTS / "data_flaws.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
