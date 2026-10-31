"""RepoMem's task track on TypeScript, Go and Rust repositories, as a generalisation check.

The models were fitted on Python repositories and conversations, and nothing is refitted here. Only the task track
applies, because the fact track reads Python source. Writes results/fresh_repomem-other.json and other_languages.json.
"""

from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np

from experiments.final_test import compare, run, score_pairs
from experiments.harness import BUDGETS, RESULTS, git_commit
from experiments.repomem import LANGUAGES
from kanerva import RetrievalConfig

NAME = "repomem-other"
SPEC = {"dataset": NAME, "cache": NAME, "split": "other", "keyed": False}
METHODS = {"kanerva", "kanerva-ce", "kanerva-lex", "rerank", "hybrid", "bm25"}
PAIRS = (("kanerva", "hybrid"), ("kanerva", "rerank"), ("kanerva-ce", "rerank"), ("kanerva-lex", "bm25"))


def repository(stream: str) -> str:
    return stream.rsplit("-", 1)[0]


def by_language(records: list[dict], language: str) -> dict:
    repositories = LANGUAGES[language]
    rows = [r for r in records if repository(r["stream"]) in repositories]
    out: dict = {"queries": len({(r["stream"], r["qid"]) for r in rows})}
    for method in sorted({r["method"] for r in rows}):
        for budget in BUDGETS:
            out[f"{method}@{budget}"] = float(np.mean([r["recall"] for r in rows if r["method"] == method and r["budget"] == budget]))
    return out


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if sys.argv[1:] != ["retest"]:
        score_pairs(NAME, SPEC)
        run(NAME, SPEC, METHODS, RetrievalConfig())
    records = json.load(gzip.open(RESULTS / "raw" / f"fresh_{NAME}.json.gz", "rt"))
    rng = np.random.default_rng(0)
    report: dict = {"meta": {"experiment": "task track beyond Python", "commit": git_commit(), "repositories": LANGUAGES}}
    report["tests"] = {f"{a} vs {b}": compare(records, NAME, a, b, rng) for a, b in PAIRS}
    report["languages"] = {language: by_language(records, language) for language in LANGUAGES}
    (RESULTS / "other_languages.json").write_text(json.dumps(report, indent=1))
    print("commit", git_commit())


if __name__ == "__main__":
    sys.exit(main())
