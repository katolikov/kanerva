"""The third held-out test: fifteen RepoMem repositories in four languages that nothing was fitted or tuned on.

``kanerva`` is the method after all of the changes since 0.4.1: refitted on the 34 earlier repositories, with the query
rule for near-twins and a hashed removal threshold. ``kanerva-0.4`` is the released method, and the ``any-twin`` variants
isolate the rule. Writes results/fresh_repomem-final*.json, final_tests.json, final_tracks.json and final_languages.json.
"""

from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np

from experiments.final_test import BASELINES, by_track, compare, run, score_pairs
from experiments.harness import BUDGETS, RESULTS, git_commit
from experiments.repomem import C_FINAL_LANGUAGES, C_LANGUAGES, C_LAST_LANGUAGES, FINAL_LANGUAGES
from kanerva import RetrievalConfig

KEYED = {"dataset": "repomem-final", "cache": "repomem-final", "split": "final", "keyed": True}
PLAIN = {**KEYED, "keyed": False}
METHODS = BASELINES | {"kanerva", "kanerva-ce", "kanerva-lex", "kanerva-any-twin", "kanerva-lex-any-twin", "kanerva-0.4", "kanerva-lex-0.4"}
BEAM = {"dataset": "beam", "cache": "beam", "split": "test", "keyed": False}
PAIRS = (
    ("kanerva", "hybrid"), ("kanerva", "rerank"), ("kanerva", "hybrid-live"), ("kanerva", "kanerva-0.4"), ("kanerva", "kanerva-any-twin"),
    ("kanerva-ce", "rerank"), ("kanerva-lex", "bm25"), ("kanerva-lex", "kanerva-lex-0.4"), ("kanerva-lex", "kanerva-lex-any-twin"),
)


def stale(records: list[dict]) -> dict:
    changed = [r for r in records if r["budget"] == 1024 and r["category"].endswith("-changed")]
    return {
        m: {k: float(np.mean([r[k] for r in changed if r["method"] == m])) for k in ("stale_any", "stale_error", "recall")}
        for m in sorted({r["method"] for r in changed})
    }


def repository(stream: str) -> str:
    return stream.rsplit("-", 1)[0]


def by_language(records: list[dict], languages: dict = FINAL_LANGUAGES) -> dict:
    out: dict = {}
    methods = sorted({r["method"] for r in records})
    for language, repositories in languages.items():
        rows = [r for r in records if repository(r["stream"]) in repositories]
        entry: dict = {"queries": len({(r["stream"], r["qid"]) for r in rows})}
        for track, member in (("all", lambda c: True), ("fact", lambda c: c != "task"), ("task", lambda c: c == "task")):
            part = [r for r in rows if member(r["category"])]
            if not part:
                continue
            for method in methods:
                values = [float(np.mean([r["recall"] for r in part if r["method"] == method and r["budget"] == b])) for b in BUDGETS]
                entry[f"{track}/{method}"] = values + [float(np.mean(values))]
        out[language] = entry
    return out


def beam_check() -> None:
    """BEAM-100K and 500K were never fitted on, but earlier tests have seen them, so this is a check and not a test."""
    records = run("beam-check", BEAM, {"kanerva", "kanerva-lex", "kanerva-0.4", "kanerva-lex-0.4", "hybrid", "bm25"}, RetrievalConfig())
    rng = np.random.default_rng(0)
    pairs = (("kanerva", "hybrid"), ("kanerva", "kanerva-0.4"), ("kanerva-lex", "bm25"), ("kanerva-lex", "kanerva-lex-0.4"))
    report = {"meta": {"experiment": "BEAM check of the refitted method", "commit": git_commit()},
              "beam": {f"{a} vs {b}": compare(records, "beam", a, b, rng) for a, b in pairs}}
    (RESULTS / "beam_check.json").write_text(json.dumps(report, indent=1))


def fact_answers(records: list[dict]) -> dict:
    """Whether the facts' values reached the context at 1024 tokens. Recall asks for one note, and a newer note can hold the same value."""
    rows = [r for r in records if r["budget"] == 1024 and r["category"] != "task" and np.isfinite(r["answer"])]
    return {m: float(np.mean([r["answer"] for r in rows if r["method"] == m])) for m in sorted({r["method"] for r in rows})}


def fact_losses(split: str = "c-last") -> None:
    """The facts a method drops where hybrid search keeps them at 1024 tokens, and whether their value reached the context anyway."""
    rows = json.load(gzip.open(RESULTS / "raw" / f"fresh_repomem-{split}-plain.json.gz", "rt"))

    def facts(method: str) -> dict:
        return {(r["stream"], r["qid"]): r for r in rows if r["method"] == method and r["budget"] == 1024 and r["category"] != "task"}

    ours, theirs = facts("kanerva"), facts("hybrid")
    lost = [key for key in ours if ours[key]["recall"] < theirs[key]["recall"]]
    values = [ours[key]["answer"] for key in lost if np.isfinite(ours[key]["answer"])]
    report = {"meta": {"experiment": "facts lost against hybrid", "commit": git_commit(), "split": split}, "facts": len(ours), "lost": len(lost),
              "lost_locations": sum(ours[key]["category"].startswith("fact-location") for key in lost), "value_in_context": float(np.mean(values))}
    (RESULTS / "fact_losses.json").write_text(json.dumps(report, indent=1))


def twin_rule() -> None:
    """The near-twin rule of 0.5.1 against that of 0.5.0, on data that has been read, so a check and not a test."""
    sets = {"third test": {"dataset": "repomem-final", "cache": "repomem-final", "split": "final", "keyed": False}, "c-test": {"dataset": "repomem-c-test", "cache": "repomem-c-test", "split": "c-test", "keyed": False},
            "beam": BEAM}
    methods = {"kanerva", "kanerva-0.5", "kanerva-lex", "kanerva-lex-0.5"}
    rng = np.random.default_rng(0)
    report: dict = {"meta": {"experiment": "near-twin rule of 0.5.1", "commit": git_commit()}}
    for label, spec in sets.items():
        records = run(f"twins-{label.replace(' ', '-')}", spec, methods, RetrievalConfig())
        name = spec["dataset"] if spec["dataset"].startswith("repomem") else "beam"
        report[label] = {f"{a} vs {b}": compare(records, name, a, b, rng)["mean"] for a, b in (("kanerva", "kanerva-0.5"), ("kanerva-lex", "kanerva-lex-0.5"))}
        report[label]["stale"] = {m: float(v["stale_any"]) for m, v in stale(records).items()}
        report[label]["answers"] = fact_answers(records)
    (RESULTS / "twin_rule.json").write_text(json.dumps(report, indent=1))


def c_family(split: str) -> None:
    """C and C++, with facts read by the regular expressions of experiments/repomem.py. ``c-dev`` is for fixing what these
    languages break. ``c-test`` ran once on version 0.5.0, ``c-final`` once on the near-twin rule that it led to, and ``c-last``
    once on version 0.5.2, with templates, overloads and class methods among its facts."""
    name = f"repomem-{split}"
    keyed = {"dataset": name, "cache": name, "split": split, "keyed": True}
    plain = {**keyed, "keyed": False}
    methods = {"bm25", "hybrid", "hybrid-dedup", "hybrid-live", "rerank", "kanerva", "kanerva-ce", "kanerva-lex", "kanerva-0.4", "kanerva-lex-0.4"}
    pairs = PAIRS
    if split in ("c-final", "c-last"):
        methods |= {"kanerva-0.5", "kanerva-lex-0.5"}
        pairs = PAIRS + (("kanerva", "kanerva-0.5"), ("kanerva-lex", "kanerva-lex-0.5"))
    score_pairs(name, keyed)
    outputs = {name: run(name, keyed, methods, RetrievalConfig()), f"{name}-plain": run(f"{name}-plain", plain, methods, RetrievalConfig())}
    rng = np.random.default_rng(0)
    report: dict = {"meta": {"experiment": f"C and C++, {split}", "commit": git_commit()}}
    for label, records in outputs.items():
        report[label] = {f"{a} vs {b}": compare(records, label, a, b, rng) for a, b in pairs if {a, b} <= methods}
        report[label]["stale_on_changed_facts"] = stale(records)
        report[label]["tracks"] = by_track(label)
        report[label]["answers"] = fact_answers(records)
    if split != "c-dev":
        groups = {"c-test": C_LANGUAGES, "c-final": C_FINAL_LANGUAGES, "c-last": C_LAST_LANGUAGES}
        report["languages"] = by_language(outputs[f"{name}-plain"], groups[split])
    (RESULTS / f"{split}_tests.json").write_text(json.dumps(report, indent=1))


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if sys.argv[1:] == ["beam"]:
        return beam_check()
    if sys.argv[1:] == ["twins"]:
        return twin_rule()
    if sys.argv[1:] == ["losses"]:
        return fact_losses()
    if sys.argv[1:] in (["c-dev"], ["c-test"], ["c-final"], ["c-last"]):
        return c_family(sys.argv[1])
    if sys.argv[1:] != ["retest"]:
        score_pairs("repomem-final", KEYED)
    outputs = {"repomem-final": run("repomem-final", KEYED, METHODS, RetrievalConfig()),
               "repomem-final-plain": run("repomem-final-plain", PLAIN, METHODS, RetrievalConfig())}
    rng = np.random.default_rng(0)
    report: dict = {"meta": {"experiment": "third held-out test", "commit": git_commit()}}
    for name, records in outputs.items():
        have = {r["method"] for r in records}
        report[name] = {f"{a} vs {b}": compare(records, name, a, b, rng) for a, b in PAIRS if a in have and b in have}
        report[name]["stale_on_changed_facts"] = stale(records)
    (RESULTS / "final_tests.json").write_text(json.dumps(report, indent=1))
    tracks = {"meta": {"experiment": "third held-out test by track", "commit": git_commit()}, "results": {name: by_track(name) for name in outputs}}
    (RESULTS / "final_tracks.json").write_text(json.dumps(tracks, indent=1))
    languages = {"meta": {"experiment": "third held-out test by language", "commit": git_commit()}, "results": {name: by_language(r) for name, r in outputs.items()}}
    (RESULTS / "final_languages.json").write_text(json.dumps(languages, indent=1))
    print("commit", git_commit())


if __name__ == "__main__":
    sys.exit(main())
