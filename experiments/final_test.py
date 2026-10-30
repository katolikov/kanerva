"""The held-out test: BEAM-100K and BEAM-500K, and five RepoMem repositories that fitting never saw.

Steps: score cross-encoder pairs in one process, run every method, then write paired tests to results/fresh_tests.json.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time

import numpy as np

from experiments.cross_encoder import LiveCrossEncoder
from experiments.embed_cache import TextCache
from experiments.harness import BUDGETS, RESULTS, apply_write, git_commit, new_memory, run_dataset, save, table
from experiments.methods import Context
from experiments.registry import load
from experiments.stats import holm, paired
from kanerva import RetrievalConfig

BASELINES = {"recency", "bm25", "dense", "hybrid", "hybrid-live", "hybrid-newest", "hybrid-dedup", "mmr", "gen-agents", "facility", "rerank"}
OURS = {"kanerva", "kanerva-ce", "kanerva-lex", "kanerva-x128", "kanerva-r1", "-neighbours", "-density", "-validity", "-supersession"}
RUNS = {
    "beam": {"dataset": "beam", "cache": "beam", "split": "test", "keyed": False},
    "repomem-fresh": {"dataset": "repomem-fresh", "cache": "repomem-fresh", "split": "fresh", "keyed": True},
    "repomem-fresh-plain": {"dataset": "repomem-fresh", "cache": "repomem-fresh", "split": "fresh", "keyed": False},
}
STATIC_RUNS = {
    "beam-static": {"dataset": "beam", "cache": "beam-m2v", "split": "test", "keyed": False},
    "repomem-fresh-static": {"dataset": "repomem-fresh", "cache": "repomem-fresh-m2v", "split": "fresh", "keyed": False},
}


def streams_for(spec: dict) -> list:
    return [s for s in load(spec["dataset"]) if s.split == spec["split"]]


def score_pairs(name: str, spec: dict) -> None:
    """Cross-encoder logits for the pools of both the rerank baseline and the kanerva-ce variant."""
    cache = TextCache(spec["cache"])
    live = LiveCrossEncoder(spec["cache"])
    cfg = RetrievalConfig(use_bandit=False, use_reranker=True, model="rerank")
    start = time.time()
    for index, stream in enumerate(streams_for(spec)):
        mem = new_memory(cache, cfg, live)
        id_of: dict[str, int] = {}
        cursor = 0
        for query in stream.queries:
            while cursor < query.pos:
                apply_write(mem, cache, stream.writes[cursor], id_of, spec["keyed"])
                cursor += 1
            if not {id_of[t] for t in query.gold if t in id_of}:
                continue
            ctx = Context(mem, query.question, query.now, cache.vector(query.question))
            mem.prepare(query.question, query.now, None, cfg, ctx.qvec)
            top = ctx.hybrid()[:50]
            rows = mem._fetch(top)
            live(query.question, [rows[i]["text"] for i in top])
        mem.close()
        live.save()
        print(f"{name}: scored stream {index + 1} in {time.time() - start:.0f}s", flush=True)


def run(name: str, spec: dict, selected: set[str], base: RetrievalConfig) -> list[dict]:
    start = time.time()
    streams = streams_for(spec)
    records = run_dataset(streams, spec["cache"], base, keyed=spec["keyed"], selected=selected)
    meta = {"experiment": "fresh test", "dataset": name, "split": spec["split"], "streams": len(streams),
            "queries": len({r["qid"] for r in records}), "budgets": BUDGETS, "wall_seconds": round(time.time() - start, 1)}
    save(f"fresh_{name}", records, meta)
    print(f"== {name} ({meta['queries']} queries, {meta['wall_seconds']}s)\n{table(records)}\n{table(records, 'answer')}", flush=True)
    return records


def per_query(records: list[dict], method: str, metric: str, budgets) -> dict:
    """Mean of ``metric`` over ``budgets`` for each query of ``method``."""
    out: dict = {}
    for r in records:
        if r["method"] == method and r["budget"] in budgets and np.isfinite(r[metric]):
            out.setdefault((r["stream"], r["qid"]), []).append(r[metric])
    return {k: float(np.mean(v)) for k, v in out.items()}


def compare(records: list[dict], name: str, a: str, b: str, rng, metric: str = "recall") -> dict:
    """Per-budget tests with Holm correction within this comparison, plus the mean over all budgets."""
    unit = "repomem" if name.startswith("repomem") else "beam"  # a conversation or a repository is the unit
    rows = {}
    for budget in BUDGETS:
        rows[str(budget)] = paired(per_query(records, a, metric, (budget,)), per_query(records, b, metric, (budget,)), rng, unit)
    holm(list(rows.values()))
    rows["mean"] = paired(per_query(records, a, metric, BUDGETS), per_query(records, b, metric, BUDGETS), rng, unit)
    return rows


def tests(all_records: dict[str, list[dict]]) -> None:
    rng = np.random.default_rng(0)
    report: dict = {"meta": {"experiment": "paired tests on the held-out data", "commit": git_commit()}}
    for name, records in all_records.items():
        have = {r["method"] for r in records}
        pairs = [("kanerva", b) for b in ("hybrid", "rerank", "kanerva-r1", "bm25", "hybrid-live") if b in have]
        pairs += [("kanerva-ce", b) for b in ("rerank", "kanerva") if "kanerva-ce" in have and b in have]
        pairs += [("kanerva-lex", "bm25")] if "kanerva-lex" in have else []
        pairs += [("kanerva", b) for b in ("-neighbours", "-density", "-validity", "-supersession") if b in have]
        report[name] = {f"{a} vs {b}": compare(records, name, a, b, rng) for a, b in pairs}
        if "kanerva-x128" in have:
            report[name]["kanerva-x128 vs kanerva (answer)"] = compare(records, name, "kanerva-x128", "kanerva", rng, "answer")
        changed = [r for r in records if r["budget"] == 1024 and r["category"].endswith("-changed")]
        report[name]["stale_on_changed_facts"] = {
            m: {"stale_any": float(np.mean([r["stale_any"] for r in changed if r["method"] == m])),
                "stale_error": float(np.mean([r["stale_error"] for r in changed if r["method"] == m])),
                "recall": float(np.mean([r["recall"] for r in changed if r["method"] == m]))}
            for m in sorted({r["method"] for r in changed})
        }
    (RESULTS / "fresh_tests.json").write_text(json.dumps(report, indent=1))


def by_track(name: str) -> dict:
    """Recall on RepoMem's fact and task tracks at each budget, from the stored per-query rows."""
    rows = json.load(gzip.open(RESULTS / "raw" / f"fresh_{name}.json.gz", "rt"))
    out: dict = {}
    for track, member in (("fact", lambda c: c != "task"), ("task", lambda c: c == "task")):
        for method in sorted({r["method"] for r in rows}):
            for budget in BUDGETS:
                values = [r["recall"] for r in rows if r["method"] == method and r["budget"] == budget and member(r["category"])]
                out[f"{track}/{method}@{budget}"] = float(np.mean(values))
    return out


def tracks() -> None:
    report = {"meta": {"experiment": "RepoMem tracks", "commit": git_commit()},
              "results": {name: by_track(name) for name in ("repomem-fresh", "repomem-fresh-plain")}}
    (RESULTS / "fresh_tracks.json").write_text(json.dumps(report, indent=1))


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    for name in ("beam", "repomem-fresh"):
        score_pairs(name, RUNS[name])
    outputs = {}
    for name, spec in RUNS.items():
        outputs[name] = run(name, spec, BASELINES | OURS, RetrievalConfig())
    for name, spec in STATIC_RUNS.items():
        os.environ["KANERVA_CE_CACHE"] = spec["cache"]
        outputs[name] = run(name, spec, {"bm25", "dense", "hybrid", "kanerva"}, RetrievalConfig(model="static"))
    tests(outputs)
    tracks()
    print("commit", git_commit())


if __name__ == "__main__":
    sys.exit(main())
