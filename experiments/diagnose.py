"""Where recall is lost, on the dev splits only.

Every gold item a method misses falls into exactly one stage of the pipeline: it never reached the
candidate pool, a staleness rule removed it, or it was a candidate that selection did not take.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

from experiments.embed_cache import TextCache
from experiments.harness import RESULTS, apply_write, new_memory
from experiments.methods import Context, fill
from experiments.registry import load
from experiments.run_rerank import CachedScores
from kanerva import RetrievalConfig

BUDGETS = (128, 256, 512, 1024, 4096)
RUNS = {"lme": ("lme", False), "locomo": ("locomo", False), "repomem": ("repomem", True), "repomem-plain": ("repomem", False)}


def diagnose(name: str) -> dict:
    dataset, keyed = RUNS[name]
    cache = TextCache(dataset)
    cfg = RetrievalConfig()
    scores = CachedScores(dataset)
    rows: list[dict] = []
    gold_costs: list[int] = []
    pool_sizes: list[int] = []
    for stream in load(dataset):
        if stream.split != "dev":
            continue
        mem = new_memory(cache, cfg)
        id_of: dict[str, int] = {}
        cursor = 0
        for query in stream.queries:
            while cursor < query.pos:
                apply_write(mem, cache, stream.writes[cursor], id_of, keyed)
                cursor += 1
            gold = {id_of[t] for t in query.gold if t in id_of}
            if not gold:
                continue
            ctx = Context(mem, query.question, query.now, cache.vector(query.question))
            prepared = mem.prepare(query.question, query.now, None, cfg, ctx.qvec)
            raw_pool = set(ctx.hybrid())
            cand = set(prepared.candidates.ids)
            pool_sizes.append(len(raw_pool))
            hybrid_rows = mem._fetch(ctx.hybrid())
            gold_costs.extend(int(hybrid_rows[i]["tokens"]) + 14 for i in gold & raw_pool)
            pool = ctx.hybrid()[:50]
            logits = np.asarray(scores(query.question, [hybrid_rows[i]["text"] for i in pool]), dtype=float)
            order = [pool[i] for i in np.argsort(-np.nan_to_num(logits, nan=-99.0))] + ctx.hybrid()[50:]
            for budget in BUDGETS:
                chosen = {item.id for item in mem.select(prepared, budget, record=False).items}
                base = set(fill(ctx.hybrid(), ctx, budget))
                rerank = set(fill(order, ctx, budget))
                row = {"budget": budget, "category": query.category, "n": len(gold)}
                row["hit"] = len(gold & chosen) / len(gold)
                row["hybrid"] = len(gold & base) / len(gold)
                row["rerank"] = len(gold & rerank) / len(gold)
                row["pool_miss"] = len(gold - raw_pool) / len(gold)
                row["removed"] = len((gold & raw_pool) - cand) / len(gold)
                row["unselected"] = len((gold & cand) - chosen) / len(gold)
                rows.append(row)
        mem.close()
    out: dict = {"queries": len(rows) // len(BUDGETS), "pool_size_mean": float(np.mean(pool_sizes)),
                 "gold_cost_pctl": [float(np.percentile(gold_costs, p)) for p in (10, 50, 90)], "by_budget": {}}
    keys = ("hit", "hybrid", "rerank", "pool_miss", "removed", "unselected")
    for budget in BUDGETS:
        subset = [r for r in rows if r["budget"] == budget]
        out["by_budget"][budget] = {k: round(float(np.mean([r[k] for r in subset])), 3) for k in keys}
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in subset:
            groups[r["category"]].append(r)
        out["by_budget"][budget]["categories"] = {
            c: {"n": len(g), **{k: round(float(np.mean([r[k] for r in g])), 3) for k in keys}} for c, g in sorted(groups.items())
        }
    return out


if __name__ == "__main__":
    report = {name: diagnose(name) for name in sys.argv[1:] or RUNS}
    (RESULTS / "diagnosis_dev.json").write_text(json.dumps(report, indent=1))
    for name, data in report.items():
        print(f"== {name}: {data['queries']} queries, pool {data['pool_size_mean']:.0f}, gold cost p10/50/90 {data['gold_cost_pctl']}")
        for budget, row in data["by_budget"].items():
            print(budget, {k: v for k, v in row.items() if k != "categories"})
        for c, row in data["by_budget"][1024]["categories"].items():
            print("  @1024", c, row)
