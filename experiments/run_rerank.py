"""E2: hybrid retrieval followed by a cross-encoder reranker, the strongest non-LLM reading baseline."""

from __future__ import annotations

import sys
import time

import numpy as np

from experiments.embed_cache import TextCache
from experiments.harness import BUDGETS, apply_write, new_memory, save, score
from experiments.methods import Context, fill
from experiments.registry import load
from kanerva import RetrievalConfig

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
POOL = 50


def run(name: str, dataset: str, keyed: bool) -> None:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(MODEL)
    cache = TextCache(dataset)
    records = []
    start = time.time()
    streams = [s for s in load(dataset) if s.split == "test"]
    for stream in streams:
        mem = new_memory(cache, RetrievalConfig())
        id_of: dict[str, int] = {}
        cursor = 0
        for query in stream.queries:
            while cursor < query.pos:
                apply_write(mem, cache, stream.writes[cursor], id_of, keyed)
                cursor += 1
            gold = {id_of[t] for t in query.gold if t in id_of}
            stale = {id_of[t] for t in query.stale if t in id_of} - gold
            if not gold:
                continue
            ctx = Context(mem, query.question, query.now, cache.vector(query.question))
            pool = ctx.hybrid()[:POOL]
            rows = mem._fetch(pool)
            t0 = time.perf_counter()
            scores = model.predict([(query.question, rows[i]["text"][:1500]) for i in pool], batch_size=64, show_progress_bar=False)
            ms = (time.perf_counter() - t0) * 1000
            order = [pool[i] for i in np.argsort(-np.asarray(scores))]
            for budget in BUDGETS:
                ids = fill(order, ctx, budget)
                row = {"method": "rerank", "budget": budget, "qid": query.qid, "category": query.category, "stream": stream.name}
                row.update(score(ids, gold, stale, ctx.cost(ids) if ids else {}, ms))
                records.append(row)
        mem.close()
    save(f"rerank_{name}", records, {"experiment": "rerank", "model": MODEL, "pool": POOL, "dataset": name,
                                     "wall_seconds": round(time.time() - start, 1)})
    print(name, len(records), f"{time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    runs = {"lme": ("lme", False), "locomo": ("locomo", False), "repomem-keyed": ("repomem", True)}
    for name in sys.argv[1:] or runs:
        run(name, *runs[name])
