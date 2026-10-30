"""Replay memory streams and score every method at every budget."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.datasets import ROOT, Query, Stream
from experiments.embed_cache import TextCache
from experiments.methods import BASELINES, Context, kanerva_variant, variants
from kanerva import Memory, RetrievalConfig

RESULTS = ROOT / "results"
BUDGETS = (128, 256, 512, 1024, 2048, 4096)
METRICS = ("recall", "all", "any", "tokens", "precision", "stale_any", "stale_error", "latency_ms", "answer")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty", "--abbrev=12"], capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return "uncommitted"


def answer_stems(answer: str, question: str) -> list[str]:
    """Stems of the reference answer that the question does not already contain."""
    from kanerva.text import content_terms, light_stem

    asked = {light_stem(t) for t in content_terms(question)}
    return [s for s in dict.fromkeys(light_stem(t) for t in content_terms(answer)) if s not in asked][:62]


def answer_recall(texts: list[str], wanted: list[str]) -> float:
    """Share of the reference answer's stems (those not already in the question) present in the injected text."""
    if not wanted:
        return float("nan")
    from kanerva.relevance import stems

    have: set[str] = set()
    for text in texts:
        have |= stems(text)
    return sum(1 for s in wanted if s in have) / len(wanted)


def score(ids: list[int], gold: set[int], stale: set[int], costs: dict[int, int], ms: float) -> dict:
    chosen = set(ids)
    hits = chosen & gold
    tokens = sum(costs[i] for i in ids)
    stale_hit = chosen & stale
    ranks = {i: r for r, i in enumerate(ids)}
    first_stale = min((ranks[i] for i in stale_hit), default=None)
    first_gold = min((ranks[i] for i in hits), default=None)
    error = first_stale is not None and (first_gold is None or first_stale < first_gold)
    return {
        "recall": len(hits) / len(gold) if gold else 0.0,
        "all": float(bool(gold) and hits == gold),
        "any": float(bool(hits)),
        "tokens": tokens,
        "precision": sum(costs[i] for i in hits) / tokens if tokens else 0.0,
        "stale_any": float(bool(stale_hit)),
        "stale_error": float(error),
        "latency_ms": ms,
    }


def new_memory(cache: TextCache, cfg: RetrievalConfig, reranker=None) -> Memory:
    return Memory(":memory:", count_tokens=cache.count, config=cfg, reranker=reranker)


def apply_write(mem: Memory, cache: TextCache, turn, id_of: dict[str, int], keyed: bool) -> None:
    item = mem.write(
        turn.text, at=turn.at, kind=turn.kind, key=turn.key if keyed else None, embedding=cache.vector(turn.text),
        session=turn.session,
    )
    if item is not None:
        id_of[turn.text] = item


def run_stream(args: tuple) -> list[dict]:
    """Stateless evaluation: all methods see the same store at each query."""
    stream, cache_name, base, keyed, budgets, selected, fixed_k = args
    cache = TextCache(cache_name)
    reranker = None
    if selected and {"kanerva-ce", "rerank"} & set(selected):
        from experiments.cross_encoder import CachedScores

        reranker = CachedScores(os.environ.get("KANERVA_CE_CACHE", cache_name))
    mem = new_memory(cache, base, reranker)
    id_of: dict[str, int] = {}
    methods = dict(BASELINES)
    configs = variants(base)
    for name, cfg in configs.items():
        methods[name] = kanerva_variant(cfg.with_(use_bandit=False))
    if reranker is None:
        methods.pop("kanerva-ce")
        methods.pop("rerank")
    if selected:
        methods = {k: v for k, v in methods.items() if k in selected}
    fixed = {f"hybrid-k{k}": k for k in fixed_k}
    records: list[dict] = []
    cursor = 0
    for query in stream.queries:
        while cursor < query.pos:
            apply_write(mem, cache, stream.writes[cursor], id_of, keyed)
            cursor += 1
        records.extend(evaluate_query(mem, cache, query, id_of, methods, budgets, stream.name, fixed))
    mem.close()
    return records


def evaluate_query(mem, cache, query: Query, id_of, methods, budgets, stream_name, fixed=()) -> list[dict]:
    gold = {id_of[t] for t in query.gold if t in id_of}
    stale = {id_of[t] for t in query.stale if t in id_of} - gold
    if not gold:
        return []
    ctx = Context(mem, query.question, query.now, cache.vector(query.question))
    wanted = answer_stems(query.answer, query.question) if query.answer else []
    rows = []
    for name, method in methods.items():
        for budget in budgets:
            ctx.cache.pop("texts", None)
            start = time.perf_counter()
            ids = method(ctx, budget)
            ms = (time.perf_counter() - start) * 1000
            costs = ctx.cost(ids) if ids else {}
            row = {"method": name, "budget": budget, "qid": query.qid, "category": query.category, "stream": stream_name}
            row.update(score(ids, gold, stale, costs, ms))
            texts = ctx.cache.get("texts")
            if texts is None:
                fetched = mem._fetch(ids)
                texts = [fetched[i]["text"] for i in ids]
            else:
                row["tokens"] = sum(ctx.cache["text_costs"])  # excerpts cost less than whole items
            row["answer"] = answer_recall(texts, wanted)
            rows.append(row)
    for name, k in dict(fixed).items():
        ids = ctx.hybrid()[:k]
        row = {"method": name, "budget": k, "qid": query.qid, "category": query.category, "stream": stream_name}
        row.update(score(ids, gold, stale, ctx.cost(ids) if ids else {}, 0.0))
        rows.append(row)
    return rows


def run_dataset(streams: list[Stream], cache_name: str, cfg: RetrievalConfig, keyed: bool = False,
                budgets=BUDGETS, selected=None, workers: int | None = None, fixed_k=()) -> list[dict]:
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    jobs = [(s, cache_name, cfg, keyed, budgets, selected, fixed_k) for s in streams]
    records: list[dict] = []
    with ProcessPoolExecutor(workers) as pool:
        for chunk in pool.map(run_stream, jobs, chunksize=max(1, len(jobs) // (workers * 4))):
            records.extend(chunk)
    return records


def aggregate(records: list[dict]) -> dict:
    table: dict[tuple, list[dict]] = {}
    for r in records:
        table.setdefault((r["method"], r["budget"]), []).append(r)
    return {
        f"{m}@{b}": {k: float(np.nanmean([r.get(k, np.nan) for r in rows])) if any(np.isfinite(r.get(k, np.nan)) for r in rows)
                     else float("nan") for k in METRICS} | {"n": len(rows)}
        for (m, b), rows in sorted(table.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    }


def save(name: str, records: list[dict], meta: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "raw").mkdir(exist_ok=True)
    with gzip.open(RESULTS / "raw" / f"{name}.json.gz", "wt") as fh:
        json.dump(records, fh)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps({"meta": meta | {"commit": git_commit()}, "summary": aggregate(records)}, indent=1))
    return out


def table(records: list[dict], metric: str = "recall", budgets=BUDGETS) -> str:
    agg = aggregate(records)
    methods = sorted({k.split("@")[0] for k in agg})
    lines = ["method".ljust(16) + "".join(f"{b:>8}" for b in budgets)]
    for m in sorted(methods, key=lambda m: -agg.get(f"{m}@{budgets[len(budgets) // 2]}", {}).get(metric, 0)):
        lines.append(m.ljust(16) + "".join(f"{agg.get(f'{m}@{b}', {}).get(metric, float('nan')):8.3f}" for b in budgets))
    return "\n".join(lines)
