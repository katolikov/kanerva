"""E5-E8: consolidation, forgetting and store growth, multi-resolution retrieval, and scale."""

from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.embed_cache import TextCache
from experiments.harness import RESULTS, apply_write, git_commit, new_memory
from experiments.registry import load
from kanerva import Memory, RetrievalConfig
from kanerva.consolidate import consolidate, extract_summary, forget
from kanerva.embed import HashingEmbedder
from kanerva.text import split_sentences

BUDGET = 1024


def dump(name: str, payload: dict) -> None:
    payload["meta"] = payload.get("meta", {}) | {"experiment": name, "commit": git_commit()}
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=1))


def _consolidation_job(args: tuple) -> dict:
    stream, dataset, threshold, rate = args
    cache = TextCache(dataset)
    mem = new_memory(cache, RetrievalConfig())
    id_of: dict[str, int] = {}
    for turn in stream.writes:
        apply_write(mem, cache, turn, id_of, False)
    before = mem.stats()["active_tokens"]
    report = consolidate(mem, threshold=threshold, rate=rate) if threshold is not None else None
    parents: dict[int, int] = {}
    for row in mem.db.execute("SELECT id, parents FROM items WHERE kind='semantic' AND parents<>'[]'"):
        for p in json.loads(row["parents"]):
            parents[p] = int(row["id"])
    recalls = []
    for query in stream.queries:
        gold = {id_of[t] for t in query.gold if t in id_of}
        if not gold:
            continue
        mapped = {parents.get(g, g) for g in gold}
        result = mem.retrieve(query.question, BUDGET, now=query.now, query_vector=cache.vector(query.question), record=False)
        chosen = {i.id for i in result.items}
        recalls.append(len(chosen & mapped) / len(mapped))
    out = {"before": before, "after": mem.stats()["active_tokens"], "recall": recalls,
           "clusters": report.clusters if report else 0, "sources": report.sources if report else 0,
           "distortion": report.distortion if report else 0.0}
    mem.close()
    return out


def consolidation() -> None:
    results: dict = {}
    grid = [(None, 0.0)] + [(t, r) for t in (0.6, 0.7, 0.8, 0.9) for r in (0.3, 0.5)]
    for dataset in ("locomo", "lme"):
        streams = [s for s in load(dataset) if s.split == "test"]
        if dataset == "lme":
            streams = streams[:60]
        for threshold, rate in grid:
            jobs = [(s, dataset, threshold, rate) for s in streams]
            with ProcessPoolExecutor(8) as pool:
                rows = list(pool.map(_consolidation_job, jobs))
            before, after = sum(r["before"] for r in rows), sum(r["after"] for r in rows)
            recall = [x for r in rows for x in r["recall"]]
            key = f"{dataset}|{threshold}|{rate}"
            results[key] = {"threshold": threshold, "rate": rate, "tokens_before": before, "tokens_after": after,
                            "saving": 1 - after / before, "recall": float(np.mean(recall)), "n": len(recall),
                            "clusters": sum(r["clusters"] for r in rows), "sources": sum(r["sources"] for r in rows)}
            print(key, {k: round(v, 4) if isinstance(v, float) else v for k, v in results[key].items()}, flush=True)
    dump("consolidation", {"results": results, "meta": {"budget": BUDGET, "lme_streams": 60}})


def rate_distortion() -> None:
    """Empirical D(R) on real clusters of sentences, against the bound of Theorem 3 with Z* = X."""
    streams = [s for s in load("locomo") if s.split == "test"]
    embedder = HashingEmbedder()
    rng = random.Random(0)
    rates = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
    curves: dict = {r: [] for r in rates}
    violations = 0
    total = 0
    for stream in streams:
        turns = [t.text for t in stream.writes]
        rng.shuffle(turns)
        for start in range(0, min(len(turns), 400), 8):
            sentences = list(dict.fromkeys(s for t in turns[start:start + 8] for s in split_sentences(t)))
            if len(sentences) < 4:
                continue
            count = lambda s: len(s.split())
            cost = sum(count(s) for s in sentences)
            for r in rates:
                _, d = extract_summary(sentences, embedder, count, max(int(r * cost), 1))
                curves[r].append(d)
                total += 1
                violations += d > np.exp(-r) + 1e-9
    dump("rate_distortion", {"rates": rates, "mean_distortion": [float(np.mean(curves[r])) for r in rates],
                             "bound": [float(np.exp(-r)) for r in rates], "violations": int(violations), "total": total,
                             "meta": {"embedder": "hashing"}})


def _forgetting_job(args: tuple) -> dict:
    stream, idle_days, capacity = args
    cache = TextCache("repomem")
    mem = new_memory(cache, RetrievalConfig())
    id_of: dict[str, int] = {}
    cursor = 0
    trace = []
    recalls = []
    last_session = None
    for query in stream.queries:
        while cursor < query.pos:
            apply_write(mem, cache, stream.writes[cursor], id_of, True)
            cursor += 1
        session = query.qid.split("-")[1]
        if session != last_session:
            last_session = session
            if idle_days is not None or capacity is not None:
                forget(mem, query.now, idle_days=idle_days, capacity_tokens=capacity)
            stats = mem.stats()
            trace.append({"writes": cursor, "active": stats["active"], "tokens": stats["active_tokens"]})
        gold = {id_of[t] for t in query.gold if t in id_of}
        if not gold:
            continue
        result = mem.retrieve(query.question, BUDGET, now=query.now, query_vector=cache.vector(query.question))
        chosen = {i.id for i in result.items}
        recalls.append(len(chosen & gold) / len(gold))
    mem.close()
    return {"trace": trace, "recall": recalls}


def forgetting() -> None:
    streams = [s for s in load("repomem") if s.split == "test"]
    settings = {"keep-all": (None, None), "idle-730d": (730.0, None), "idle-365d": (365.0, None), "idle-180d": (180.0, None),
                "cap-8k": (None, 8000), "cap-4k": (None, 4000)}
    out: dict = {}
    for name, (idle, cap) in settings.items():
        with ProcessPoolExecutor(8) as pool:
            rows = list(pool.map(_forgetting_job, [(s, idle, cap) for s in streams]))
        final = [r["trace"][-1] for r in rows if r["trace"]]
        out[name] = {"recall": float(np.mean([x for r in rows for x in r["recall"]])),
                     "final_active": float(np.mean([f["active"] for f in final])),
                     "final_tokens": float(np.mean([f["tokens"] for f in final])),
                     "max_tokens": float(np.mean([max(t["tokens"] for t in r["trace"]) for r in rows if r["trace"]])),
                     "trace": rows[0]["trace"]}
        print(name, {k: round(v, 3) for k, v in out[name].items() if k != "trace"}, flush=True)
    dump("forgetting", {"results": out, "meta": {"budget": BUDGET, "streams": len(streams)}})


def _snippet_job(args: tuple) -> dict:
    stream, threshold = args
    cache = TextCache("lme")
    cfg = RetrievalConfig(snippet_threshold=threshold)
    mem = new_memory(cache, cfg)
    id_of: dict[str, int] = {}
    for turn in stream.writes:
        apply_write(mem, cache, turn, id_of, False)
    out = {}
    for query in stream.queries:
        gold = {id_of[t] for t in query.gold if t in id_of}
        if not gold:
            continue
        for budget in (256, 512, 1024, 2048):
            result = mem.retrieve(query.question, budget, now=query.now, query_vector=cache.vector(query.question), record=False)
            chosen = {i.id for i in result.items}
            out[budget] = {"recall": len(chosen & gold) / len(gold), "tokens": result.tokens,
                           "items": len(result.items), "cut": sum(i.truncated for i in result.items)}
    mem.close()
    return out


def multires() -> None:
    streams = [s for s in load("lme") if s.split == "test"]
    out: dict = {}
    for threshold in (0, 48, 96):
        with ProcessPoolExecutor(8) as pool:
            rows = list(pool.map(_snippet_job, [(s, threshold) for s in streams]))
        for budget in (256, 512, 1024, 2048):
            subset = [r[budget] for r in rows if budget in r]
            out[f"{threshold}@{budget}"] = {k: float(np.mean([r[k] for r in subset])) for k in ("recall", "tokens", "items", "cut")}
            print(threshold, budget, {k: round(v, 3) for k, v in out[f"{threshold}@{budget}"].items()}, flush=True)
    dump("multires", {"results": out})


def scale() -> None:
    cache = TextCache("lme")
    streams = [s for s in load("lme")]
    texts = list(dict.fromkeys(t.text for s in streams for t in s.writes))
    rng = random.Random(0)
    rng.shuffle(texts)
    questions = [q for s in streams for q in s.queries][:300]
    out: dict = {}
    for n in (1000, 10000, 100000):
        mem = Memory(":memory:", count_tokens=cache.count)
        start = time.perf_counter()
        for i, text in enumerate(texts[:n]):
            mem.write(text, at=1.6e9 + i * 60, embedding=cache.vector(text))
        write_ms = (time.perf_counter() - start) * 1000 / n
        for mode, cfg in (("hybrid", RetrievalConfig()), ("lexical", RetrievalConfig(use_dense=False, gamma=0.0))):
            latencies = []
            for query in questions:
                vector = cache.vector(query.question)
                start = time.perf_counter()
                mem.retrieve(query.question, 1024, now=1.8e9, query_vector=vector, config=cfg, record=False)
                latencies.append((time.perf_counter() - start) * 1000)
            out[f"{mode}@{n}"] = {"write_ms": write_ms, "p50_ms": float(np.percentile(latencies, 50)), "p95_ms": float(np.percentile(latencies, 95))}
            print(mode, n, out[f"{mode}@{n}"], flush=True)
        mem.close()
    dump("scale", {"results": out})


if __name__ == "__main__":
    jobs = {"consolidation": consolidation, "rd": rate_distortion, "forgetting": forgetting, "multires": multires, "scale": scale}
    for name in sys.argv[1:] or jobs:
        jobs[name]()
