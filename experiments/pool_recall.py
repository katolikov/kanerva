"""Candidate-pool recall on development data: pool size, neighbour expansion and pseudo-relevance feedback."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.dump_features import DATASETS
from experiments.embed_cache import TextCache
from experiments.harness import RESULTS, apply_write, git_commit, new_memory
from experiments.registry import load
from kanerva import RetrievalConfig

SIZES = (50, 100, 200, 400)


def expand_neighbours(mem, ids: list[int], top: int, reach: int) -> set[int]:
    """Items written within ``reach`` turns of the ``top`` best candidates, in the same session."""
    out: set[int] = set()
    rows = mem._fetch(ids[:top])
    for i in ids[:top]:
        session = rows[i]["session"]
        near = mem.db.execute(
            "SELECT id FROM items WHERE id BETWEEN ? AND ? AND session IS ? AND tier='active'", (i - reach, i + reach, session)
        ).fetchall()
        out.update(int(r["id"]) for r in near)
    return out


def prf_vector(mem, qvec: np.ndarray, dense: list[tuple[int, float]], k: int, alpha: float) -> np.ndarray:
    """Rocchio: the query vector moved toward the mean of the top-k dense hits."""
    all_ids, matrix = mem._matrix(None)
    pos = np.searchsorted(all_ids, [i for i, _ in dense[:k]])
    centroid = matrix[pos].mean(0)
    v = (1 - alpha) * qvec + alpha * centroid
    return v / (np.linalg.norm(v) or 1.0)


def stream_pools(args) -> list[dict]:
    stream, cache_name, keyed = args
    cache = TextCache(cache_name)
    mem = new_memory(cache, RetrievalConfig())
    id_of: dict[str, int] = {}
    cursor = 0
    out = []
    for query in stream.queries:
        while cursor < query.pos:
            apply_write(mem, cache, stream.writes[cursor], id_of, keyed)
            cursor += 1
        gold = {id_of[t] for t in query.gold if t in id_of}
        if not gold:
            continue
        qvec = cache.vector(query.question)
        lex = mem.lexical_search(query.question, max(SIZES))
        dense = mem.dense_search(query.question, max(SIZES), query_vector=qvec)
        row = {"category": query.category, "stream": stream.name}
        for n in SIZES:
            pool = {i for i, _ in lex[:n]} | {i for i, _ in dense[:n]}
            row[f"pool{n}"] = len(gold & pool) / len(gold)
            row[f"size{n}"] = len(pool)
        base = {i for i, _ in lex[:100]} | {i for i, _ in dense[:100]}
        fused: dict[int, float] = {}
        for ranked in (lex[:100], dense[:100]):
            for rank, (i, _) in enumerate(ranked):
                fused[i] = fused.get(i, 0.0) + 1.0 / (61 + rank)
        order = sorted(fused, key=lambda i: -fused[i])
        for top, reach in ((20, 1), (50, 1), (20, 2)):
            pool = base | expand_neighbours(mem, order, top, reach)
            row[f"nb{top}x{reach}"] = len(gold & pool) / len(gold)
            row[f"nb{top}x{reach}_size"] = len(pool)
        prf = mem.dense_search(query.question, 100, query_vector=prf_vector(mem, qvec, dense, 5, 0.5))
        pool = base | {i for i, _ in prf}
        row["prf"] = len(gold & pool) / len(gold)
        row["prf_size"] = len(pool)
        out.append(row)
    mem.close()
    return out


def main(names: list[str]) -> None:
    report = {}
    for name in names:
        loader, cache_name, keyed, splits = DATASETS[name]
        streams = [s for s in load(loader) if s.split in splits]
        with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 3)) as pool:
            rows = [r for part in pool.map(stream_pools, [(s, cache_name, keyed) for s in streams]) for r in part]
        keys = [k for k in rows[0] if k not in ("category", "stream")]
        report[name] = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in keys}
        cats = sorted({r["category"] for r in rows})
        report[name]["by_category"] = {
            c: {k: round(float(np.mean([r[k] for r in rows if r["category"] == c])), 3) for k in ("pool100", "pool400", "nb20x1", "prf")}
            for c in cats
        }
        print(name, json.dumps({k: v for k, v in report[name].items() if k != "by_category"}), flush=True)
    (RESULTS / "dev_pool.json").write_text(json.dumps({"meta": {"commit": git_commit()}, "results": report}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["beam-dev", "locomo", "repomem", "lme"])
