"""Dump per-candidate features and labels on development data, for fitting and comparing relevance models offline.

For every query: the round-1 candidate pool (after staleness removal), the round-1 features, the round-2 features,
gold and stale labels, token costs, the round-1 method's selection at each budget, and which answer stems each
candidate contains. Writes experiments/cache/r2_<dataset>.npz.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.embed_cache import CACHE, TextCache
from experiments.harness import BUDGETS, apply_write, new_memory
from experiments.registry import load
from kanerva import RetrievalConfig
from kanerva.relevance import ItemView, feature_matrix, stems
from kanerva.text import content_terms, light_stem

# Candidate generation for the dump, as RetrievalConfig overrides; R2_POOL='{"candidates": 200}' for example.
POOL = json.loads(os.environ.get("R2_POOL", "{}"))
SUFFIX = os.environ.get("R2_SUFFIX", "")

DATASETS = {
    # name: (loader, embedding cache, keyed, splits used as development data)
    "lme": ("lme", "lme", False, ("dev", "test")),
    "locomo": ("locomo", "locomo", False, ("dev", "test")),
    "repomem": ("repomem", "repomem", True, ("dev", "test")),
    "repomem-holdout": ("repomem-holdout", "repomem-holdout", True, ("holdout",)),
    "beam-dev": ("beam-dev", "beam", False, ("dev",)),
    "lme-updates": ("lme-updates", "lme", False, ("dev", "test")),
    # The static model2vec embedder that the plugin can use; no cross-encoder features.
    "lme-m2v": ("lme", "lme-m2v", False, ("dev", "test")),
    "locomo-m2v": ("locomo", "locomo-m2v", False, ("dev", "test")),
    "repomem-m2v": ("repomem", "repomem-m2v", True, ("dev", "test")),
    "beam-dev-m2v": ("beam-dev", "beam-m2v", False, ("dev",)),
}


class LiveCrossEncoder:
    """Cross-encoder logits computed on demand and memoised to experiments/cache/ce_<name>.json."""

    def __init__(self, name: str):
        from experiments.run_rerank import MODEL, pair_key

        self.path = CACHE / f"ce_{name}.json"
        self.scores: dict[str, float] = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.key = pair_key
        self.model_name = MODEL
        self.model = None
        self.dirty = False

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        missing = [t for t in texts if self.key(query, t) not in self.scores]
        if missing:
            if self.model is None:
                from sentence_transformers import CrossEncoder

                self.model = CrossEncoder(self.model_name)
            values = self.model.predict([(query, t[:1500]) for t in missing], batch_size=32, show_progress_bar=False)
            for t, v in zip(missing, values):
                self.scores[self.key(query, t)] = round(float(v), 4)
            self.dirty = True
        return [self.scores[self.key(query, t)] for t in texts]

    def save(self) -> None:
        """Merge into the file on disk, since parallel workers may have added other pairs meanwhile."""
        if self.dirty:
            merged = json.loads(self.path.read_text()) if self.path.exists() else {}
            merged.update(self.scores)
            tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(merged))
            tmp.replace(self.path)


def answer_stems(answer: str, question: str) -> list[str]:
    asked = {light_stem(t) for t in content_terms(question)}
    return [s for s in dict.fromkeys(light_stem(t) for t in content_terms(answer)) if s not in asked][:62]


def dump_stream(args) -> dict:
    stream, cache_name, keyed, ce_name = args
    cache = TextCache(cache_name)
    reranker = LiveCrossEncoder(ce_name) if ce_name else None
    cfg = RetrievalConfig(use_bandit=False, use_reranker=reranker is not None, relevance="linear", **POOL)
    mem = new_memory(cache, cfg, reranker)
    id_of: dict[str, int] = {}
    out: dict[str, list] = {k: [] for k in ("x1", "x2", "gold", "stale", "cost", "answer", "qid", "category", "old", "nans", "stream")}
    cursor = 0
    for query in stream.queries:
        while cursor < query.pos:
            apply_write(mem, cache, stream.writes[cursor], id_of, keyed)
            cursor += 1
        gold = {id_of[t] for t in query.gold if t in id_of}
        if not gold:
            continue
        stale = {id_of[t] for t in query.stale if t in id_of} - gold
        qvec = cache.vector(query.question)
        prepared = mem.prepare(query.question, query.now, None, cfg, qvec)
        cand, rows = prepared.candidates, prepared.rows
        if not cand.ids:
            continue
        near, near2, session = mem.context_signals(cand.ids, qvec, None)
        views = [ItemView(rows[i]["text"], int(rows[i]["tokens"]), float(rows[i]["created_at"]), rows[i]["session"]) for i in cand.ids]
        x2 = feature_matrix(query.question, views, cand.lex_rank, cand.dense_rank, cand.lex_score, cand.dense_score,
                            cand.term_cover, cand.activation, near, near2, session, cand.rerank_rank, cand.rerank_score)
        wanted = answer_stems(query.answer, query.question) if query.answer else []
        masks = []
        for i in cand.ids:
            have = stems(rows[i]["text"])
            masks.append(sum(1 << b for b, s in enumerate(wanted) if s in have))
        old_cfg = RetrievalConfig(use_bandit=False, relevance="linear")
        old_prepared = mem.prepare(query.question, query.now, None, old_cfg, qvec)
        old = []
        for budget in BUDGETS:
            chosen = {item.id for item in mem.select(old_prepared, budget, record=False).items}
            old.append(len(chosen & gold) / len(gold))
        out["x1"].append(prepared.features.astype(np.float32))
        out["x2"].append(x2.astype(np.float32))
        out["gold"].append(np.array([i in gold for i in cand.ids]))
        out["stale"].append(np.array([i in stale for i in cand.ids]))
        out["cost"].append(np.array([int(rows[i]["tokens"]) + 14 for i in cand.ids], dtype=np.int32))
        out["answer"].append(np.array(masks, dtype=np.int64))
        out["nans"].append(len(wanted))
        out["qid"].append(query.qid)
        out["category"].append(query.category)
        out["old"].append(old)
        out["stream"].append(stream.name)
        out.setdefault("ngold", []).append(len(gold))
        out.setdefault("split", []).append(stream.split)
    mem.close()
    if reranker is not None:
        reranker.save()
    return out


def dump(name: str, workers: int | None = None) -> None:
    loader, cache_name, keyed, splits = DATASETS[name]
    streams = [s for s in load(loader) if s.split in splits]
    ce_name = None if cache_name.endswith("-m2v") else cache_name
    start = time.time()
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    parts = []
    jobs = [(s, cache_name, keyed, ce_name) for s in streams]
    if name == "beam-dev" and ce_name:
        # Live cross-encoder scoring writes one shared cache file, so these streams run in one process.
        parts = [dump_stream(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers) as pool:
            parts = list(pool.map(dump_stream, jobs))
    merged: dict[str, list] = {}
    for part in parts:
        for k, v in part.items():
            merged.setdefault(k, []).extend(v)
    sizes = np.array([len(g) for g in merged["gold"]])
    np.savez_compressed(
        CACHE / f"r2_{name}{SUFFIX}.npz",
        offsets=np.concatenate([[0], np.cumsum(sizes)]),
        x1=np.vstack(merged["x1"]), x2=np.vstack(merged["x2"]),
        gold=np.concatenate(merged["gold"]), stale=np.concatenate(merged["stale"]),
        cost=np.concatenate(merged["cost"]), answer=np.concatenate(merged["answer"]),
        nans=np.array(merged["nans"]), ngold=np.array(merged["ngold"]), old=np.array(merged["old"]),
        qid=np.array(merged["qid"]), category=np.array(merged["category"]), stream=np.array(merged["stream"]),
        split=np.array(merged["split"]),
    )
    print(f"{name}: {len(sizes)} queries, {sizes.sum()} candidates, {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or DATASETS:
        dump(name)
