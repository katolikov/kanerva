"""Baselines and method variants. A method maps (context, budget) to item ids in rank order.

All methods share the same memory store, candidate generators, token counts and per-item overhead.
Only the selection rule differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kanerva import Memory, RetrievalConfig
from kanerva.policy import LEXICAL_THETA, RERANK_THETA
from kanerva.select import SelectionProblem, budgeted_select

# The first version of the method: a linear prior, no neighbours, a smaller pool. Kept as a baseline.
FIRST_VERSION = {"relevance": "linear", "candidates": 100, "neighbour_top": 0, "gamma": 0.05, "newest_first": False}


@dataclass
class Context:
    mem: Memory
    question: str
    now: float
    qvec: np.ndarray
    limit: int = 100
    overhead: int = 14
    cache: dict = field(default_factory=dict)

    def lexical(self) -> list[tuple[int, float]]:
        if "lex" not in self.cache:
            self.cache["lex"] = self.mem.lexical_search(self.question, self.limit)
        return self.cache["lex"]

    def dense(self) -> list[tuple[int, float]]:
        if "dense" not in self.cache:
            self.cache["dense"] = self.mem.dense_search(self.question, self.limit, query_vector=self.qvec)
        return self.cache["dense"]

    def hybrid(self) -> list[int]:
        if "hybrid" not in self.cache:
            scores: dict[int, float] = {}
            for ranked in (self.lexical(), self.dense()):
                for rank, (i, _) in enumerate(ranked):
                    scores[i] = scores.get(i, 0.0) + 1.0 / (60 + rank + 1)
            self.cache["hybrid"] = sorted(scores, key=lambda i: -scores[i])
        return self.cache["hybrid"]

    def cost(self, ids: list[int]) -> dict[int, int]:
        rows = self.mem._fetch(ids)
        return {i: int(rows[i]["tokens"]) + self.overhead for i in rows}

    def all_recent(self) -> list[int]:
        if "recent" not in self.cache:
            rows = self.mem.db.execute(
                "SELECT id FROM items WHERE tier='active' ORDER BY created_at DESC, id DESC LIMIT 4000"
            ).fetchall()
            self.cache["recent"] = [int(r["id"]) for r in rows]
        return self.cache["recent"]


def fill(order: list[int], ctx: Context, budget: int) -> list[int]:
    """Walk a ranking and keep each item that still fits in the budget."""
    costs = ctx.cost(order)
    chosen, spent = [], 0
    for i in order:
        if spent + costs[i] <= budget:
            chosen.append(i)
            spent += costs[i]
    return chosen


def recency(ctx: Context, budget: int) -> list[int]:
    return fill(ctx.all_recent(), ctx, budget)


def bm25(ctx: Context, budget: int) -> list[int]:
    return fill([i for i, _ in ctx.lexical()], ctx, budget)


def dense(ctx: Context, budget: int) -> list[int]:
    return fill([i for i, _ in ctx.dense()], ctx, budget)


def hybrid(ctx: Context, budget: int) -> list[int]:
    return fill(ctx.hybrid(), ctx, budget)


def hybrid_live(ctx: Context, budget: int) -> list[int]:
    """Hybrid ranking with superseded versions dropped: temporal invalidation on its own."""
    order = ctx.hybrid()
    rows = ctx.mem._fetch(order)
    return fill([i for i in order if rows[i]["valid_to"] is None], ctx, budget)


def hybrid_newest(ctx: Context, budget: int) -> list[int]:
    """The hybrid selection, listed newest first. Same set as hybrid, so only the order changes."""
    chosen = hybrid(ctx, budget)
    rows = ctx.mem._fetch(chosen)
    return sorted(chosen, key=lambda i: -rows[i]["created_at"])


def hybrid_dedup(ctx: Context, budget: int, threshold: float = 0.97) -> list[int]:
    """Hybrid with older near-duplicates removed, listed newest first: a one-line staleness rule."""
    if "dedup" not in ctx.cache:
        ids = ctx.hybrid()
        if not ids:
            return []
        vectors = _vectors(ctx, ids)
        rows = ctx.mem._fetch(ids)
        created = np.array([rows[i]["created_at"] for i in ids])
        sim = vectors @ vectors.T
        np.fill_diagonal(sim, 0.0)
        drop = ((sim >= threshold) & (created[None, :] > created[:, None])).any(axis=1)
        ctx.cache["dedup"] = [i for i, d in zip(ids, drop) if not d]
    chosen = fill(ctx.cache["dedup"], ctx, budget)
    rows = ctx.mem._fetch(chosen)
    return sorted(chosen, key=lambda i: -rows[i]["created_at"])


def _vectors(ctx: Context, ids: list[int]) -> np.ndarray:
    rows = ctx.mem._fetch(ids)
    return np.frombuffer(b"".join(rows[i]["embedding"] for i in ids), dtype=np.float32).reshape(len(ids), -1)


def mmr(ctx: Context, budget: int, lam: float = 0.7) -> list[int]:
    """Maximal marginal relevance over the hybrid pool with dense relevance, then budget fill."""
    if "mmr" not in ctx.cache:
        ids = ctx.hybrid()
        if not ids:
            return []
        vectors = _vectors(ctx, ids)
        relevance = vectors @ ctx.qvec
        sim = vectors @ vectors.T
        remaining = np.ones(len(ids), dtype=bool)
        best = np.full(len(ids), -np.inf)
        order: list[int] = []
        for _ in range(len(ids)):
            redundancy = np.where(np.isinf(best), 0.0, best)
            score = np.where(remaining, lam * relevance - (1 - lam) * redundancy, -np.inf)
            pick = int(np.argmax(score))
            order.append(ids[pick])
            remaining[pick] = False
            best = np.maximum(best, sim[pick])
        ctx.cache["mmr"] = order
    return fill(ctx.cache["mmr"], ctx, budget)


def generative_agents(ctx: Context, budget: int) -> list[int]:
    """Park et al. scoring: recency + relevance over all items; importance needs an LLM and is constant."""
    if "ga" not in ctx.cache:
        ids, matrix = ctx.mem._matrix(None)
        if ids.size == 0:
            return []
        rows = ctx.mem.db.execute("SELECT id, created_at FROM items WHERE tier='active'").fetchall()
        created = {int(r["id"]): r["created_at"] for r in rows}
        hours = np.array([(ctx.now - created[int(i)]) / 3600.0 for i in ids]).clip(min=0.0)
        rec = 0.995 ** hours
        rel = np.asarray(matrix @ ctx.qvec, dtype=float)

        def norm(a: np.ndarray) -> np.ndarray:
            span = a.max() - a.min()
            return (a - a.min()) / span if span > 0 else np.zeros_like(a)

        score = norm(rec) + norm(rel)
        top = np.argsort(-score)[:200]
        ctx.cache["ga"] = [int(ids[i]) for i in top]
    return fill(ctx.cache["ga"], ctx, budget)


def facility_location(ctx: Context, budget: int) -> list[int]:
    """Relevance-weighted facility location under a knapsack, as in token-budgeted RAG selectors."""
    ids = ctx.hybrid()
    if not ids:
        return []
    if "fl" not in ctx.cache:
        vectors = _vectors(ctx, ids)
        relevance = np.clip(vectors @ ctx.qvec, 0.0, 1.0)
        costs = ctx.cost(ids)
        ctx.cache["fl"] = SelectionProblem(
            weights=np.zeros(len(ids)),
            costs=np.array([costs[i] for i in ids], dtype=float),
            affinity=np.ones((len(ids), 1)),
            facet_weights=np.ones(1),
            similarity=np.clip(vectors @ vectors.T, 0.0, 1.0),
            prior_weights=relevance,
            gamma=1.0,
        )
    picked = budgeted_select(ctx.cache["fl"], budget, (1.0,))
    return [ids[i] for i in picked]


def rerank(ctx: Context, budget: int, pool: int = 50) -> list[int]:
    """Hybrid order with its top ``pool`` items re-sorted by the cross-encoder."""
    if "rerank" not in ctx.cache:
        ids = ctx.hybrid()
        top = ids[:pool]
        rows = ctx.mem._fetch(top)
        logits = np.asarray(ctx.mem.reranker(ctx.question, [rows[i]["text"] for i in top]), dtype=float)
        order = np.argsort(-np.nan_to_num(logits, nan=-1e9), kind="stable")
        ctx.cache["rerank"] = [top[i] for i in order] + ids[pool:]
    return fill(ctx.cache["rerank"], ctx, budget)


def kanerva_variant(cfg: RetrievalConfig):
    def run(ctx: Context, budget: int) -> list[int]:
        key = ("kanerva", id(cfg))
        if key not in ctx.cache:
            ctx.cache[key] = ctx.mem.prepare(ctx.question, ctx.now, None, cfg, ctx.qvec)
        result = ctx.mem.select(ctx.cache[key], budget, record=False)
        if any(item.truncated for item in result.items):
            ctx.cache["texts"] = [item.text for item in result.items]
            ctx.cache["text_costs"] = [ctx.mem.count_tokens(item.text) + ctx.overhead for item in result.items]
        return [item.id for item in result.items]

    return run


BASELINES = {
    "recency": recency,
    "bm25": bm25,
    "dense": dense,
    "hybrid": hybrid,
    "hybrid-live": hybrid_live,
    "hybrid-newest": hybrid_newest,
    "hybrid-dedup": hybrid_dedup,
    "mmr": mmr,
    "gen-agents": generative_agents,
    "facility": facility_location,
    "rerank": rerank,
}


def variants(base: RetrievalConfig) -> dict[str, RetrievalConfig]:
    """The full static method and its ablations (one component removed at a time)."""
    return {
        "kanerva": base,
        "-density": base.with_(cost_exponents=(0.0,)),
        "-validity": base.with_(use_validity=False),
        "-supersession": base.with_(use_supersession=False),
        "kanerva-ce": base.with_(use_reranker=True, theta0=RERANK_THETA),
        "kanerva-lex": base.with_(use_dense=False, gamma=0.0, neighbour_top=0, theta0=LEXICAL_THETA),
        "kanerva-r1": base.with_(**FIRST_VERSION),
        "-neighbours": base.with_(candidates=100, neighbour_top=0),
        "kanerva-x128": base.with_(excerpt_threshold=160, excerpt_tokens=128),
    }
