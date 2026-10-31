"""Reversible consolidation and forgetting.

Consolidation merges clusters of similar episodic items into one extractive semantic item and moves the
sources to the archive tier, so that ``Memory.expand`` can restore them. Forgetting moves idle or
low-value items to the archive. Only ``Memory.delete`` removes data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .embed import Embedder, HashingEmbedder
from .memory import DAY, Memory
from .text import split_sentences
from .validity import activation_score


@dataclass
class ConsolidationReport:
    clusters: int = 0
    sources: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    distortion: float = 0.0  # token-weighted mean of 1 - coverage over merged clusters


def _star_clusters(sim: np.ndarray, threshold: float) -> list[list[int]]:
    """Leader clustering: every member is within ``threshold`` of its leader, so clusters cannot chain."""
    free = np.ones(sim.shape[0], dtype=bool)
    groups: list[list[int]] = []
    for leader in range(sim.shape[0]):
        if not free[leader]:
            continue
        members = np.flatnonzero(free & (sim[leader] >= threshold))
        free[members] = False
        if members.size > 1:
            groups.append(members.tolist())
    return groups


def extract_summary(
    sentences: list[str], embedder: Embedder, count_tokens, budget: int
) -> tuple[list[int], float]:
    """Density greedy on sentence facility location. Returns chosen indices and the distortion."""
    vectors = embedder.encode(sentences)
    sim = np.clip(vectors @ vectors.T, 0.0, 1.0)
    costs = np.array([max(count_tokens(s), 1) for s in sentences], dtype=float)
    weights = costs / costs.sum()
    best = np.zeros(len(sentences))
    chosen: list[int] = []
    spent = 0.0
    available = np.ones(len(sentences), dtype=bool)
    while available.any():
        gains = np.clip(sim - best, 0.0, None) @ weights
        density = np.where(available, gains / costs, -np.inf)
        i = int(np.argmax(density))
        available[i] = False
        if gains[i] <= 1e-9:
            break
        if spent + costs[i] > budget:
            continue
        chosen.append(i)
        spent += costs[i]
        best = np.maximum(best, sim[i])
    if not chosen:
        chosen = [int(np.argmax(sim @ weights))]
        best = sim[chosen[0]]
    return sorted(chosen), float(1.0 - weights @ best)


def consolidate(
    mem: Memory,
    scope: str | None = None,
    threshold: float = 0.9,
    rate: float = 0.5,
    now: float | None = None,
    max_items: int = 4000,
) -> ConsolidationReport:
    """Merge clusters of live episodic items whose pairwise similarity reaches ``threshold``."""
    embedder = mem.embedder or HashingEmbedder()
    sql = "SELECT * FROM items WHERE tier='active' AND kind='episodic' AND valid_to IS NULL AND pinned=0"
    args: list = []
    if scope is not None:
        sql += " AND scope=?"
        args.append(scope)
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(max_items)
    rows = mem.db.execute(sql, args).fetchall()
    report = ConsolidationReport()
    if len(rows) < 2:
        return report
    if all(r["embedding"] is not None for r in rows):
        vectors = np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    else:
        vectors = embedder.encode([r["text"] for r in rows])
    distortion_mass = 0.0
    for group in _star_clusters(vectors @ vectors.T, threshold):
        members = [rows[i] for i in group]
        if len({m["scope"] for m in members}) > 1:
            continue
        sentences = list(dict.fromkeys(s for m in members for s in split_sentences(m["text"])))
        before = sum(m["tokens"] for m in members)
        chosen, distortion = extract_summary(sentences, embedder, mem.count_tokens, max(int(rate * before), 16))
        text = " ".join(sentences[i] for i in chosen)
        at = max(m["created_at"] for m in members) if now is None else now
        item = mem.write(
            text,
            scope=members[0]["scope"],
            kind="semantic",
            at=at,
            parents=[m["id"] for m in members],
            source={"consolidated": True, "distortion": distortion},
        )
        if item is None or item in {m["id"] for m in members}:
            continue
        mem.archive(m["id"] for m in members)
        report.clusters += 1
        report.sources += len(members)
        report.tokens_before += before
        report.tokens_after += mem.count_tokens(text)
        distortion_mass += distortion * before
    if report.tokens_before:
        report.distortion = distortion_mass / report.tokens_before
    return report


def forget(
    mem: Memory,
    now: float,
    idle_days: float | None = None,
    capacity_tokens: int | None = None,
    scope: str | None = None,
) -> int:
    """Archive items idle for ``idle_days`` and then the lowest value-density items above capacity."""
    sql = "SELECT * FROM items WHERE tier='active' AND pinned=0"
    args: list = []
    if scope is not None:
        sql += " AND scope=?"
        args.append(scope)
    rows = mem.db.execute(sql, args).fetchall()
    if not rows:
        return 0
    ids = [int(r["id"]) for r in rows]
    ages = mem._access_ages(ids, now)
    evict: list[int] = []
    kept: list[tuple[float, int, int]] = []
    for row in rows:
        item_id = int(row["id"])
        idle = min(ages[item_id], default=float("inf"))
        if idle_days is not None and idle > idle_days:
            evict.append(item_id)
            continue
        survival = 0.05 if row["valid_to"] is not None else mem._hazard_for(row["key"], row["kind"]).survival(idle)
        usefulness = row["used"] / (row["used"] + row["unused"])
        density = survival * activation_score(ages[item_id]) * usefulness / max(row["tokens"], 1)
        kept.append((density, item_id, int(row["tokens"])))
    if capacity_tokens is not None:
        total = sum(t for _, _, t in kept)
        for _, item_id, tokens in sorted(kept):
            if total <= capacity_tokens:
                break
            evict.append(item_id)
            total -= tokens
    return mem.archive(evict)


def expand(mem: Memory, item_id: int) -> list[dict]:
    """Return an item's stored text and its archived sources, one level down."""
    row = mem.db.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
    if row is None:
        return []
    out = [{"id": item_id, "text": row["text"], "kind": row["kind"]}]
    for parent in json.loads(row["parents"]):
        source = mem.db.execute("SELECT * FROM items WHERE id=?", (parent,)).fetchone()
        if source is not None:
            out.append({"id": int(parent), "text": source["text"], "kind": source["kind"]})
    return out


__all__ = ["ConsolidationReport", "DAY", "consolidate", "expand", "extract_summary", "forget"]
