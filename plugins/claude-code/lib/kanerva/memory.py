"""SQLite-backed memory: write, retrieve, feedback, consolidate, forget."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .bandit import LinearBandit
from .embed import Embedder, HashingEmbedder
from .policy import (
    BANDIT_THETA,
    LEXICAL_THETA,
    RERANK_THETA,
    Candidates,
    RetrievalConfig,
    bandit_features,
    feature_matrix,
    subset,
)
from .relevance import ItemView, load_model
from .relevance import feature_matrix as relevance_features
from .secrets_filter import redact
from .select import SelectionProblem, budgeted_select, optimality_certificate, value
from .text import (
    approx_tokens,
    content_terms,
    fts_query,
    light_stem,
    query_snippet,
    query_window,
    search_text,
    tokenize,
)
from .validity import HazardPosterior, activation_score

DAY = 86400.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    scope TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    key TEXT,
    digest TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    created_at REAL NOT NULL,
    confirmed_at REAL NOT NULL,
    valid_to REAL,
    superseded_by INTEGER,
    tier TEXT NOT NULL DEFAULT 'active',
    pinned INTEGER NOT NULL DEFAULT 0,
    used REAL NOT NULL DEFAULT 1,
    unused REAL NOT NULL DEFAULT 1,
    parents TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL DEFAULT '{}',
    embedding BLOB,
    session TEXT
);
CREATE INDEX IF NOT EXISTS items_scope ON items(scope, tier);
CREATE INDEX IF NOT EXISTS items_key ON items(scope, key);
CREATE INDEX IF NOT EXISTS items_digest ON items(scope, digest);
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(body, tokenize = 'porter unicode61');
CREATE TABLE IF NOT EXISTS accesses (item_id INTEGER NOT NULL, at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS accesses_item ON accesses(item_id);
CREATE TABLE IF NOT EXISTS retrievals (
    id INTEGER PRIMARY KEY,
    at REAL NOT NULL,
    query TEXT NOT NULL,
    item_ids TEXT NOT NULL,
    features BLOB NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_HISTORICAL = re.compile(
    r"\b(previous(ly)?|used to|formerly|originally|initial(ly)?|earlier|before|old|former|prior|"
    r"history|changed from|at first|first time)\b",
    re.I,
)


@dataclass
class Item:
    id: int
    text: str
    kind: str
    scope: str
    tokens: int
    created_at: float
    key: str | None = None
    superseded: bool = False
    weight: float = 0.0
    truncated: bool = False

    def render(self) -> str:
        date = time.strftime("%Y-%m-%d", time.gmtime(self.created_at))
        mark = " superseded" if self.superseded else ""
        mark += " excerpt" if self.truncated else ""
        return f"[#{self.id} {date}{mark}] {self.text}"


@dataclass
class RetrievalResult:
    items: list[Item]
    tokens: int
    budget: int
    retrieval_id: int | None = None
    objective: float = 0.0
    upper_bound: float = 0.0
    stats: dict = field(default_factory=dict)

    def render(self) -> str:
        return "\n".join(item.render() for item in self.items)


@dataclass
class Prepared:
    query: str
    now: float
    config: RetrievalConfig
    candidates: Candidates
    rows: dict
    problem: SelectionProblem | None
    features: np.ndarray | None
    snippets: dict[int, str]


class Memory:
    """A memory store. ``path`` may be ``":memory:"``; ``embedder`` is optional."""

    def __init__(
        self,
        path: str | Path = ":memory:",
        embedder: Embedder | None = None,
        count_tokens: Callable[[str], int] = approx_tokens,
        config: RetrievalConfig | None = None,
        hazard_prior: tuple[float, float] = (1.0, 365.0),
        reranker: Callable[[str, list[str]], Sequence[float]] | None = None,
    ):
        if config is None:
            config = RetrievalConfig() if embedder else RetrievalConfig(use_dense=False, gamma=0.0, neighbour_top=0, theta0=LEXICAL_THETA)
            if reranker is not None:
                config = config.with_(use_reranker=True, theta0=RERANK_THETA)
        self.reranker = reranker
        self.config = config
        self.embedder = embedder
        self.count_tokens = count_tokens
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(_SCHEMA)
        columns = {r["name"] for r in self.db.execute("PRAGMA table_info(items)")}
        if "session" not in columns:
            self.db.execute("ALTER TABLE items ADD COLUMN session TEXT")
        self.hazard_prior = hazard_prior
        self.hazards: dict[str, HazardPosterior] = {}
        self.bandit = self._load_bandit()
        self._matrix_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._session_codes: dict[tuple, np.ndarray] = {}
        self._writes_since_refresh = 0

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    # write ---------------------------------------------------------------------------------

    def write(
        self,
        text: str,
        *,
        scope: str = "global",
        kind: str = "episodic",
        key: str | None = None,
        at: float | None = None,
        source: dict | None = None,
        parents: Sequence[int] = (),
        embedding: np.ndarray | None = None,
        pinned: bool = False,
        session: str | None = None,
    ) -> int | None:
        """Store an item and return its id, or the id of the identical item it reconfirms.

        A write with a ``key`` that already has a live item supersedes that item, keeping it for
        history. Secrets are redacted. Empty text returns ``None``.
        """
        text, secret_kinds = redact(text.strip())
        if not text:
            return None
        now = time.time() if at is None else at
        digest = hashlib.sha1(" ".join(text.lower().split()).encode()).hexdigest()
        row = self.db.execute(
            "SELECT id FROM items WHERE scope=? AND digest=? AND valid_to IS NULL AND tier='active'",
            (scope, digest),
        ).fetchone()
        if row is not None:
            self.db.execute("UPDATE items SET confirmed_at=? WHERE id=?", (now, row["id"]))
            self.db.execute("INSERT INTO accesses VALUES (?, ?)", (row["id"], now))
            return int(row["id"])
        if embedding is None and self.embedder is not None:
            embedding = self.embedder.encode([text])[0]
        blob = None if embedding is None else np.asarray(embedding, dtype=np.float32).tobytes()
        src = dict(source or {})
        if session is None and src.get("session") is not None:
            session = str(src["session"])
        if secret_kinds:
            src["redacted"] = secret_kinds
        cursor = self.db.execute(
            "INSERT INTO items (scope, kind, text, key, digest, tokens, created_at, confirmed_at,"
            " parents, source, embedding, pinned, session) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (scope, kind, text, key, digest, self.count_tokens(text), now, now,
             json.dumps(list(parents)), json.dumps(src), blob, int(pinned), session),
        )
        item_id = int(cursor.lastrowid)
        self.db.execute("INSERT INTO items_fts (rowid, body) VALUES (?, ?)", (item_id, search_text(text)))
        self.db.execute("INSERT INTO accesses VALUES (?, ?)", (item_id, now))
        if key is not None:
            self.db.execute(
                "UPDATE items SET valid_to=?, superseded_by=? WHERE scope=? AND key=? AND valid_to IS NULL AND id<>?",
                (now, item_id, scope, key, item_id),
            )
        self._matrix_cache.clear()
        self._writes_since_refresh += 1
        return item_id

    def refresh_hazards(self, now: float) -> dict[str, HazardPosterior]:
        """Recompute the per-class Gamma posteriors from the version history in the store."""
        rows = self.db.execute(
            "SELECT CASE WHEN key IS NOT NULL AND instr(key, ':') > 0 THEN substr(key, 1, instr(key, ':') - 1)"
            " ELSE kind END AS cls, count(valid_to) AS changes,"
            " sum(max(coalesce(valid_to, ?) - created_at, 0.0)) AS exposure FROM items GROUP BY cls",
            (now,),
        ).fetchall()
        alpha, beta = self.hazard_prior
        self.hazards = {
            r["cls"]: HazardPosterior(alpha, beta, float(r["changes"]), float(r["exposure"]) / DAY) for r in rows
        }
        self._writes_since_refresh = 0
        return self.hazards

    def _hazard_for(self, key: str | None, kind: str) -> HazardPosterior:
        cls = key.split(":", 1)[0] if key and ":" in key else kind
        posterior = self.hazards.get(cls)
        if posterior is None:
            alpha, beta = self.hazard_prior
            posterior = self.hazards[cls] = HazardPosterior(alpha, beta)
        return posterior

    # retrieve ------------------------------------------------------------------------------

    def lexical_search(self, query: str, limit: int, scope: str | None = None) -> list[tuple[int, float]]:
        terms = content_terms(query) or list(dict.fromkeys(tokenize(query)))
        if not terms:
            return []
        sql = (
            "SELECT items_fts.rowid AS id, -bm25(items_fts) AS score FROM items_fts"
            " JOIN items ON items.id = items_fts.rowid"
            " WHERE items_fts MATCH ? AND items.tier='active'"
        )
        args: list = [fts_query(terms)]
        if scope is not None:
            sql += " AND items.scope IN (?, 'global')"
            args.append(scope)
        sql += " ORDER BY bm25(items_fts) LIMIT ?"
        args.append(limit)
        return [(int(r["id"]), float(r["score"])) for r in self.db.execute(sql, args)]

    def _matrix(self, scope: str | None) -> tuple[np.ndarray, np.ndarray]:
        cached = self._matrix_cache.get((scope,))
        if cached is not None:
            return cached
        sql = "SELECT id, embedding, session FROM items WHERE tier='active' AND embedding IS NOT NULL"
        args: list = []
        if scope is not None:
            sql += " AND scope IN (?, 'global')"
            args.append(scope)
        rows = self.db.execute(sql + " ORDER BY id", args).fetchall()
        codes: dict[str | None, int] = {}
        self._session_codes[(scope,)] = np.fromiter(
            (codes.setdefault(r["session"], len(codes)) if r["session"] is not None else -1 for r in rows),
            dtype=np.int64, count=len(rows),
        )
        if not rows:
            result = (np.zeros(0, dtype=np.int64), np.zeros((0, 1), dtype=np.float32))
        else:
            ids = np.fromiter((r["id"] for r in rows), dtype=np.int64, count=len(rows))
            matrix = np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32)
            result = (ids, matrix.reshape(len(rows), -1))
        self._matrix_cache[(scope,)] = result
        return result

    def dense_search(
        self, query: str, limit: int, scope: str | None = None, query_vector: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        if query_vector is None:
            if self.embedder is None:
                return []
            query_vector = self.embedder.encode([query])[0]
        ids, matrix = self._matrix(scope)
        if ids.size == 0:
            return []
        scores = matrix @ query_vector
        k = min(limit, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(ids[i]), float(scores[i])) for i in top]

    def context_signals(
        self, ids: Sequence[int], query_vector: np.ndarray | None, scope: str | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dense similarity of each item's neighbours (one and two turns away, same session) and its session's best.

        Items are neighbours when they are adjacent in write order and share a session. Without embeddings or a
        session, the signals are zero.
        """
        n = len(ids)
        zeros = (np.zeros(n), np.zeros(n), np.zeros(n))
        if query_vector is None or n == 0:
            return zeros
        all_ids, matrix = self._matrix(scope)
        if all_ids.size == 0:
            return zeros
        codes = self._session_codes[(scope,)]
        scores = np.asarray(matrix @ query_vector, dtype=float)
        best = np.full(int(codes.max()) + 2, -1.0)
        has = codes >= 0
        np.maximum.at(best, codes[has], scores[has])
        pos = np.searchsorted(all_ids, np.asarray(ids, dtype=np.int64))
        pos = np.clip(pos, 0, all_ids.size - 1)
        found = all_ids[pos] == np.asarray(ids, dtype=np.int64)
        own = np.where(found, codes[pos], -1)

        def neighbour(offset: int) -> np.ndarray:
            out = np.zeros(n)
            for sign in (-1, 1):
                other = np.clip(pos + sign * offset, 0, all_ids.size - 1)
                same = found & (own >= 0) & (codes[other] == own) & (other != pos)
                out = np.maximum(out, np.where(same, scores[other], 0.0))
            return out

        near = neighbour(1)
        near2 = np.maximum(near, neighbour(2))
        session = np.where(found & (own >= 0), best[np.maximum(own, 0)], 0.0)
        return near, near2, session

    def _fetch(self, ids: Sequence[int]) -> dict[int, sqlite3.Row]:
        rows: dict[int, sqlite3.Row] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ",".join("?" * len(chunk))
            for row in self.db.execute(f"SELECT * FROM items WHERE id IN ({marks})", list(chunk)):
                rows[int(row["id"])] = row
        return rows

    def _access_ages(self, ids: Sequence[int], now: float) -> dict[int, list[float]]:
        ages: dict[int, list[float]] = {i: [] for i in ids}
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ",".join("?" * len(chunk))
            for row in self.db.execute(f"SELECT item_id, at FROM accesses WHERE item_id IN ({marks})", list(chunk)):
                ages[int(row["item_id"])].append(max((now - row["at"]) / DAY, 0.0))
        return ages

    def _facets(self, query: str, scope: str | None) -> tuple[list[str], np.ndarray]:
        terms = list(dict.fromkeys(light_stem(t) for t in content_terms(query)))[:16]
        if not terms:
            return [], np.zeros(0)
        total = self.db.execute("SELECT count(*) AS n FROM items WHERE tier='active'").fetchone()["n"] or 1
        weights = []
        for term in terms:
            df = self.db.execute(
                "SELECT count(*) AS n FROM items_fts WHERE items_fts MATCH ?", ('"' + term + '"',)
            ).fetchone()["n"]
            weights.append(math.log(1.0 + (total + 1.0) / (df + 1.0)))
        arr = np.asarray(weights)
        return terms, arr / arr.sum()

    def candidates(
        self,
        query: str,
        now: float,
        scope: str | None,
        cfg: RetrievalConfig,
        query_vector: np.ndarray | None = None,
    ) -> tuple[Candidates, dict[int, sqlite3.Row], np.ndarray | None, list[str], np.ndarray, dict]:
        lexical = self.lexical_search(query, cfg.candidates, scope)
        dense = self.dense_search(query, cfg.candidates, scope, query_vector) if cfg.use_dense else []
        lex = {i: (rank, s) for rank, (i, s) in enumerate(lexical)}
        den = {i: (rank, s) for rank, (i, s) in enumerate(dense)}
        ids = list(dict.fromkeys([i for i, _ in lexical] + [i for i, _ in dense]))
        rows = self._fetch(ids)
        ids = [i for i in ids if i in rows]
        if cfg.neighbour_top > 0 and ids:
            extra = self._neighbours(ids, lex, den, rows, cfg, scope)
            rows.update(extra)
            ids.extend(extra)
        historical = bool(_HISTORICAL.search(query))
        if cfg.use_validity and cfg.drop_superseded and not historical:
            ids = [i for i in ids if rows[i]["valid_to"] is None]
        terms, term_weights = self._facets(query, scope)
        stems = {i: {light_stem(t) for t in tokenize(rows[i]["text"])} for i in ids}
        n = len(ids)
        cover = np.zeros(n)
        if terms:
            for idx, i in enumerate(ids):
                cover[idx] = sum(w for t, w in zip(terms, term_weights) if t in stems[i])
        ages = self._access_ages(ids, now)
        validity = np.ones(n)
        activation = np.zeros(n)
        usefulness = np.full(n, 0.5)
        for idx, i in enumerate(ids):
            row = rows[i]
            activation[idx] = activation_score(ages[i])
            usefulness[idx] = row["used"] / (row["used"] + row["unused"])
            if cfg.use_validity:
                if row["valid_to"] is not None:
                    validity[idx] = 1.0 if historical else cfg.stale_factor
                else:
                    age = (now - row["confirmed_at"]) / DAY
                    validity[idx] = self._hazard_for(row["key"], row["kind"]).survival(age)
        cand = Candidates(
            ids=ids,
            lex_rank=np.array([lex[i][0] if i in lex else np.inf for i in ids], dtype=float),
            dense_rank=np.array([den[i][0] if i in den else np.inf for i in ids], dtype=float),
            lex_score=np.array([lex[i][1] if i in lex else 0.0 for i in ids], dtype=float),
            dense_score=np.array([den[i][1] if i in den else 0.0 for i in ids], dtype=float),
            term_cover=cover,
            activation=activation,
            usefulness=usefulness,
            validity=validity,
        )
        if cfg.use_reranker and self.reranker is not None and n:
            self._rerank(query, cand, rows, cfg)
        if cfg.relevance == "trees" and n:
            near, near2, session = self.context_signals(ids, query_vector if cfg.use_dense else None, scope)
            views = [ItemView(rows[i]["text"], int(rows[i]["tokens"]), float(rows[i]["created_at"]), rows[i]["session"])
                     for i in ids]
            cand.relevance = relevance_features(
                query, views, cand.lex_rank, cand.dense_rank, cand.lex_score, cand.dense_score, cand.term_cover,
                cand.activation, near, near2, session, cand.rerank_rank, cand.rerank_score, cfg.rank_kappa,
            )
        embedded = [rows[i]["embedding"] for i in ids]
        vectors = None
        if n and all(e is not None for e in embedded):
            vectors = np.frombuffer(b"".join(embedded), dtype=np.float32).reshape(n, -1)
        if cfg.use_validity and cfg.use_supersession and n > 1 and not historical:
            twins = vectors if vectors is not None else HashingEmbedder().encode([rows[i]["text"] for i in ids])
            keep = self._soft_supersession(cand, rows, twins, cfg)
            if not keep.all():
                cand = subset(cand, keep)
                vectors = None if vectors is None else vectors[keep]
        return cand, rows, vectors, terms, term_weights, stems

    def _neighbours(
        self, ids: list[int], lex: dict, den: dict, rows: dict, cfg: RetrievalConfig, scope: str | None
    ) -> dict[int, sqlite3.Row]:
        """Rows written next to the best fused candidates in the same session, which are not candidates yet."""
        def fused(i: int) -> float:
            return sum(1.0 / (61.0 + ranks[i][0]) for ranks in (lex, den) if i in ranks)

        top = sorted(ids, key=fused, reverse=True)[: cfg.neighbour_top]
        wanted = {i + step * sign for i in top for step in range(1, cfg.neighbour_reach + 1) for sign in (-1, 1)}
        wanted -= set(ids)
        found = self._fetch(sorted(wanted))
        origin = {i: rows[i]["session"] for i in top}
        keep: dict[int, sqlite3.Row] = {}
        for i, row in found.items():
            near = [j for j in top if abs(j - i) <= cfg.neighbour_reach]
            in_scope = scope is None or row["scope"] in (scope, "global")
            if in_scope and row["tier"] == "active" and row["session"] is not None and any(origin[j] == row["session"] for j in near):
                keep[i] = row
        return keep

    def _rerank(self, query: str, cand: Candidates, rows, cfg: RetrievalConfig) -> None:
        """Cross-encoder signals for the top of the fused ranking; other candidates stay unscored."""
        fused = 1.0 / (60.0 + cand.lex_rank + 1.0) + 1.0 / (60.0 + cand.dense_rank + 1.0)
        pool = np.argsort(-fused, kind="stable")[: cfg.rerank_pool]
        logits = np.asarray(self.reranker(query, [rows[cand.ids[i]]["text"] for i in pool]), dtype=float)
        known = np.isfinite(logits)
        order = pool[known][np.argsort(-logits[known], kind="stable")]
        n = len(cand.ids)
        cand.rerank_rank = np.full(n, np.inf)
        cand.rerank_score = np.zeros(n)
        cand.rerank_rank[order] = np.arange(order.size)
        cand.rerank_score[pool[known]] = 1.0 / (1.0 + np.exp(-logits[known]))

    def _soft_supersession(self, cand: Candidates, rows, vectors: np.ndarray, cfg: RetrievalConfig) -> np.ndarray:
        """Discount older near-duplicates of newer items, and return the mask of items to keep.

        This is a heuristic for text without keys. Keyed items have exact version chains and are exempt.
        """
        sim = vectors @ vectors.T
        np.fill_diagonal(sim, 0.0)
        confirmed = np.array([rows[i]["confirmed_at"] for i in cand.ids])
        newer = confirmed[None, :] > confirmed[:, None]
        tau = cfg.supersession_threshold
        gate = np.clip((sim - tau) / (1.0 - tau), 0.0, 1.0) * newer
        unkeyed = np.array([rows[i]["key"] is None for i in cand.ids], dtype=bool)
        if not cfg.soft_supersede_keyed:
            gate = gate * unkeyed[:, None]
        cand.validity = cand.validity * np.prod(1.0 - cfg.supersession_strength * gate, axis=1)
        return ~(((sim >= cfg.supersession_drop) & newer).any(axis=1) & unkeyed)

    def build_problem(
        self,
        cand: Candidates,
        rows,
        vectors,
        terms: list[str],
        term_weights: np.ndarray,
        stems,
        cfg: RetrievalConfig,
        bandit: LinearBandit | None,
    ) -> tuple[SelectionProblem, np.ndarray, dict[int, str]]:
        """Effective weights w = rho * mu and the structures F needs. Returns problem, x and snippets.

        With the relevance model, mu is its calibrated probability (adjusted online by the bandit) and F is the
        expected number of needed items selected, plus the optional facility-location term.
        """
        trees = cfg.relevance == "trees" and cand.relevance is not None
        if trees:
            prior = self._relevance_model(cfg).predict(cand.relevance)
            x = bandit_features(prior, cand)
        else:
            x = feature_matrix(cand, cfg)
            prior = np.clip(x @ np.asarray(cfg.theta0), 0.0, 1.0)
        if bandit is None:
            mu = prior
        elif trees:
            mu = bandit.score(np.column_stack([np.zeros(len(prior)), x[:, 1:]]))
        else:
            mu = bandit.score(x)
        if cfg.min_weight > 0:
            mu = np.where(mu >= cfg.min_weight, mu, 0.0)
        if cfg.floor > 0 and not trees:
            rank_signal = 0.5 * (x[:, 1] + x[:, 2])
            mu = np.maximum(mu, cfg.floor * rank_signal)
            prior = np.maximum(prior, cfg.floor * rank_signal)
        weights = cand.validity * mu
        prior = cand.validity * prior
        n = len(cand.ids)
        costs = np.zeros(n)
        snippets: dict[int, str] = {}
        for idx, i in enumerate(cand.ids):
            tokens = rows[i]["tokens"]
            if cfg.snippet_threshold and tokens > cfg.snippet_threshold:
                snippets[idx] = query_snippet(rows[i]["text"], terms, cfg.snippet_tokens, self.count_tokens)
                tokens = self.count_tokens(snippets[idx])
            costs[idx] = tokens + cfg.item_overhead
        f = len(terms)
        affinity = np.zeros((n, f + 1))
        affinity[:, 0] = 1.0
        for idx, i in enumerate(cand.ids):
            for t_idx, term in enumerate(terms):
                if term in stems[i]:
                    affinity[idx, t_idx + 1] = 1.0
        facet_weights = np.concatenate([[cfg.query_facet if f else 1.0], (1.0 - cfg.query_facet) * term_weights])
        modular = cfg.modular
        if trees:
            facet_weights = np.zeros_like(facet_weights)
            modular = 1.0
        similarity = None
        if vectors is not None and cfg.gamma > 0 and cfg.use_dense:
            similarity = np.clip(vectors @ vectors.T, 0.0, 1.0)
        problem = SelectionProblem(
            weights=weights,
            costs=costs,
            affinity=affinity,
            facet_weights=facet_weights,
            similarity=similarity,
            prior_weights=prior if similarity is not None else None,
            gamma=cfg.gamma if similarity is not None else 0.0,
            modular=modular,
        )
        if cfg.excerpt_threshold and trees:
            problem, x, snippets = self._add_excerpts(problem, x, snippets, cand, rows, terms, term_weights, cfg)
        return problem, x, snippets

    def _add_excerpts(self, problem, x, snippets, cand, rows, terms, term_weights, cfg: RetrievalConfig):
        """Offer each long item a second version: its best query window, worth p times the query weight it keeps.

        Versions of one item form a group, and selection takes at most one of them. Extra versions are appended
        after the n original candidates, and ``snippets`` maps their index to the excerpt text.
        """
        n = len(cand.ids)
        extra: list[tuple[int, str, float, int]] = []
        for idx, i in enumerate(cand.ids):
            if rows[i]["tokens"] <= cfg.excerpt_threshold or problem.weights[idx] <= 0:
                continue
            text, kept, total = query_window(rows[i]["text"], terms, list(term_weights), cfg.excerpt_tokens, self.count_tokens)
            tokens = self.count_tokens(text)
            share = kept / total if total > 0 else 0.0
            retain = max(share, tokens / rows[i]["tokens"])
            extra.append((idx, text, retain, tokens))
        if not extra:
            return problem, x, snippets
        src = np.array([e[0] for e in extra])
        retain = np.array([e[2] for e in extra])

        def grow(values: np.ndarray | None, add: np.ndarray | None = None) -> np.ndarray | None:
            if values is None:
                return None
            return np.concatenate([values, values[src] if add is None else add])

        problem.weights = grow(problem.weights, problem.weights[src] * retain)
        problem.costs = grow(problem.costs, np.array([e[3] + cfg.item_overhead for e in extra], dtype=float))
        problem.affinity = np.vstack([problem.affinity, problem.affinity[src]])
        problem.prior_weights = grow(problem.prior_weights, None if problem.prior_weights is None else problem.prior_weights[src] * retain)
        if problem.similarity is not None:
            index = np.concatenate([np.arange(n), src])
            problem.similarity = problem.similarity[np.ix_(index, index)]
        problem.groups = np.concatenate([np.arange(n), src])
        x = np.vstack([x, x[src]])
        snippets = dict(snippets)
        for k, (_, text, _, _) in enumerate(extra):
            snippets[n + k] = text
        return problem, x, snippets

    def retrieve(
        self,
        query: str,
        budget: int,
        *,
        scope: str | None = None,
        now: float | None = None,
        config: RetrievalConfig | None = None,
        query_vector: np.ndarray | None = None,
        record: bool = True,
        certificate: bool = False,
        min_weight: float = 0.0,
    ) -> RetrievalResult:
        cfg = config or self.config
        now = time.time() if now is None else now
        prepared = self.prepare(query, now, scope, cfg, query_vector)
        return self.select(prepared, budget, record=record, certificate=certificate, min_weight=min_weight)

    def prepare(
        self,
        query: str,
        now: float,
        scope: str | None = None,
        config: RetrievalConfig | None = None,
        query_vector: np.ndarray | None = None,
    ) -> Prepared:
        """Everything about a query that does not depend on the budget."""
        cfg = config or self.config
        if self._writes_since_refresh >= 64 or not self.hazards:
            self.refresh_hazards(now)
        if query_vector is None and cfg.use_dense and self.embedder is not None:
            query_vector = self.embedder.encode([query])[0]
        cand, rows, vectors, terms, term_weights, stems = self.candidates(query, now, scope, cfg, query_vector)
        bandit = self.bandit if cfg.use_bandit else None
        problem, x, snippets = None, None, {}
        if cand.ids:
            problem, x, snippets = self.build_problem(cand, rows, vectors, terms, term_weights, stems, cfg, bandit)
        return Prepared(query, now, cfg, cand, rows, problem, x, snippets)

    def select(
        self, prepared: Prepared, budget: int, record: bool = True, certificate: bool = False, min_weight: float = 0.0
    ) -> RetrievalResult:
        problem = prepared.problem
        if problem is None:
            return RetrievalResult([], 0, budget)
        cand, rows = prepared.candidates, prepared.rows
        chosen = budgeted_select(problem, budget, prepared.config.cost_exponents)
        chosen = [i for i in chosen if problem.weights[i] >= min_weight]
        chosen.sort(key=lambda i: -problem.weights[i])
        origin = (lambda i: int(problem.groups[i])) if problem.groups is not None else (lambda i: i)
        items = [self._item(rows[cand.ids[origin(i)]], float(problem.weights[i]), prepared.snippets.get(i)) for i in chosen]
        if prepared.config.newest_first:
            order = sorted(range(len(items)), key=lambda k: -items[k].created_at)
            items, chosen = [items[k] for k in order], [chosen[k] for k in order]
        result = RetrievalResult(
            items=items,
            tokens=int(sum(problem.costs[i] for i in chosen)),
            budget=budget,
            objective=value(problem, chosen),
        )
        if certificate:
            result.upper_bound = optimality_certificate(problem, chosen, budget)
        if record and chosen:
            ids = [cand.ids[origin(i)] for i in chosen]
            result.retrieval_id = self._record(prepared.query, prepared.now, ids, prepared.features[chosen])
        return result

    def _item(self, row: sqlite3.Row, weight: float = 0.0, snippet: str | None = None) -> Item:
        return Item(
            id=int(row["id"]),
            text=row["text"] if snippet is None else snippet,
            kind=row["kind"],
            scope=row["scope"],
            tokens=int(row["tokens"]),
            created_at=float(row["created_at"]),
            key=row["key"],
            superseded=row["valid_to"] is not None,
            weight=weight,
            truncated=snippet is not None,
        )

    def _record(self, query: str, now: float, ids: list[int], x: np.ndarray) -> int:
        cursor = self.db.execute(
            "INSERT INTO retrievals (at, query, item_ids, features) VALUES (?,?,?,?)",
            (now, query, json.dumps(ids), x.astype(np.float64).tobytes()),
        )
        self.db.executemany("INSERT INTO accesses VALUES (?, ?)", [(i, now) for i in ids])
        return int(cursor.lastrowid)

    # feedback ------------------------------------------------------------------------------

    def feedback(self, retrieval_id: int, rewards: dict[int, float]) -> int:
        """Apply per-item rewards in [0, 1] for a logged retrieval. Items not named count as 0.

        An item superseded by now says nothing about usefulness given validity, so the policy skips it.
        """
        row = self.db.execute("SELECT * FROM retrievals WHERE id=?", (retrieval_id,)).fetchone()
        if row is None or row["resolved"]:
            return 0
        ids = json.loads(row["item_ids"])
        x = np.frombuffer(row["features"], dtype=np.float64).reshape(len(ids), -1)
        y = np.array([float(rewards.get(i, 0.0)) for i in ids])
        live = np.array([self._live(i) for i in ids], dtype=bool)
        if self.config.learn:
            self.bandit.update(x[live], y[live])
        for item_id, reward in zip(ids, y):
            self.db.execute(
                "UPDATE items SET used = used + ?, unused = unused + ? WHERE id=?",
                (float(reward), 1.0 - float(reward), item_id),
            )
            if reward >= 0.5:  # use in a task counts as a fresh confirmation that the fact holds
                self.db.execute(
                    "UPDATE items SET confirmed_at = max(confirmed_at, ?) WHERE id=?", (row["at"], item_id)
                )
        self.db.execute("UPDATE retrievals SET resolved=1 WHERE id=?", (retrieval_id,))
        self._save_state()
        return len(ids)

    def _live(self, item_id: int) -> bool:
        row = self.db.execute("SELECT valid_to FROM items WHERE id=?", (item_id,)).fetchone()
        return row is not None and row["valid_to"] is None

    def _relevance_model(self, cfg: RetrievalConfig):
        name = cfg.model
        if not name:
            static = type(self.embedder).__name__ == "Model2VecEmbedder"
            name = "rerank" if cfg.use_reranker and self.reranker is not None else (
                ("static" if static else "dense") if cfg.use_dense else "lexical")
        model = load_model(name)
        if model is None:
            raise RuntimeError(f"relevance model {name!r} is missing from kanerva/relevance_model.json")
        return model

    def _load_bandit(self) -> LinearBandit:
        theta0 = BANDIT_THETA if self.config.relevance == "trees" else self.config.theta0
        row = self.db.execute("SELECT value FROM meta WHERE key='bandit'").fetchone()
        if row is not None:
            bandit = LinearBandit.from_state(json.loads(row["value"]))
            if bandit.theta0.shape[0] == len(theta0):
                return bandit
        return LinearBandit(np.asarray(theta0, dtype=float), self.config.bandit_ridge, self.config.bandit_beta)

    def _save_state(self) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('bandit', ?)", (json.dumps(self.bandit.state()),))
        self.db.commit()

    # inspection ----------------------------------------------------------------------------

    def get(self, item_id: int) -> Item | None:
        row = self.db.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        return None if row is None else self._item(row)

    def edit(self, item_id: int, text: str) -> bool:
        text, _ = redact(text.strip())
        row = self.db.execute("SELECT id FROM items WHERE id=?", (item_id,)).fetchone()
        if row is None or not text:
            return False
        blob = None
        if self.embedder is not None:
            blob = self.embedder.encode([text])[0].astype(np.float32).tobytes()
        digest = hashlib.sha1(" ".join(text.lower().split()).encode()).hexdigest()
        self.db.execute(
            "UPDATE items SET text=?, digest=?, tokens=?, embedding=? WHERE id=?",
            (text, digest, self.count_tokens(text), blob, item_id),
        )
        self.db.execute("DELETE FROM items_fts WHERE rowid=?", (item_id,))
        self.db.execute("INSERT INTO items_fts (rowid, body) VALUES (?, ?)", (item_id, search_text(text)))
        self.db.commit()
        self._matrix_cache.clear()
        return True

    def list_items(self, scope: str | None = None, tier: str = "active", limit: int = 100) -> list[Item]:
        sql = "SELECT * FROM items WHERE tier=?"
        args: list = [tier]
        if scope is not None:
            sql += " AND scope=?"
            args.append(scope)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        return [self._item(r) for r in self.db.execute(sql, args)]

    def stats(self) -> dict:
        row = self.db.execute(
            "SELECT count(*) AS n, coalesce(sum(tokens), 0) AS t FROM items WHERE tier='active'"
        ).fetchone()
        archived = self.db.execute("SELECT count(*) AS n FROM items WHERE tier='archive'").fetchone()
        stale = self.db.execute("SELECT count(*) AS n FROM items WHERE valid_to IS NOT NULL").fetchone()
        return {"active": row["n"], "active_tokens": row["t"], "archived": archived["n"], "superseded": stale["n"]}

    def delete(self, item_ids: Iterable[int]) -> int:
        """Permanently remove items. Use ``archive`` for reversible forgetting."""
        removed = 0
        for item_id in item_ids:
            removed += self.db.execute("DELETE FROM items WHERE id=?", (item_id,)).rowcount
            self.db.execute("DELETE FROM items_fts WHERE rowid=?", (item_id,))
            self.db.execute("DELETE FROM accesses WHERE item_id=?", (item_id,))
        self.db.commit()
        self._matrix_cache.clear()
        return removed

    def archive(self, item_ids: Iterable[int]) -> int:
        count = 0
        for item_id in item_ids:
            count += self.db.execute("UPDATE items SET tier='archive' WHERE id=?", (item_id,)).rowcount
        self.db.commit()
        self._matrix_cache.clear()
        return count

    def commit(self) -> None:
        self.db.commit()


def open_store(path: str | Path | None = None) -> Memory:
    """The user's store. ``KANERVA_EMBEDDER=static`` adds the model2vec embedder, falling back to lexical search."""
    from .paths import db_path

    embedder = None
    if os.environ.get("KANERVA_EMBEDDER") == "static":
        try:
            from .embed import Model2VecEmbedder

            embedder = Model2VecEmbedder()
        except Exception:  # missing package or no network on first use: lexical search still works
            embedder = None
    return Memory(path or db_path(), embedder=embedder)
