"""SQLite-backed memory: write, retrieve, feedback, consolidate, forget."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .bandit import LinearBandit
from .embed import Embedder
from .policy import Candidates, RetrievalConfig, feature_matrix
from .secrets_filter import redact
from .select import SelectionProblem, budgeted_select, optimality_certificate, value
from .text import approx_tokens, content_terms, fts_query, light_stem, search_text, tokenize
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
    embedding BLOB
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

    def render(self) -> str:
        date = time.strftime("%Y-%m-%d", time.gmtime(self.created_at))
        mark = " superseded" if self.superseded else ""
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


class Memory:
    """A memory store. ``path`` may be ``":memory:"``; ``embedder`` is optional."""

    def __init__(
        self,
        path: str | Path = ":memory:",
        embedder: Embedder | None = None,
        count_tokens: Callable[[str], int] = approx_tokens,
        config: RetrievalConfig | None = None,
        hazard_prior: tuple[float, float] = (1.0, 365.0),
    ):
        self.config = config or RetrievalConfig()
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
        self.hazard_prior = hazard_prior
        self.hazards: dict[str, HazardPosterior] = {}
        self.bandit = self._load_bandit()
        self._matrix_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
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
        if secret_kinds:
            src["redacted"] = secret_kinds
        cursor = self.db.execute(
            "INSERT INTO items (scope, kind, text, key, digest, tokens, created_at, confirmed_at,"
            " parents, source, embedding, pinned) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (scope, kind, text, key, digest, self.count_tokens(text), now, now,
             json.dumps(list(parents)), json.dumps(src), blob, int(pinned)),
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
        sql = "SELECT id, embedding FROM items WHERE tier='active' AND embedding IS NOT NULL"
        args: list = []
        if scope is not None:
            sql += " AND scope IN (?, 'global')"
            args.append(scope)
        rows = self.db.execute(sql, args).fetchall()
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
        dense = self.dense_search(query, cfg.candidates, scope, query_vector)
        lex = {i: (rank, s) for rank, (i, s) in enumerate(lexical)}
        den = {i: (rank, s) for rank, (i, s) in enumerate(dense)}
        ids = list(dict.fromkeys([i for i, _ in lexical] + [i for i, _ in dense]))
        rows = self._fetch(ids)
        ids = [i for i in ids if i in rows]
        terms, term_weights = self._facets(query, scope)
        stems = {i: {light_stem(t) for t in tokenize(rows[i]["text"])} for i in ids}
        n = len(ids)
        cover = np.zeros(n)
        if terms:
            for idx, i in enumerate(ids):
                cover[idx] = sum(w for t, w in zip(terms, term_weights) if t in stems[i])
        ages = self._access_ages(ids, now)
        historical = bool(_HISTORICAL.search(query))
        validity = np.ones(n)
        activation = np.zeros(n)
        usefulness = np.full(n, 0.5)
        for idx, i in enumerate(ids):
            row = rows[i]
            activation[idx] = activation_score(ages[i])
            usefulness[idx] = row["used"] / (row["used"] + row["unused"])
            if cfg.use_validity:
                if row["valid_to"] is not None and not historical:
                    validity[idx] = cfg.stale_factor
                elif row["valid_to"] is None:
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
        embedded = [rows[i]["embedding"] for i in ids]
        vectors = None
        if n and all(e is not None for e in embedded):
            vectors = np.frombuffer(b"".join(embedded), dtype=np.float32).reshape(n, -1)
        if cfg.use_validity and cfg.use_supersession and vectors is not None and n > 1 and not historical:
            self._soft_supersession(cand, rows, vectors, cfg)
        return cand, rows, vectors, terms, term_weights, stems

    def _soft_supersession(self, cand: Candidates, rows, vectors: np.ndarray, cfg: RetrievalConfig) -> None:
        sim = vectors @ vectors.T
        np.fill_diagonal(sim, 0.0)
        created = np.array([rows[i]["created_at"] for i in cand.ids])
        tau = cfg.supersession_threshold
        gate = np.clip((sim - tau) / (1.0 - tau), 0.0, 1.0) * (created[None, :] > created[:, None])
        cand.validity = cand.validity * np.prod(1.0 - cfg.supersession_strength * gate, axis=1)

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
        count_cost: Callable[[sqlite3.Row], int] | None = None,
    ) -> tuple[SelectionProblem, np.ndarray, np.ndarray]:
        """Weights from the policy and the structures F needs. Returns problem, x, prior weights."""
        x = feature_matrix(cand, cfg)
        prior = np.clip(x @ np.asarray(cfg.theta0), 0.0, 1.0)
        weights = bandit.score(x) if bandit is not None else prior
        if cfg.min_weight > 0:
            weights = np.where(weights >= cfg.min_weight, weights, 0.0)
        if cfg.floor > 0:
            rank_signal = 0.5 * (x[:, 1] + x[:, 2])
            weights = np.maximum(weights, cfg.floor * rank_signal)
            prior = np.maximum(prior, cfg.floor * rank_signal)
        costs = np.array(
            [(count_cost(rows[i]) if count_cost else rows[i]["tokens"]) + cfg.item_overhead for i in cand.ids],
            dtype=float,
        )
        n = len(cand.ids)
        f = len(terms)
        affinity = np.zeros((n, f + 1))
        affinity[:, 0] = 1.0
        for idx, i in enumerate(cand.ids):
            for t_idx, term in enumerate(terms):
                if term in stems[i]:
                    affinity[idx, t_idx + 1] = 1.0
        facet_weights = np.concatenate([[cfg.query_facet if f else 1.0], (1.0 - cfg.query_facet) * term_weights])
        similarity = None
        if vectors is not None and cfg.gamma > 0:
            similarity = np.clip(vectors @ vectors.T, 0.0, 1.0)
        problem = SelectionProblem(
            weights=weights,
            costs=costs,
            affinity=affinity,
            facet_weights=facet_weights,
            similarity=similarity,
            prior_weights=prior if similarity is not None else None,
            gamma=cfg.gamma if similarity is not None else 0.0,
            modular=cfg.modular,
        )
        return problem, x, prior

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
    ) -> RetrievalResult:
        cfg = config or self.config
        now = time.time() if now is None else now
        prepared = self.prepare(query, now, scope, cfg, query_vector)
        return self.select(prepared, budget, record=record, certificate=certificate)

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
        cand, rows, vectors, terms, term_weights, stems = self.candidates(query, now, scope, cfg, query_vector)
        bandit = self.bandit if cfg.use_bandit else None
        problem, x = None, None
        if cand.ids:
            problem, x, _ = self.build_problem(cand, rows, vectors, terms, term_weights, stems, cfg, bandit)
        return Prepared(query, now, cfg, cand, rows, problem, x)

    def select(self, prepared: Prepared, budget: int, record: bool = True, certificate: bool = False) -> RetrievalResult:
        problem = prepared.problem
        if problem is None:
            return RetrievalResult([], 0, budget)
        cand, rows = prepared.candidates, prepared.rows
        chosen = budgeted_select(problem, budget, prepared.config.cost_exponents)
        chosen.sort(key=lambda i: -problem.weights[i])
        items = [self._item(rows[cand.ids[i]], float(problem.weights[i])) for i in chosen]
        result = RetrievalResult(
            items=items,
            tokens=int(sum(problem.costs[i] for i in chosen)),
            budget=budget,
            objective=value(problem, chosen),
        )
        if certificate:
            result.upper_bound = optimality_certificate(problem, chosen, budget)
        if record and chosen:
            ids = [cand.ids[i] for i in chosen]
            result.retrieval_id = self._record(prepared.query, prepared.now, ids, prepared.features[chosen])
        return result

    def _item(self, row: sqlite3.Row, weight: float = 0.0) -> Item:
        return Item(
            id=int(row["id"]),
            text=row["text"],
            kind=row["kind"],
            scope=row["scope"],
            tokens=int(row["tokens"]),
            created_at=float(row["created_at"]),
            key=row["key"],
            superseded=row["valid_to"] is not None,
            weight=weight,
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
        """Apply per-item rewards in [0, 1] for a logged retrieval. Items not named count as 0."""
        row = self.db.execute("SELECT * FROM retrievals WHERE id=?", (retrieval_id,)).fetchone()
        if row is None or row["resolved"]:
            return 0
        ids = json.loads(row["item_ids"])
        x = np.frombuffer(row["features"], dtype=np.float64).reshape(len(ids), -1)
        y = np.array([float(rewards.get(i, 0.0)) for i in ids])
        self.bandit.update(x, y)
        for item_id, reward in zip(ids, y):
            self.db.execute(
                "UPDATE items SET used = used + ?, unused = unused + ? WHERE id=?",
                (float(reward), 1.0 - float(reward), item_id),
            )
        self.db.execute("UPDATE retrievals SET resolved=1 WHERE id=?", (retrieval_id,))
        self._save_state()
        return len(ids)

    def _load_bandit(self) -> LinearBandit:
        row = self.db.execute("SELECT value FROM meta WHERE key='bandit'").fetchone()
        if row is not None:
            return LinearBandit.from_state(json.loads(row["value"]))
        return LinearBandit(np.asarray(self.config.theta0, dtype=float), self.config.bandit_ridge, self.config.bandit_beta)

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
