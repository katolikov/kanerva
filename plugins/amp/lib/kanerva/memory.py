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
    QUERY_GATES,
    RERANK_THETA,
    Candidates,
    RetrievalConfig,
    bandit_features,
    feature_matrix,
    query_signals,
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
    expected REAL NOT NULL DEFAULT 0,
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

_WORD = re.compile(r"\w+")
COMMON_FILE_NAMES = frozenset(
    """__init__.py __main__.py main.py setup.py conftest.py index.ts index.tsx index.js types.ts utils.ts mod.rs lib.rs
    main.rs main.go doc.go go.mod package.json pyproject.toml Cargo.toml README.md Makefile Dockerfile main.c main.cc
    main.cpp common.h config.h types.h util.h utils.h CMakeLists.txt""".split()
)

_HISTORICAL = re.compile(
    r"\b(previous(ly)?|used to|formerly|originally|initial(ly)?|earlier|before|old|former|prior|"
    r"history|changed from|at first|first time|"
    r"прежн\w+|прежде|предыдущ\w+|раньше|ранее|бывш\w+|изначальн\w+|первоначальн\w+|поначалу|"
    r"стар(ый|ая|ое|ые|ого|ой|ому|ую|ым|ых|ом|ыми)|истори(я|и|ю|ей)|изменил\w* с|поменял\w* с|в первый раз|впервые)\b",
    re.I,
)
SCHEMA_VERSION = 2  # 2: items_fts holds the stems of Russian words

FILE_COMMON = 0.05  # a file this share of the notes names says nothing about any of them
FILE_COMMON_NOTES = 20  # a small store has no such files
SIMILAR_POOL = 50  # candidates taken from lexical search, and four times that from vectors
SIMILAR_STORE = 1000  # word rarity is judged as if the store held at least this many notes
SIMILAR_SPREAD = 2.0  # a note with more than this many times the words of another is not a version of it
SIMILAR_OVERLAP = 0.5  # share of two notes' word weight that they have in common
SIMILAR_SHARED = 3.5  # weight in common, where a word no other note has weighs 1
SIMILAR_COSINE = 0.3  # closeness of two notes' vectors, set low because the list is short and agents read it with care
_PLACEHOLDER = re.compile(r"\[REDACTED:[\w-]+\]")


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


def alternate(first: list, second: list) -> list:
    """The two lists taken in turn, and whatever is left of the longer one."""
    pairs = [entry for pair in zip(first, second) for entry in pair]
    return pairs + first[len(second):] + second[len(first):]


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
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.executescript(_SCHEMA)
        self._add_column("session", "session TEXT")
        self._add_column("expected", "expected REAL NOT NULL DEFAULT 0")
        self._upgrade_index()
        self.hazard_prior = hazard_prior
        self.hazards: dict[str, HazardPosterior] = {}
        self.bandit = self._load_bandit()
        self._matrix_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._session_codes: dict[tuple, np.ndarray] = {}
        self._writes_since_refresh = 0

    def _add_column(self, name: str, definition: str) -> None:
        """Hooks start side by side, so another process can add the column between the check and the change."""
        if any(r["name"] == name for r in self.db.execute("PRAGMA table_info(items)")):
            return
        try:
            self.db.execute(f"ALTER TABLE items ADD COLUMN {definition}")
        except sqlite3.OperationalError as error:
            if "duplicate column" not in str(error):
                raise

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def _upgrade_index(self) -> None:
        """Bring ``items_fts`` up to ``SCHEMA_VERSION``, once. Only notes with Cyrillic text index differently."""
        current = "SELECT value FROM meta WHERE key='schema_version' AND CAST(value AS INTEGER) >= ?"
        if self.db.execute(current, (SCHEMA_VERSION,)).fetchone() is not None:
            return
        try:
            self.db.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError:  # another process holds the write lock, so the next open tries again
            return
        if self.db.execute(current, (SCHEMA_VERSION,)).fetchone() is None:  # a process that was first has done it
            for row in self.db.execute("SELECT id, text FROM items WHERE text GLOB '*[а-яёА-ЯЁ]*'").fetchall():
                self.db.execute("DELETE FROM items_fts WHERE rowid=?", (row["id"],))
                self.db.execute("INSERT INTO items_fts (rowid, body) VALUES (?, ?)", (row["id"], search_text(row["text"])))
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        self.db.commit()

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

    def similar(
        self,
        text: str,
        scope: str,
        limit: int = 5,
        *,
        exclude: Iterable[int] = (),
        query_vector: np.ndarray | None = None,
        overlap: float = SIMILAR_OVERLAP,
        shared: float = SIMILAR_SHARED,
        cosine: float = SIMILAR_COSINE,
    ) -> list[Item]:
        """Live facts of ``scope`` most likely about the same subject as ``text``, closest first.

        A fact qualifies when most of the word weight of the two notes is common to both, when they have enough rare
        words in common, or when their vectors are close. This only lists candidates: the host agent decides, and
        ``supersede`` records its answer. Global notes count too, so a personal fact finds its older version from any project.
        """
        near = self._near_facts(text, scope, exclude, query_vector)
        worded = [(row, common) for row, common, weight, _ in near if common >= overlap or weight >= shared]
        known = {row["id"] for row, _ in worded}
        phrased = sorted((entry for entry in near if entry[3] >= cosine and entry[0]["id"] not in known), key=lambda entry: -entry[3])
        found = alternate(worded, [(row, common) for row, common, _, _ in phrased])  # neither kind may crowd out the other
        items = [self._item(row, common) for row, common in found[:limit]]
        for item in items:
            item.text = redact(item.text)[0]  # a note can predate a pattern of the filter
        return items

    @staticmethod
    def _subject_words(text: str) -> dict[str, str]:
        """Stems of a note, each with a word that has it. What the secret filter left behind says nothing about the subject."""
        return {light_stem(t): t for t in content_terms(_PLACEHOLDER.sub(" ", text))}

    def _near_facts(
        self, text: str, scope: str, exclude: Iterable[int] = (), query_vector: np.ndarray | None = None
    ) -> list[tuple[sqlite3.Row, float, float, float]]:
        """Live facts found for ``text``, each with its share of word weight, weight and cosine in common with it."""
        text = redact(text.strip())[0]
        words = self._subject_words(text)
        if not words:
            return []
        close = dict(self.dense_search(text, 4 * SIMILAR_POOL, scope, query_vector))
        lexical = self.db.execute(
            "SELECT items_fts.rowid AS id FROM items_fts JOIN items ON items.id = items_fts.rowid"
            " WHERE items_fts MATCH ? AND items.scope IN (?, 'global') AND items.tier='active' AND items.kind<>'episodic'"
            " AND items.valid_to IS NULL ORDER BY bm25(items_fts) LIMIT ?",
            (fts_query(words.values()), scope, SIMILAR_POOL),
        ).fetchall()
        rows = self._fetch(list(dict.fromkeys([r["id"] for r in lexical] + list(close))))
        skip = set(exclude)
        total = self.db.execute("SELECT count(*) AS n FROM items WHERE tier='active'").fetchone()["n"]
        total = max(total, SIMILAR_STORE)
        weights: dict[str, float] = {}

        def weight(term: str) -> float:
            if term not in weights:
                df = self.db.execute(
                    "SELECT count(*) AS n FROM items_fts WHERE items_fts MATCH ?", (fts_query([term]),)
                ).fetchone()["n"]
                weights[term] = math.log(1.0 + (total + 1.0) / (df + 1.0)) / math.log(2.0 + total)
            return weights[term]

        found = []
        for i, row in rows.items():
            fact = row["scope"] in (scope, "global") and row["tier"] == "active" and row["kind"] != "episodic" and row["valid_to"] is None
            if not fact or i in skip:
                continue
            have = self._subject_words(row["text"])
            closeness = close.get(i, 0.0)
            if closeness < SIMILAR_COSINE and max(len(have), len(words)) > SIMILAR_SPREAD * min(len(have), len(words)):
                continue
            both = sum(weight(words[s]) for s in have.keys() & words.keys())
            union = sum(weight(have.get(s) or words[s]) for s in have.keys() | words.keys())
            found.append((row, both / union, both, closeness))
        found.sort(key=lambda entry: -entry[1])
        return found

    def supersede(self, old: int, new: int, at: float | None = None) -> bool:
        """Close ``old`` with ``new`` the way a key chain does, and say whether anything changed.

        ``old`` stays in the store as history and ``new`` takes over its key, so a later write with that key still
        finds the chain. Both must be live active items of one scope, or ``new`` a global note, and ``new`` must not be
        the older one, which keeps swapped arguments from closing the newer note.
        """
        rows = self._fetch([old, new])
        if old == new or len(rows) < 2:
            return False
        before, after = rows[old], rows[new]
        live = all(r["tier"] == "active" and r["valid_to"] is None for r in (before, after))
        one_scope = before["scope"] == after["scope"] or after["scope"] == "global"
        if not live or not one_scope or after["confirmed_at"] < before["created_at"]:
            return False
        now = time.time() if at is None else at
        self.db.execute("UPDATE items SET valid_to=?, superseded_by=? WHERE id=?", (now, new, old))
        if after["key"] is None and before["key"] is not None:
            self.db.execute("UPDATE items SET key=? WHERE id=?", (before["key"], new))
        self.db.commit()
        return True

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
        if self.embedder is not None:  # a store can hold vectors from an embedder used earlier
            sql += " AND length(embedding) = ?"
            args.append(4 * self.embedder.dim)
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
        lift = np.ones(n)
        for idx, i in enumerate(ids):
            row = rows[i]
            activation[idx] = activation_score(ages[i])
            usefulness[idx] = row["used"] / (row["used"] + row["unused"])
            lift[idx] = (row["used"] - 1.0 + cfg.lift_prior) / (row["expected"] + cfg.lift_prior)
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
            lift=lift,
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
        sizes = {None if e is None else len(e) for e in embedded}
        if len(sizes) == 1 and None not in sizes and (self.embedder is None or sizes == {4 * self.embedder.dim}):
            vectors = np.frombuffer(b"".join(embedded), dtype=np.float32).reshape(n, -1)
        if cfg.use_validity and cfg.use_supersession and n > 1 and not historical:
            hashed = vectors is None or cfg.hash_twins
            twins = HashingEmbedder().encode([rows[i]["text"] for i in cand.ids]) if hashed else vectors
            whole = self._whole_words(query, cand, rows) if cfg.twin_exact else None
            keep = self._soft_supersession(cand, rows, twins, cfg, cfg.hashed_drop if hashed else cfg.supersession_drop, whole)
            if cfg.word_twins and not hashed:
                words = HashingEmbedder().encode([rows[i]["text"] for i in cand.ids])
                keep = keep & self._soft_supersession(cand, rows, words, cfg, cfg.hashed_drop, whole)
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

    @staticmethod
    def _note_words(row) -> set[str]:
        return {word.lower() for word in _WORD.findall(row["text"])}

    @staticmethod
    def _whole_words(query: str, cand: Candidates, rows) -> np.ndarray:
        """How many of the query's words each candidate holds whole, with case and identifiers as written."""
        wanted = {word.lower() for word in _WORD.findall(query)}
        return np.array([len(wanted & {word.lower() for word in _WORD.findall(rows[i]["text"])}) for i in cand.ids], dtype=float)

    def _soft_supersession(
        self, cand: Candidates, rows, vectors: np.ndarray, cfg: RetrievalConfig, drop: float, whole: np.ndarray | None = None
    ) -> np.ndarray:
        """Discount older near-duplicates of newer items, and return the mask of items to keep.

        This is a heuristic for facts without keys. Keyed items have exact version chains and are exempt, and
        episodic items are exempt because two similar events are still two events. A newer twin stands in for an older
        item only if it covers the query at least as well: facts about two neighbouring functions read alike, but a
        question names one of them.
        """
        sim = vectors @ vectors.T
        np.fill_diagonal(sim, 0.0)
        confirmed = np.array([rows[i]["confirmed_at"] for i in cand.ids])
        newer = confirmed[None, :] > confirmed[:, None]
        if cfg.twin_needs_query:
            newer = newer & (cand.term_cover[None, :] >= cand.term_cover[:, None] - 1e-9)
            if whole is not None:
                newer = newer & (whole[None, :] >= whole[:, None])
        if not cfg.supersede_events:
            facts = np.array([rows[i]["kind"] != "episodic" for i in cand.ids], dtype=bool)
            newer = newer & facts[:, None] & facts[None, :]
        tau = cfg.supersession_threshold
        gate = np.clip((sim - tau) / (1.0 - tau), 0.0, 1.0) * newer
        unkeyed = np.array([rows[i]["key"] is None for i in cand.ids], dtype=bool)
        if not cfg.soft_supersede_keyed:
            gate = gate * unkeyed[:, None]
        cand.validity = cand.validity * np.prod(1.0 - cfg.supersession_strength * gate, axis=1)
        replaces = (sim >= drop) & newer
        if whole is not None:
            for old, new in zip(*np.nonzero(replaces & unkeyed[:, None])):
                before, after = self._note_words(rows[cand.ids[old]]), self._note_words(rows[cand.ids[new]])
                if after < before and len(before - after) > 1:  # it only says less, though one dropped word can be a change
                    replaces[old, new] = False
        return ~(replaces.any(axis=1) & unkeyed)

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
        if trees and cfg.item_lift:
            mu = np.clip(mu * cand.lift, 0.0, 1.0)
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
        gated: bool = False,
    ) -> RetrievalResult:
        cfg = config or self.config
        now = time.time() if now is None else now
        prepared = self.prepare(query, now, scope, cfg, query_vector)
        return self.select(prepared, budget, record=record, certificate=certificate, min_weight=min_weight, gated=gated)

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

    def has_answer(self, prepared: Prepared) -> bool:
        """Whether the store looks able to answer this query at all, judging by its result list."""
        cand = prepared.candidates
        vectors = prepared.config.use_dense and bool(cand.ids) and float(cand.dense_score.max()) > 0  # notes can lack them
        gate = QUERY_GATES["full" if vectors else "lexical"]
        weights = None if prepared.problem is None else prepared.problem.weights
        signals = query_signals(cand, weights, len(content_terms(prepared.query)))
        return bool(gate.score(signals[: len(gate.use)]) >= gate.threshold)

    def select(
        self,
        prepared: Prepared,
        budget: int,
        record: bool = True,
        certificate: bool = False,
        min_weight: float = 0.0,
        gated: bool = False,
    ) -> RetrievalResult:
        problem = prepared.problem
        if problem is None or (gated and not self.has_answer(prepared)):
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
            (now, redact(query)[0], json.dumps(ids), x.astype(np.float64).tobytes()),
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
        expected = x[:, 1] if self.config.relevance == "trees" else np.zeros(len(ids))
        for item_id, reward, chance in zip(ids, y, expected):
            self.db.execute(
                "UPDATE items SET used = used + ?, unused = unused + ?, expected = expected + ? WHERE id=?",
                (float(reward), 1.0 - float(reward), float(chance), item_id),
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
        """Replace an item's text. The earlier text stays as an archived parent, which ``expand`` shows."""
        text, _ = redact(text.strip())
        row = self.db.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        if row is None or not text:
            return False
        columns = ", ".join(c for c in row.keys() if c != "id")
        earlier = self.db.execute(f"INSERT INTO items ({columns}) SELECT {columns} FROM items WHERE id=?", (item_id,)).lastrowid
        self.db.execute("UPDATE items SET tier='archive', key=NULL, valid_to=? WHERE id=?", (time.time(), earlier))
        parents = json.dumps(json.loads(row["parents"]) + [earlier])
        self.db.execute("UPDATE items SET parents=? WHERE id=?", (parents, item_id))
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

    def reembed(self, batch_size: int = 64, limit: int = -1) -> int:
        """Give up to ``limit`` items without a vector from the current embedder one, and return how many changed."""
        if self.embedder is None:
            return 0
        size = 4 * self.embedder.dim
        rows = self.db.execute(
            "SELECT id, text FROM items WHERE embedding IS NULL OR length(embedding) <> ? ORDER BY id DESC LIMIT ?", (size, limit)
        ).fetchall()
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            vectors = self.embedder.encode([row["text"] for row in chunk])
            self.db.executemany(
                "UPDATE items SET embedding = ? WHERE id = ?",
                [(vector.astype(np.float32).tobytes(), row["id"]) for vector, row in zip(vectors, chunk)],
            )
            self.db.commit()
        self._matrix_cache.clear()
        return len(rows)

    def about_file(self, path: str, budget: int, scope: str | None = None, skip: Iterable[int] = ()) -> list[Item]:
        """Live notes that name this file, newest first, within ``budget`` tokens.

        ``path`` is relative to the project. A name that many projects share counts only together with its folder, and
        the notes the edit hook writes by itself say nothing about the file. Neither does a file that most notes name.
        """
        name = path.rsplit("/", 1)[-1]
        target = path.rsplit("/", 2)[-2] + "/" + name if name in COMMON_FILE_NAMES and "/" in path else name
        terms = list(dict.fromkeys(_WORD.findall(target.lower())))
        if not terms:
            return []
        sql = (
            "SELECT items.* FROM items_fts JOIN items ON items.id = items_fts.rowid"
            " WHERE items_fts MATCH ? AND items.tier='active' AND items.valid_to IS NULL"
        )
        args: list = [" AND ".join('"' + t + '"' for t in terms)]
        if scope is not None:
            sql += " AND items.scope IN (?, 'global')"
            args.append(scope)
        sql += " ORDER BY items.created_at DESC, items.id DESC LIMIT 200"
        pattern = re.compile(r"(?<![\w-])" + re.escape(target) + r"(?![\w-])")
        skip = set(skip)
        found = [row for row in self.db.execute(sql, args)
                 if row["id"] not in skip and not row["text"].startswith("Edited ") and pattern.search(row["text"])]
        total = self.db.execute("SELECT count(*) AS n FROM items WHERE tier='active'").fetchone()["n"]
        if len(found) >= FILE_COMMON_NOTES and len(found) >= FILE_COMMON * total:
            return []
        items, spent = [], 0
        for row in found:
            cost = row["tokens"] + self.config.item_overhead
            if spent + cost <= budget:
                items.append(self._item(row))
                spent += cost
        return items

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


def _user_embedder() -> Embedder | None:
    """The embedder ``KANERVA_EMBEDDER`` names, MiniLM by default and none for ``off``.

    Lexical search takes over whenever one cannot start, such as until the model has arrived.
    """
    name = os.environ.get("KANERVA_EMBEDDER") or "minilm"
    try:
        if name == "static":
            from .embed import Model2VecEmbedder

            return Model2VecEmbedder()
        if name == "minilm":
            from .embed import OnnxEmbedder, fetch_in_background, minilm_ready

            if minilm_ready():
                return OnnxEmbedder()
            fetch_in_background()
        if name == "multilingual":
            from .embed import MultilingualEmbedder, fetch_in_background, multilingual_ready

            if multilingual_ready():
                return MultilingualEmbedder()
            fetch_in_background()
    except Exception:  # a missing package or no network on first use
        return None
    return None


def open_store(path: str | Path | None = None, embed: bool = True) -> Memory:
    """The user's store, with the embedder ``KANERVA_EMBEDDER`` asks for, or lexical search.

    ``embed=False`` is for work that reads no vectors and writes no notes, and skips the cost of loading a model.
    """
    from .paths import db_path

    return Memory(path or db_path(), embedder=_user_embedder() if embed else None)
