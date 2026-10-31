"""Benchmark loaders. Each yields Episodes: one query against one memory stream."""

from __future__ import annotations

import calendar
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@dataclass
class Turn:
    text: str
    at: float  # unix seconds
    key: str | None = None
    kind: str = "episodic"
    session: str | None = None


@dataclass
class Query:
    pos: int  # number of writes applied before this query
    qid: str
    question: str
    now: float
    category: str
    gold: set[str] = field(default_factory=set)  # item texts that answer the query
    stale: set[str] = field(default_factory=set)  # item texts with an outdated answer
    historical: bool = False
    answer: str = ""  # reference answer, for answer-term recall; empty when there is none
    files: tuple[str, ...] = ()  # the files a task touches
    previous_files: tuple[str, ...] = ()  # the files of the change just before it


@dataclass
class Stream:
    """A write sequence with queries interleaved; one store per stream."""

    name: str
    split: str
    writes: list[Turn]
    queries: list[Query]


def digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def load_lme_updates() -> list[Stream]:
    """Natural-language staleness: knowledge-update questions with evidence in more than one session.

    The evidence turns of the latest session are gold, and those of earlier sessions hold the outdated fact.
    """
    raw = {q["question_id"]: q for q in json.loads((DATA / "longmemeval" / "longmemeval_s_cleaned.json").read_text())}
    streams = []
    for stream in load_longmemeval():
        q = raw[stream.name]
        if q["question_type"] != "knowledge-update":
            continue
        latest: list[str] = []
        earlier: list[str] = []
        for session in q["haystack_sessions"]:  # sessions are in date order
            flagged = [f"{t['role']}: {t['content']}" for t in session if t.get("has_answer")]
            if flagged:
                earlier.extend(latest)
                latest = flagged
        if earlier:
            query = stream.queries[0]
            query.gold, query.stale = set(latest), set(earlier) - set(latest)
            query.category = "update"
            streams.append(stream)
    return streams


def _lme_time(value: str) -> float:
    dt = datetime.strptime(value.split(" (")[0] + " " + value.split(") ")[1], "%Y/%m/%d %H:%M")
    return dt.replace(tzinfo=timezone.utc).timestamp()


def _split_for(qid: str, dev_fraction: float, salt: str) -> str:
    h = int(hashlib.sha1((salt + qid).encode()).hexdigest(), 16)
    return "dev" if (h % 10000) / 10000 < dev_fraction else "test"


def load_longmemeval(dev_fraction: float = 0.2) -> list[Stream]:
    """Turn-level LongMemEval-S, one store per question; questions with no flagged turn are dropped."""
    raw = json.loads((DATA / "longmemeval" / "longmemeval_s_cleaned.json").read_text())
    streams = []
    for q in raw:
        turns: list[Turn] = []
        gold: set[str] = set()
        for index, (date, session) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
            at = _lme_time(date)
            for offset, turn in enumerate(session):
                text = f"{turn['role']}: {turn['content']}"
                turns.append(Turn(text, at + offset, session=str(index)))
                if turn.get("has_answer"):
                    gold.add(text)
        if not gold:
            continue
        query = Query(
            pos=len(turns),
            qid=q["question_id"],
            question=q["question"],
            now=_lme_time(q["question_date"]),
            category=q["question_type"],
            gold=gold,
            answer=str(q["answer"]),
        )
        streams.append(Stream(q["question_id"], _split_for(q["question_id"], dev_fraction, "lme"), turns, [query]))
    return streams


_LOCOMO_TIME = re.compile(r"(\d+):(\d+) (am|pm) on (\d+) (\w+),? (\d+)", re.I)
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}


def _locomo_time(value: str) -> float:
    m = _LOCOMO_TIME.match(value.strip())
    hour, minute, ampm, day, month, year = m.groups()
    hour = int(hour) % 12 + (12 if ampm.lower() == "pm" else 0)
    dt = datetime(int(year), _MONTHS[month.lower()], int(day), hour, int(minute), tzinfo=timezone.utc)
    return dt.timestamp()


def load_locomo() -> list[Stream]:
    """Turn-level LoCoMo, one store per conversation; conversations 0-2 are dev, the others test."""
    raw = json.loads((DATA / "locomo" / "locomo10.json").read_text())
    streams = []
    for index, sample in enumerate(raw):
        conv = sample["conversation"]
        turns: list[Turn] = []
        by_id: dict[str, str] = {}
        sessions = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1]))
        last = 0.0
        for name in sessions:
            at = _locomo_time(conv[f"{name}_date_time"])
            last = max(last, at)
            for offset, turn in enumerate(conv[name]):
                text = f"{turn['speaker']}: {turn['text']}"
                if turn.get("blip_caption"):
                    text += f" [shared image: {turn['blip_caption']}]"
                by_id[turn["dia_id"]] = text
                turns.append(Turn(text, at + offset, session=name))
        queries = []
        for q_index, qa in enumerate(sample["qa"]):
            gold = {by_id[e.strip()] for e in qa.get("evidence", []) if e.strip() in by_id}
            if gold:
                queries.append(
                    Query(len(turns), f"{sample['sample_id']}-{q_index}", qa["question"], last + 86400, str(qa["category"]), gold,
                          answer=str(qa.get("answer", "")))
                )
        streams.append(Stream(str(sample["sample_id"]), "dev" if index < 3 else "test", turns, queries))
    return streams


BEAM_SPLITS = {"1M": "dev", "100K": "test", "500K": "test"}
BEAM_ANSWER_FIELDS = ("answer", "ideal_answer", "ideal_summary")


def _beam_ids(value) -> list[int]:
    """Flatten ``source_chat_ids``, which is a list or a dict of lists, written as a Python literal."""
    import ast

    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if isinstance(parsed, dict):
        return [i for ids in parsed.values() for i in _beam_ids(ids)]
    if isinstance(parsed, (list, tuple)):
        return [i for ids in parsed for i in _beam_ids(ids)]
    return [int(parsed)]


def _beam_time(anchor: str) -> float | None:
    try:
        return datetime.strptime(anchor.strip(), "%B-%d-%Y").replace(tzinfo=timezone.utc).timestamp()
    except (ValueError, AttributeError):
        return None


def load_beam(splits: tuple[str, ...] = ("1M", "100K", "500K")) -> list[Stream]:
    """BEAM, one store per conversation. Abstention questions have no evidence and are dropped.

    BEAM-1M is development data; BEAM-100K and BEAM-500K are the held-out test.
    """
    import ast

    import pyarrow.parquet as pq

    streams = []
    for split in splits:
        for row in pq.read_table(DATA / "beam" / f"{split}.parquet").to_pylist():
            turns: list[Turn] = []
            by_id: dict[int, str] = {}
            at = _beam_time("January-01-2024") or 0.0
            for s_index, session in enumerate(row["chat"]):
                for message in session:
                    stamp = _beam_time(message.get("time_anchor") or "")
                    at = max(at + 60.0, stamp or 0.0)
                    text = f"{message['role']}: {message['content']}"
                    turns.append(Turn(text, at, session=str(s_index)))
                    by_id[int(message["id"])] = text
            queries = []
            probing = ast.literal_eval(row["probing_questions"])
            for qtype, items in probing.items():
                for index, q in enumerate(items):
                    if "source_chat_ids" not in q:
                        continue
                    gold = {by_id[i] for i in _beam_ids(q["source_chat_ids"]) if i in by_id}
                    if not gold:
                        continue
                    answer = next((str(q[f]) for f in BEAM_ANSWER_FIELDS if q.get(f)), "")
                    queries.append(Query(len(turns), f"beam{split}-{row['conversation_id']}-{qtype}-{index}", q["question"],
                                         at + 86400, qtype, gold, answer=answer))
            streams.append(Stream(f"beam{split}-{row['conversation_id']}", BEAM_SPLITS[split], turns, queries))
    return streams


def all_texts(streams: list[Stream]) -> list[str]:
    seen: dict[str, None] = {}
    for stream in streams:
        for turn in stream.writes:
            seen.setdefault(turn.text)
        for query in stream.queries:
            seen.setdefault(query.question)
    return list(seen)
