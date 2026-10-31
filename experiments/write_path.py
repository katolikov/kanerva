"""The write path without keys: does ``Memory.similar`` show the older version of a changed note?

Streams are replayed with no keys, and ``similar`` is asked about each note as it is written. No LLM takes part. The
thresholds are picked on the dev streams, and every reported number comes from streams they were not picked on:
repomem-new, repomem-fresh, repomem-final and the test part of lme-updates.

The upper bound is simulated: an oracle stands in for the host agent and marks the true older version whenever
``similar`` lists it. Conversation turns are replayed as facts, since ``similar`` skips events and an agent's notes are
facts. Writes results/write_path.json.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Iterator

import numpy as np

from experiments.datasets import Stream, Turn
from experiments.embed_cache import TextCache
from experiments.harness import RESULTS, evaluate_query, git_commit, new_memory
from experiments.methods import kanerva_variant
from experiments.registry import load
from kanerva import Memory, RetrievalConfig, memory
from kanerva.mcp_server import short_note
from kanerva.policy import LEXICAL_THETA
from kanerva.text import approx_tokens

SCOPE = "global"  # what ``Memory.write`` uses by default
BUDGET = 1024
TALK = {"lme-updates"}
PICK_ON = {"repomem": ("repomem", "dev"), "lme-updates": ("lme", "dev")}
REPORT_ON = {"lme-updates": ("lme", "test"), "repomem-new": ("repomem-new", "new"), "repomem-fresh": ("repomem-fresh", "fresh"),
             "repomem-final": ("repomem-final", "final")}
SPREADS = (1.5, 2.0, 3.0)
OVERLAPS = (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
SHAREDS = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
COSINES = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)  # reported, not picked: experiments/agent_write_path.py sets this threshold
SHORT_UPDATES = ("The API timeout is {} seconds", "Deploys go through the {} branch")  # the plainest update, in neither dataset
VALUES = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet")


def streams_for(dataset: str, split: str) -> list[Stream]:
    streams = [s for s in load(dataset) if s.split == split]
    return streams


def code_notes(stream: Stream) -> Iterator[tuple[Turn, list[str] | None]]:
    """Each write with the older versions it replaces, or None when no note is asked about.

    A fact seen for the first time replaces nothing. The first snapshot only fills the store, so it is not asked about,
    and neither are commit notes and reconfirmations.
    """
    chains: dict[str, list[str]] = {}
    for turn in stream.writes:
        chain = None if turn.key is None else chains.setdefault(turn.key, [])
        if chain is None or (chain and chain[-1] == turn.text):
            yield turn, None
            continue
        older = [t for t in chain if t != turn.text]
        chain.append(turn.text)
        yield turn, None if len(chain) == 1 and turn.session == "0" else older


def talk_notes(stream: Stream) -> Iterator[tuple[Turn, list[str] | None]]:
    """The turn with the latest fact replaces the earlier evidence. User turns after the first session replace nothing."""
    query = stream.queries[0]
    seen: list[str] = []
    for turn in stream.writes:
        fact = replace(turn, kind="semantic")
        if turn.text in query.gold:
            yield fact, list(seen) or None
        elif turn.text in query.stale:
            seen.append(turn.text)
            yield fact, None
        else:
            yield fact, [] if turn.session != "0" and turn.text.startswith("user:") else None


def notes_of(stream: Stream, dataset: str) -> Iterator[tuple[Turn, list[str] | None]]:
    return talk_notes(stream) if dataset in TALK else code_notes(stream)


def config(vectors: bool) -> RetrievalConfig:
    if vectors:
        return RetrievalConfig(use_bandit=False)
    return RetrievalConfig(use_bandit=False, use_dense=False, gamma=0.0, neighbour_top=0, theta0=LEXICAL_THETA)


def store(mem: Memory, cache: TextCache, turn: Turn, id_of: dict[str, int], vectors: bool) -> None:
    """Write with no key. Without vectors the store is the one the plugins start with, where near-twins go by hashed words."""
    vector = cache.vector(turn.text) if vectors else None
    item = mem.write(turn.text, at=turn.at, kind=turn.kind, embedding=vector, session=turn.session)
    if item is not None:
        id_of[turn.text] = item


def candidates(job: tuple) -> list[dict]:
    """Scores of everything ``similar`` could show for each note of one stream, for picking its thresholds."""
    stream, dataset, cache_name, spread = job
    memory.SIMILAR_SPREAD = spread
    cache = TextCache(cache_name)
    mem = new_memory(cache, config(True))
    id_of: dict[str, int] = {}
    out = []
    for turn, older in notes_of(stream, dataset):
        store(mem, cache, turn, id_of, True)
        item = id_of.get(turn.text)
        if older is None or item is None:
            continue
        old = {id_of[t] for t in older if t in id_of} - {item}
        if older and not old:
            continue
        near = mem._near_facts(turn.text, SCOPE, (item,), cache.vector(turn.text))
        lexical = mem._near_facts(turn.text, SCOPE, (item,))
        out.append({
            "changed": bool(old),
            "near": [(int(row["id"]) in old, common, weight, closeness) for row, common, weight, closeness in near],
            "lexical": [(int(row["id"]) in old, common, weight, 0.0) for row, common, weight, _ in lexical],
        })
    mem.close()
    return out


def rates(rows: list[dict], which: str, overlap: float, shared: float, cosine: float) -> tuple[float, float]:
    """Share of changed notes with the older version in the list, and share of other notes with any list."""
    hits, shown = [], []
    for row in rows:
        words = [hit for hit, common, weight, _ in row[which] if common >= overlap or weight >= shared]
        close = [entry for entry in row[which] if entry[3] >= cosine and not (entry[1] >= overlap or entry[2] >= shared)]
        listed = memory.alternate(words, [entry[0] for entry in sorted(close, key=lambda entry: -entry[3])])[:5]
        if row["changed"]:
            hits.append(any(listed))
        else:
            shown.append(bool(listed))
    return float(np.mean(hits)), float(np.mean(shown))


def lists_short_updates(overlap: float, shared: float) -> bool:
    """Whether these thresholds show an old sentence when one word of it changes, with or without other versions around."""
    for template in SHORT_UPDATES:
        for others in (1, 9):
            mem = Memory()
            for value in VALUES[:others]:
                mem.write(template.format(value), kind="semantic")
            note = template.format(VALUES[others])
            item = mem.write(note, kind="semantic")
            listed = mem.similar(note, SCOPE, exclude=(item,), overlap=overlap, shared=shared)
            mem.close()
            if not listed:
                return False
    return True


def pick(notes: dict[tuple[str, float], list[dict]]) -> dict:
    """The thresholds with the highest mean of hit rate minus false alarm rate, with each kind of data counted once.

    Only settings that list a one-word edit of a short sentence compete. Vectors get their threshold last.
    """
    def gain(spread: float, which: str, overlap: float, shared: float, cosine: float) -> dict:
        kinds = {d: rates(notes[(d, spread)], which, overlap, shared, cosine) for d in PICK_ON}
        return {"score": float(np.mean([hit - shown for hit, shown in kinds.values()])), "rates": kinds}

    def best_of(pairs: list[tuple[float, float]]) -> dict:
        table = [{"spread": s, "overlap": o, "shared": t} | gain(s, "lexical", o, t, 2.0)
                 for s, (o, t) in itertools.product(SPREADS, pairs)]
        return max(table, key=lambda e: e["score"])

    pairs = list(itertools.product(OVERLAPS, SHAREDS))
    best = best_of([pair for pair in pairs if lists_short_updates(*pair)])
    close = [{"cosine": c} | gain(best["spread"], "near", best["overlap"], best["shared"], c) for c in COSINES]
    return {"lexical": best, "without_short_updates": best_of(pairs), "cosine": max(close, key=lambda e: e["score"]), "cosines": close}


def replay(job: tuple) -> dict:
    """One stream with no keys: what ``similar`` lists for each note, and the stale rate on changed facts at the end."""
    stream, dataset, cache_name, vectors, oracle = job
    cache = TextCache(cache_name)
    cfg = config(vectors)
    mem = new_memory(cache, cfg)
    methods = {"kanerva": kanerva_variant(cfg)}
    id_of: dict[str, int] = {}
    pending = [q for q in stream.queries if q.category.endswith("-changed") or q.category == "update"]
    notes, stale = [], []
    seconds = 0.0
    for cursor, (turn, older) in enumerate(notes_of(stream, dataset)):
        while pending and pending[0].pos <= cursor:
            stale.extend(evaluate_query(mem, cache, pending.pop(0), id_of, methods, (BUDGET,), stream.name))
        store(mem, cache, turn, id_of, vectors)
        item = id_of.get(turn.text)
        if older is None or item is None:
            continue
        old = {id_of[t] for t in older if t in id_of} - {item}
        if older and not old:
            continue
        start = time.perf_counter()
        listed = mem.similar(turn.text, SCOPE, exclude=(item,), query_vector=cache.vector(turn.text) if vectors else None)
        seconds += time.perf_counter() - start
        found = [i.id for i in listed]
        if oracle:
            for hit in found:
                if hit in old:
                    mem.supersede(hit, item, at=turn.at)
        notes.append({
            "changed": bool(old), "shown": bool(found), "top1": bool(found) and found[0] in old,
            "top3": any(i in old for i in found),
            "tokens": approx_tokens(json.dumps([{"id": i.id, "text": short_note(i.text, turn.text)} for i in listed])) if listed else 0,
        })
    for query in pending:
        stale.extend(evaluate_query(mem, cache, query, id_of, methods, (BUDGET,), stream.name))
    mem.close()
    return {"notes": notes, "stale": stale, "seconds": seconds}


def run(pool: ProcessPoolExecutor, dataset: str, cache_name: str, split: str, vectors: bool, oracle: bool) -> dict:
    jobs = [(s, dataset, cache_name, vectors, oracle) for s in streams_for(dataset, split)]
    parts = list(pool.map(replay, jobs))
    return {
        "streams": len(jobs), "notes": [n for p in parts for n in p["notes"]], "stale": [r for p in parts for r in p["stale"]],
        "seconds": sum(p["seconds"] for p in parts),
    }


def summary_of(outcome: dict) -> dict:
    notes = outcome["notes"]
    changed = [n for n in notes if n["changed"]]
    fresh = [n for n in notes if not n["changed"]]
    shown = [n for n in notes if n["shown"]]
    return {
        "streams": outcome["streams"],
        "changed_notes": len(changed),
        "top1": float(np.mean([n["top1"] for n in changed])),
        "top3": float(np.mean([n["top3"] for n in changed])),
        "other_notes": len(fresh),
        "shown_on_other_notes": float(np.mean([n["shown"] for n in fresh])),
        "notes_with_a_list": len(shown),
        "top_hit_not_older": float(np.mean([not n["top1"] for n in shown])),
        "top_hit_not_older_on_changed": float(np.mean([not n["top1"] for n in changed if n["shown"]])),
        "list_tokens_when_shown": float(np.mean([n["tokens"] for n in shown])),
        "ms_per_call": 1000 * outcome["seconds"] / max(len(notes), 1),
    }


def stale_of(outcome: dict) -> dict:
    rows = outcome["stale"]
    return {
        "queries": len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in ("stale_any", "stale_error", "recall")},
    }


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    workers = int(os.environ.get("KANERVA_WORKERS", max(1, (os.cpu_count() or 2) - 4)))
    start = time.time()
    with ProcessPoolExecutor(workers) as pool:
        notes: dict[tuple[str, float], list[dict]] = {}
        for (dataset, (cache_name, split)), spread in itertools.product(PICK_ON.items(), SPREADS):
            jobs = [(s, dataset, cache_name, spread) for s in streams_for(dataset, split)]
            notes[(dataset, spread)] = [n for chunk in pool.map(candidates, jobs) for n in chunk]
            print(f"candidates {dataset} {spread}: {len(notes[(dataset, spread)])} notes, {time.time() - start:.0f}s", flush=True)
        picked = pick(notes)
        print(json.dumps(picked), flush=True)
        chosen = {k: picked["lexical"][k] for k in ("spread", "overlap", "shared")}
        shipped = {"spread": memory.SIMILAR_SPREAD, "overlap": memory.SIMILAR_OVERLAP, "shared": memory.SIMILAR_SHARED}
        if sys.argv[1:] == ["pick"]:
            print("chosen", chosen, "shipped", shipped)
            return
        if chosen != shipped:
            raise SystemExit(f"the shipped thresholds {shipped} differ from the picked ones {chosen}")
        report: dict = {
            "meta": {
                "experiment": "write path without keys", "commit": git_commit(), "simulated": ["oracle"], "llm_calls": 0,
                "picked_on": {d: s for d, (_, s) in PICK_ON.items()}, "reported_on": {d: s for d, (_, s) in REPORT_ON.items()},
                "budget": BUDGET, "shipped": shipped | {"cosine": memory.SIMILAR_COSINE},
                "other_notes": "facts first seen after the first snapshot; user turns after the first session",
            },
            "selection": picked,
        }
        for vectors in (False, True):
            arm = "vectors" if vectors else "lexical"
            report[arm] = {}
            for dataset, (cache_name, split) in REPORT_ON.items():
                current = run(pool, dataset, cache_name, split, vectors, False)
                print(f"{arm} {dataset} current: {time.time() - start:.0f}s", flush=True)
                marked = run(pool, dataset, cache_name, split, vectors, True)
                print(f"{arm} {dataset} oracle: {time.time() - start:.0f}s", flush=True)
                report[arm][dataset] = {
                    "similar": summary_of(current),
                    "stale_on_changed_facts": {"current_rule": stale_of(current), "oracle_simulated": stale_of(marked)},
                }
                print(json.dumps(report[arm][dataset]), flush=True)
    report["meta"]["wall_seconds"] = round(time.time() - start, 1)
    (RESULTS / "write_path.json").write_text(json.dumps(report, indent=1))
    print("commit", git_commit())


if __name__ == "__main__":
    sys.exit(main())
