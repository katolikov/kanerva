"""What the plugins inject when no stored note is relevant, and a gate that lets them stay quiet.

Each store answers its own questions, questions from another store of the same kind, and questions from another kind of
data altogether. Budgets stay at the plugin default. Writes one row per query to results/raw/abstain_<set>.json.gz;
``report`` scores the gate on each set out of fold, writes results/abstention.json and prints the constants to ship.
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
from sklearn.linear_model import LogisticRegression

from experiments.embed_cache import TextCache
from experiments.harness import RESULTS, apply_write, git_commit, new_memory
from experiments.methods import LEXICAL
from experiments.registry import load
from kanerva import RetrievalConfig
from kanerva.policy import LEXICAL_SIGNALS, QUERY_GATES, QUERY_SIGNALS, QueryGate, query_signals
from kanerva.text import content_terms

BUDGET = 600
WEIGHTS = (0.0, 0.01, 0.02)
OWN_PER_STREAM = 30
OTHER_PER_STREAM = 25
MAX_STREAMS = 12
KEEP = 0.97  # the gate may cost this share of own recall on the data it is fitted on
SETS = {
    "lme": ("lme", "locomo"),
    "locomo": ("locomo", "repomem"),
    "beam-dev": ("beam", "repomem"),
    "repomem": ("repomem", "locomo"),
    "repomem-holdout": ("repomem-holdout", "locomo"),
    "repomem-fresh": ("repomem-fresh", "locomo"),
    "repomem-new": ("repomem-new", "locomo"),
    "repomem-other": ("repomem-other", "locomo"),
}
HELD_OUT = {"repomem-final": ("repomem-final", "locomo"), "repomem-c-test": ("repomem-c-test", "locomo"), "repomem-c-final": ("repomem-c-final", "locomo")}
STATIC = ("lme", "locomo", "beam-dev", "repomem-new", "repomem-fresh")  # the sets with static vectors in the cache


def configs(static: bool = False) -> dict[str, RetrievalConfig]:
    base = RetrievalConfig().with_(use_bandit=False)
    return {"static": base.with_(model="static")} if static else {"full": base, "lexical": base.with_(**LEXICAL)}


def measure(mem, name, cfg, question, qvec, now, gold, base) -> dict:
    prepared = mem.prepare(question, now, None, cfg, qvec)
    cand = prepared.candidates
    weights = prepared.problem.weights if prepared.problem is not None else None
    row = base | {"config": name} | dict(zip(QUERY_SIGNALS, query_signals(cand, weights, len(content_terms(question))).tolist()))
    for t in WEIGHTS:
        result = mem.select(prepared, BUDGET, record=False, min_weight=t)
        row[f"tokens@{t}"] = result.tokens
        if gold:
            row[f"hit@{t}"] = len(gold & {item.id for item in result.items}) / len(gold)
    return row


def run_stream(args: tuple) -> list[dict]:
    set_name, cache_name, keyed, stream, same, other, static = args
    cache = TextCache(cache_name)
    cfgs = configs(static)
    mem = new_memory(cache, next(iter(cfgs.values())))
    id_of: dict[str, int] = {}
    queries = [q for q in stream.queries if q.gold]
    keep = {id(q) for q in random.Random(stream.name).sample(queries, min(OWN_PER_STREAM, len(queries)))}
    rows, cursor, now = [], 0, 0.0
    for query in stream.queries:
        while cursor < query.pos:
            apply_write(mem, cache, stream.writes[cursor], id_of, keyed)
            cursor += 1
        now = query.now
        if id(query) not in keep:
            continue
        gold = {id_of[t] for t in query.gold if t in id_of}
        if gold:
            base = {"set": set_name, "stream": stream.name, "kind": "own", "category": str(query.category)}
            qvec = cache.vector(query.question)
            rows.extend(measure(mem, n, c, query.question, qvec, query.now, gold, base) for n, c in cfgs.items())
    while cursor < len(stream.writes):
        apply_write(mem, cache, stream.writes[cursor], id_of, keyed)
        cursor += 1
    for kind, foreign in (("same", same), ("other", other)):
        for question, qvec in foreign:
            base = {"set": set_name, "stream": stream.name, "kind": kind, "category": ""}
            rows.extend(measure(mem, n, c, question, qvec, now, set(), base) for n, c in cfgs.items())
    mem.close()
    return rows


def owner(stream) -> str:
    """Streams of one repository or conversation share their source and differ by seed."""
    stem, _, seed = stream.name.rpartition("-")
    return stem if seed in ("0", "1", "2") and not stem.startswith("beam") else stream.name


def run_set(set_name: str, cache_name: str, other_name: str, static: bool = False) -> list[dict]:
    streams = load(set_name)
    random.Random(set_name).shuffle(streams)
    chosen = streams[:MAX_STREAMS]
    suffix = "-m2v" if static else ""
    cache_name += suffix
    cache = TextCache(cache_name)
    other_cache = TextCache(other_name + suffix)
    other_questions = [q.question for s in load(other_name) for q in s.queries]
    jobs = []
    for index, stream in enumerate(chosen):
        rng = random.Random(index)
        same = [q.question for s in streams if owner(s) != owner(stream) for q in s.queries]
        same = rng.sample(same, min(OTHER_PER_STREAM, len(same)))
        other = rng.sample(other_questions, OTHER_PER_STREAM)
        jobs.append((set_name, cache_name, False, stream, [(q, cache.vector(q)) for q in same],
                     [(q, other_cache.vector(q)) for q in other], static))
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for part in pool.map(run_stream, jobs):
            rows.extend(part)
    return rows


def matrix(rows: list[dict], use: tuple[str, ...]) -> np.ndarray:
    return np.array([[r[f] for f in use] for r in rows], dtype=float)


def balance(rows: list[dict]) -> np.ndarray:
    """Each kind of query in each set weighs the same."""
    counts = Counter((r["set"], r["kind"]) for r in rows)
    weights = np.array([1.0 / counts[(r["set"], r["kind"])] for r in rows])
    return weights * len(rows) / weights.sum()


def fit_gate(rows: list[dict], use: tuple[str, ...], t: float) -> QueryGate:
    """A logistic model of "this store can answer this query", with the threshold that keeps KEEP of own recall."""
    x = matrix(rows, use)
    mean, scale = x.mean(0), x.std(0) + 1e-9
    y = np.array([r["kind"] == "own" for r in rows], dtype=int)
    model = LogisticRegression(C=1.0, max_iter=2000).fit((x - mean) / scale, y, sample_weight=balance(rows))
    gate = QueryGate(use, tuple(mean.tolist()), tuple(scale.tolist()), tuple(model.coef_[0].tolist()), float(model.intercept_[0]), 0.0)
    own = [r for r in rows if r["kind"] == "own"]
    scores = gate.score(matrix(own, use))
    hits = np.array([r[f"hit@{t}"] for r in own])
    order = np.argsort(scores)
    mass = np.cumsum(hits[order]) / hits.sum()
    return replace(gate, threshold=float(scores[order][np.searchsorted(mass, 1.0 - KEEP)]))


def outcome(rows: list[dict], gate: QueryGate | None, t: float) -> dict:
    """Recall and tokens on a store's own questions, and how often it speaks on questions it cannot answer."""
    out: dict = {}
    passed = np.ones(len(rows), dtype=bool) if gate is None else gate.score(matrix(rows, gate.use)) >= gate.threshold
    tokens = np.array([r[f"tokens@{t}"] for r in rows], dtype=float) * passed
    for kind in ("own", "same", "other"):
        now = np.array([r["kind"] == kind for r in rows], dtype=bool)
        if not now.any():
            continue
        out[kind] = {"spoke": round(float((tokens[now] > 0).mean()), 3), "tokens": round(float(tokens[now].mean()), 1),
                     "queries": int(now.sum())}
        if kind == "own":
            hits = np.array([r[f"hit@{t}"] for r, k in zip(rows, now) if k])
            out[kind]["recall"] = round(float((hits * passed[now]).mean()), 3)
    return out


def load_rows(name: str) -> list[dict]:
    return json.load(gzip.open(RESULTS / "raw" / f"abstain_{name}.json.gz", "rt"))


def report(t: float = 0.02) -> dict:
    """Each set is scored by a gate fitted on the others, then the shipped gates are fitted on all of them."""
    data = {name: load_rows(name) for name in SETS}
    out: dict = {"meta": {"experiment": "abstention", "commit": git_commit(), "budget": BUDGET, "min_weight": t, "keep": KEEP}, "sets": {}}
    gates = {}
    for config, use in (("lexical", LEXICAL_SIGNALS), ("full", QUERY_SIGNALS)):
        for name in SETS:
            train = [r for other, rows in data.items() if other != name for r in rows if r["config"] == config]
            test = [r for r in data[name] if r["config"] == config]
            out["sets"].setdefault(name, {})[config] = {"without": outcome(test, None, t), "with": outcome(test, fit_gate(train, use, t), t)}
        gates[config] = fit_gate([r for rows in data.values() for r in rows if r["config"] == config], use, t)
        print(f"{config}: {gates[config]}")
    (RESULTS / "abstention.json").write_text(json.dumps(out, indent=1))
    return gates


def final_report(name: str = "repomem-final", path: str = "abstention_final.json", t: float = 0.02) -> None:
    """The shipped gates on repositories they never saw, by language for the repositories that have no fact track."""
    rows = load_rows(name)
    out: dict = {"meta": {"experiment": f"abstention on {name}", "commit": git_commit(), "budget": BUDGET, "min_weight": t}}
    for config in ("lexical", "full"):
        now = [r for r in rows if r["config"] == config]
        out[config] = {"without": outcome(now, None, t), "with": outcome(now, QUERY_GATES[config], t)}
    (RESULTS / path).write_text(json.dumps(out, indent=1))


def static_report(t: float = 0.02) -> None:
    """The gates fitted on MiniLM scores, on stores with the static embedder, against a gate fitted on static scores."""
    data = {name: load_rows(f"{name}-static") for name in STATIC}
    out: dict = {"meta": {"experiment": "abstention, static embedder", "commit": git_commit(), "budget": BUDGET, "min_weight": t, "keep": KEEP}, "sets": {}}
    for name in STATIC:
        train = [r for other, rows in data.items() if other != name for r in rows]
        arms = {"none": None, "minilm": QUERY_GATES["full"], "lexical": QUERY_GATES["lexical"], "static": fit_gate(train, QUERY_SIGNALS, t)}
        out["sets"][name] = {arm: outcome(data[name], gate, t) for arm, gate in arms.items()}
    print(fit_gate([r for rows in data.values() for r in rows], QUERY_SIGNALS, t))
    (RESULTS / "abstention_static.json").write_text(json.dumps(out, indent=1))


def main(names: list[str]) -> None:
    for name in names:
        cache_name, other = (SETS | HELD_OUT)[name]
        rows = run_set(name, cache_name, other)
        with gzip.open(RESULTS / "raw" / f"abstain_{name}.json.gz", "wt") as f:
            json.dump(rows, f)
        print(name, len(rows), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        sys.exit(report() and None)
    if sys.argv[1:] == ["static"]:
        for name in STATIC:
            with gzip.open(RESULTS / "raw" / f"abstain_{name}-static.json.gz", "wt") as f:
                json.dump(run_set(name, *SETS[name], static=True), f)
            print(name, flush=True)
        sys.exit(static_report())
    if sys.argv[1:] == ["final"]:
        main(["repomem-final"])
        sys.exit(final_report())
    if sys.argv[1:] == ["c-test"]:
        main(["repomem-c-test"])
        sys.exit(final_report("repomem-c-test", "abstention_c.json"))
    if sys.argv[1:] == ["c-final"]:
        main(["repomem-c-final"])
        sys.exit(final_report("repomem-c-final", "abstention_c_final.json"))
    sys.exit(main(sys.argv[1:] or list(SETS)))
