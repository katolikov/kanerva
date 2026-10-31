"""Answer-level test: a local model answers each question from the notes a method selected.

``select`` replays a stream as experiments/harness.py does and stores one row per query, method and budget, with its prompt.
``read`` runs the model in its own environment, outside the project's dependencies, and appends every answer to a cache keyed
by the hash of its prompt, so reruns are free and a crash loses one answer at most. ``score`` needs no model. ``run`` does all
three, and the first run downloads the pinned model unless HF_HUB_OFFLINE is set:

    PYTHONPATH=.:src HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.answer_test run --tag full

Scoring has no judge. Prose is case-folded and cut into words, FILLER is dropped, number words become digits, ISO dates are
spelled out, ordinals lose their suffix, and words the question already contains are ignored on both sides, as in
harness.answer_stems. Code keeps its punctuation, without quotes, and its value is compared whole.

- contain: for prose, every gold word is in the answer, with repeats, in any order. For code, the gold tokens are one unbroken
  run in the answer, so 10 does not match 100.
- f1: harmonic mean of token precision and recall between the answer and the gold.
- unknown: the answer is empty or uses a phrase from REFUSALS.
- old: on a changed fact with an earlier value that differs from the current one, the answer gives an earlier value and not
  the current one. Longer values are matched first, so an answer of ">=1.2,<2" is not also an answer of ">=1.2".
- invented: on a question from another source, which the store cannot answer, the answer is not unknown. These questions run
  at GATE_BUDGET only, the budget nearest the plugin default.

A query is scorable if its category has short extractive answers and at most MAX_GOLD_TOKENS gold words, or MAX_CODE_CHARS
characters for code, remain. See ``SETS`` for the categories.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
ANSWERS = RAW / "answers_cache.jsonl"

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
READER = ("uv", "run", "--no-project", "--python", "3.12", "--with", "mlx-lm==0.32.2", "--with", "mlx==0.32.4",
          "--with", "transformers==5.22.1", "python", "-I")
MAX_TOKENS = 64

BUDGETS = (256, 512, 1024)
METHODS = ("kanerva", "kanerva-lex", "hybrid", "bm25", "kanerva-0.4", "kanerva-lex-0.4")
GATED = ("kanerva", "kanerva-lex")  # each also runs as name+weight, with the plugin threshold, and as name+gate, behind the gate
MIN_WEIGHT = 0.02
GATE_BUDGET = 512
FLOOR = "none"  # the same prompt with no notes
CEILING = "gold"  # the labelled evidence, to tell a reader's misses from a method's
PAIRS = (("kanerva", "hybrid"), ("kanerva", "bm25"), ("kanerva-lex", "bm25"), ("kanerva", "kanerva-lex"),
         ("kanerva", "kanerva-0.4"), ("kanerva-lex", "kanerva-lex-0.4"))
WORKERS = 3
NICE = 10

PROMPT = """Answer the question from these notes only. Give the answer alone, in as few words as possible. If the notes do not contain the answer, say unknown.

Today is {today}.

Notes:
{notes}

Question: {question}"""
NO_NOTES = "(no notes)"

MAX_GOLD_TOKENS = 8
MAX_CODE_CHARS = 120  # stays within MAX_TOKENS, so a full answer is never cut short
FILLER = frozenset("""a an the i me my mine you your yours is are was were am be been have has had do does did
    of to in on at for with and or""".split())
NUMBER_WORDS = dict(zip("zero one two three four five six seven eight nine ten eleven twelve".split(), map(str, range(13))))
MONTHS = "january february march april may june july august september october november december".split()
REFUSALS = ("unknown", "not mentioned", "not specified", "not stated", "not provided", "not available", "not contain",
            "not named", "not in the notes", "not specify", "not mention", "no information", "not enough information",
            "cannot be determined", "don't know", "do not know")

_WORD = re.compile(r"[^\W_]+")
_ISO_DATE = re.compile(r"\b(\d{4})-(0[1-9]|1[0-2])-(\d{2})\b")
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
_CODE = re.compile(r"\w+(?:\.\w+)*|[^\w\s]")
_ALSO = re.compile(r"^(.+?)\. (\d+ days?) \(including the last day\) is also acceptable\.?$")
_VALUE = {
    "constant": re.compile(r" is set to (.*)$", re.S),
    "signature": re.compile(r"^.+?\((.*)\) is the signature, defined in ", re.S),
    "location": re.compile(r" is defined in (.+)$", re.S),
    "dependency": re.compile(r"^The project requires \S+ (.+)$", re.S),
}


@dataclass(frozen=True)
class Spec:
    dataset: str
    cache: str
    unit: str  # the name experiments.stats.unit_of knows the independent unit by
    categories: tuple[str, ...]
    keyed: bool = False
    code: bool = False
    noise: bool = True


FACTS = tuple(f"fact-{kind}-{state}" for kind in _VALUE for state in ("changed", "stable"))
# Left out, with reasons. LoCoMo 3 asks for inference, and its golds add a free-form reason ("Yes; it's classical music").
# LoCoMo 5 is adversarial and has no gold. LoCoMo 2 golds are dates relative to a note, and LME temporal-reasoning needs date
# arithmetic: the reader contains neither gold even when handed the gold notes, so these cannot tell methods apart, and a wrong
# date still earns token F1 for its month and year. LME preference golds are rubrics for a judge. BEAM keeps knowledge_update
# only: its other golds are narratives, worked sums or restated dates, or do not exist. RepoMem tasks have no gold.
SETS = {
    "locomo": Spec("locomo", "locomo", "locomo", ("1", "4")),
    "lme": Spec("lme", "lme", "lme", ("knowledge-update", "multi-session", "single-session-assistant", "single-session-user")),
    "beam-dev": Spec("beam-dev", "beam", "beam", ("knowledge_update",)),
    "repomem": Spec("repomem", "repomem", "repomem", FACTS, code=True),
    "repomem-keyed": Spec("repomem", "repomem", "repomem", FACTS, keyed=True, code=True, noise=False),
    "repomem-final": Spec("repomem-final", "repomem-final", "repomem", FACTS, code=True),
}
# Queries per set in a full run, sized to about three hours of reading. repomem-final is the only data no decision has used.
SAMPLE = {"locomo": 60, "lme": 60, "beam-dev": 40, "repomem-final": 90}
NOISE = 40


def words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _ISO_DATE.sub(lambda m: f"{int(m[3])} {MONTHS[int(m[2]) - 1]} {m[1]}", text)
    return [NUMBER_WORDS.get(w, w) for w in _WORD.findall(_ORDINAL.sub(r"\1", text)) if w not in FILLER]


def code_tokens(text: str) -> list[str]:
    return [t for t in _CODE.findall(unicodedata.normalize("NFKC", text).casefold()) if t not in ("'", '"', "`")]


def core(text: str, question: str) -> list[str]:
    """Words of ``text`` the question does not already contain, or all of them if that leaves none."""
    asked = set(words(question))
    found = words(text)
    return [w for w in found if w not in asked] or found


def tokens_of(answer: str, gold: str, question: str, code: bool) -> tuple[list[str], list[str]]:
    if code:
        return code_tokens(answer), code_tokens(gold)
    return core(answer, question), core(gold, question)


def find_run(tokens: list[str], run: list[str]) -> int:
    """Where ``run`` starts in ``tokens``, or -1."""
    for start in range(len(tokens) - len(run) + 1):
        if tokens[start:start + len(run)] == run:
            return start
    return -1


def contains(answer: str, gold: str, question: str = "", code: bool = False) -> bool:
    have, want = tokens_of(answer, gold, question, code)
    if not want:
        return False
    return find_run(have, want) >= 0 if code else not Counter(want) - Counter(have)


def token_f1(answer: str, gold: str, question: str = "", code: bool = False) -> float:
    have, want = tokens_of(answer, gold, question, code)
    common = sum((Counter(have) & Counter(want)).values())
    if not common:
        return 0.0
    precision, recall = common / len(have), common / len(want)
    return 2 * precision * recall / (precision + recall)


def said_unknown(answer: str) -> bool:
    text = unicodedata.normalize("NFKC", answer).casefold().replace("’", "'")
    return not text.strip() or any(phrase in text for phrase in REFUSALS)


def version_given(answer: str, current: str, olds: list[str]) -> str:
    """Which version of a fact an answer gives: "current", "old", "both" or "neither"."""
    tokens = code_tokens(answer)
    values = [(code_tokens(current), "current")] + [(code_tokens(v), "old") for v in olds]
    found = set()
    for run, label in sorted(values, key=lambda v: -len(v[0])):
        start = find_run(tokens, run)
        if start >= 0:
            tokens[start:start + len(run)] = ["\0"]
            found.add(label)
    return "both" if len(found) == 2 else found.pop() if found else "neither"


def gold_forms(answer: str) -> list[str]:
    """A reference answer and the alternative it allows: "26 days. 27 days (including the last day) is also acceptable."."""
    allowed = _ALSO.match(answer)
    return [allowed.group(1), allowed.group(2)] if allowed else [answer]


def old_values(query) -> list[str]:
    """Earlier values of a changed fact that differ from the current one. A move to another file keeps the value."""
    if not query.answer or not query.category.endswith("-changed"):
        return []
    pattern = _VALUE[query.category.split("-")[1]]
    found = {m.group(1) for m in map(pattern.search, query.stale) if m}
    return sorted(v for v in found if code_tokens(v) and code_tokens(v) != code_tokens(query.answer))


def scorable(query, spec: Spec) -> bool:
    if query.category not in spec.categories or not query.answer or query.qid.endswith("_abs"):
        return False  # _abs questions in LongMemEval have the answer "the information is not enough"
    if spec.code:
        return len(query.answer) <= MAX_CODE_CHARS and any(c.isalnum() for c in query.answer)
    return all(0 < len(core(form, query.question)) <= MAX_GOLD_TOKENS for form in gold_forms(query.answer))


def sample(streams: list, spec: Spec, n: int) -> list[tuple[str, str]]:
    """``n`` scorable (stream, qid) pairs spread evenly over categories and streams. A smaller sample is a prefix."""
    rng = random.Random(f"answers-{spec.dataset}")
    groups: dict[tuple, list] = {}
    for stream in streams:
        for query in stream.queries:
            if scorable(query, spec):
                groups.setdefault((query.category, stream.name), []).append((stream.name, query.qid))
    ranked = []
    for group in groups.values():
        rng.shuffle(group)
        ranked.extend((rank, rng.random(), pair) for rank, pair in enumerate(group))
    return [pair for _, _, pair in sorted(ranked)[:n]]


def foreign_questions(streams: list, spec: Spec, n: int) -> dict[str, list[dict]]:
    """``n`` questions for stores that cannot answer them: another source, and neither the gold notes nor their key is held."""
    from experiments.stats import unit_of

    rng = random.Random(f"answers-noise-{spec.dataset}")
    by_name = {s.name: s for s in streams}
    pool = [(stream, query) for stream in streams for query in stream.queries if scorable(query, spec)]
    rng.shuffle(pool)
    order = sorted(s.name for s in streams if any(scorable(q, spec) for q in s.queries))
    rng.shuffle(order)
    held = {name: {t.text for t in by_name[name].writes} | {t.key for t in by_name[name].writes if t.key} for name in order}
    picked: dict[str, list[dict]] = {}
    for index in range(n):
        name = order[index % len(order)]
        while pool:
            stream, query = pool.pop()
            keys = {t.key for t in stream.writes if t.key and t.text in query.gold}
            same_source = (unit_of(spec.unit, stream.name) or stream.name) == (unit_of(spec.unit, name) or name)
            if not same_source and not (query.gold | keys) & held[name]:
                picked.setdefault(name, []).append({"qid": f"{stream.name}/{query.qid}", "question": query.question, "place": index / n})
                break
    return picked


def prompt_key(prompt: str) -> str:
    return hashlib.sha256(f"{MODEL}@{REVISION}/{MAX_TOKENS}\n{prompt}".encode()).hexdigest()


def render(mem, ids: list[int], texts: list[str], marked: bool) -> str:
    """Notes as Item.render shows them, numbered by position so that equal notes give equal prompts in any store."""
    rows = mem._fetch(ids)
    lines = []
    for position, (i, text) in enumerate(zip(ids, texts), 1):
        date = time.strftime("%Y-%m-%d", time.gmtime(rows[i]["created_at"]))
        mark = " superseded" if marked and rows[i]["valid_to"] is not None else ""
        mark += " excerpt" if text != rows[i]["text"] else ""
        lines.append(f"[#{position} {date}{mark}] {text}")
    return "\n".join(lines) or NO_NOTES


def thresholded(cfg, gated: bool):
    """kanerva_variant with the plugin threshold and, if ``gated``, the check that the store can answer at all."""

    def run(ctx, budget: int) -> list[int]:
        key = ("kanerva", id(cfg))
        if key not in ctx.cache:
            ctx.cache[key] = ctx.mem.prepare(ctx.question, ctx.now, None, cfg, ctx.qvec)
        result = ctx.mem.select(ctx.cache[key], budget, record=False, min_weight=MIN_WEIGHT, gated=gated)
        if any(item.truncated for item in result.items):
            ctx.cache["texts"] = [item.text for item in result.items]
            ctx.cache["text_costs"] = [ctx.mem.count_tokens(item.text) + ctx.overhead for item in result.items]
        return [item.id for item in result.items]

    return run


def selectors(base) -> tuple[dict, dict]:
    """The methods, and the threshold and gate conditions that run at GATE_BUDGET."""
    from experiments.methods import BASELINES, kanerva_variant, variants

    configs = {name: cfg.with_(use_bandit=False) for name, cfg in variants(base).items() if name in METHODS}
    methods = {name: kanerva_variant(cfg) for name, cfg in configs.items()} | {name: BASELINES[name] for name in METHODS if name in BASELINES}
    gates = {}
    for name in GATED:
        gates[f"{name}+weight"] = thresholded(configs[name], False)
        gates[f"{name}+gate"] = thresholded(configs[name], True)
    return methods, gates


def ask(mem, cache, methods: dict, gates: dict, base: dict, question: str, now: float, gold: set[int], stale: set[int],
        budgets: tuple, prompts: dict) -> list[dict]:
    """One row per method and budget for one question, plus the gate conditions, the floor and, with gold notes, the ceiling."""
    from experiments.harness import score
    from experiments.methods import Context, fill

    ctx = Context(mem, question, now, cache.vector(question))
    today = time.strftime("%Y-%m-%d", time.gmtime(now))

    def row(method: str, budget: int, ids: list[int], texts: list[str] | None = None, tokens: int | None = None) -> dict:
        if texts is None:
            fetched = mem._fetch(ids)
            texts = [fetched[i]["text"] for i in ids]
        prompt = PROMPT.format(today=today, notes=render(mem, ids, texts, method.startswith("kanerva")), question=question)
        key = prompt_key(prompt)
        prompts[key] = prompt
        scores = score(ids, gold, stale, ctx.cost(ids) if ids else {}, 0.0)
        return base | {"method": method, "budget": budget, "key": key, "tokens": scores["tokens"] if tokens is None else tokens,
                       "recall": scores["recall"], "any": scores["any"], "stale_any": scores["stale_any"]}

    def selected(name: str, method, budget: int) -> dict:
        ctx.cache.pop("texts", None)
        ids = method(ctx, budget)
        texts = ctx.cache.get("texts")
        return row(name, budget, ids, texts, None if texts is None else sum(ctx.cache["text_costs"]))

    rows = [row(FLOOR, 0, [])]
    rows.extend(selected(name, method, budget) for name, method in methods.items() for budget in budgets)
    rows.extend(selected(name, method, GATE_BUDGET) for name, method in gates.items())
    if gold:
        fetched = mem._fetch(sorted(gold))
        newest = sorted(gold, key=lambda i: -fetched[i]["created_at"])
        rows.append(row(CEILING, BUDGETS[-1], fill(newest, ctx, BUDGETS[-1])))
    return rows


def select_stream(args: tuple) -> tuple[list[dict], dict[str, str]]:
    from experiments.embed_cache import TextCache
    from experiments.harness import apply_write, new_memory
    from kanerva import RetrievalConfig

    name, spec, stream, wanted, foreign = args
    cache = TextCache(spec.cache)
    config = RetrievalConfig()
    mem = new_memory(cache, config)
    methods, gates = selectors(config)
    id_of: dict[str, int] = {}
    rows: list[dict] = []
    prompts: dict[str, str] = {}
    cursor, now = 0, 0.0
    for query in stream.queries:
        while cursor < query.pos:
            apply_write(mem, cache, stream.writes[cursor], id_of, spec.keyed)
            cursor += 1
        now = query.now
        gold = {id_of[t] for t in query.gold if t in id_of}
        if query.qid in wanted and gold:
            stale = {id_of[t] for t in query.stale if t in id_of} - gold
            base = {"set": name, "stream": stream.name, "qid": query.qid, "category": query.category, "kind": "own",
                    "question": query.question, "gold": gold_forms(query.answer), "old": old_values(query), "code": spec.code,
                    "place": wanted[query.qid]}
            rows.extend(ask(mem, cache, methods, gates, base, query.question, query.now, gold, stale, BUDGETS, prompts))
    while cursor < len(stream.writes):
        apply_write(mem, cache, stream.writes[cursor], id_of, spec.keyed)
        cursor += 1
    for item in foreign:
        base = {"set": name, "stream": stream.name, "qid": item["qid"], "category": "", "kind": "foreign",
                "question": item["question"], "gold": [], "old": [], "code": spec.code, "place": item["place"]}
        rows.extend(ask(mem, cache, methods, gates, base, item["question"], now, set(), set(), (GATE_BUDGET,), prompts))
    mem.close()
    return rows, prompts


def rows_path(tag: str, name: str) -> Path:
    return RAW / f"answers_{tag}_{name}.json.gz"


def select(name: str, tag: str, queries: int, noise: int) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from experiments.harness import git_commit
    from experiments.registry import load

    start = time.time()
    spec = SETS[name]
    streams = load(spec.dataset)
    wanted: dict[str, dict[str, float]] = {}
    picks = sample(streams, spec, queries)
    for place, (stream, qid) in enumerate(picks):
        wanted.setdefault(stream, {})[qid] = place / len(picks)
    foreign = foreign_questions(streams, spec, noise) if spec.noise else {}
    available = Counter(q.category for s in streams for q in s.queries if scorable(q, spec))
    jobs = [(name, spec, s, wanted.get(s.name, {}), foreign.get(s.name, [])) for s in streams
            if s.name in wanted or s.name in foreign]
    rows: list[dict] = []
    prompts: dict[str, str] = {}
    with ProcessPoolExecutor(WORKERS) as pool:
        for part, seen in pool.map(select_stream, jobs):
            rows.extend(part)
            prompts.update(seen)
    rows.sort(key=lambda r: r["place"])  # stopping early leaves a prefix of the sample
    prompts = {r["key"]: prompts[r["key"]] for r in rows}
    meta = {"set": name, "tag": tag, "commit": git_commit(), "streams": len(jobs), "available": dict(available),
            "wall_seconds": round(time.time() - start, 1)}
    with gzip.open(rows_path(tag, name), "wt") as fh:
        json.dump({"meta": meta, "rows": rows, "prompts": prompts}, fh)
    print(f"{name}: {len(rows)} rows, {len(prompts)} prompts, {meta['wall_seconds']}s", flush=True)


def load_answers() -> dict[str, dict]:
    """Every cached answer. A last line cut short by a crash is skipped."""
    answers: dict[str, dict] = {}
    if ANSWERS.exists():
        for line in ANSWERS.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            answers[entry["key"]] = entry
    return answers


def read(names: list[str], tag: str) -> None:
    """Answer every prompt that has no cached answer. This command runs in the reader's environment, see READER."""
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    done = load_answers()
    prompts: dict[str, str] = {}
    order: list[tuple[float, str]] = []
    for name in names:
        with gzip.open(rows_path(tag, name), "rt") as fh:
            part = json.load(fh)["prompts"]
        prompts.update(part)
        order.extend((index / len(part), key) for index, key in enumerate(part))
    todo = [key for key in dict.fromkeys(key for _, key in sorted(order)) if key not in done]  # every set advances together
    print(f"{len(prompts)} prompts, {len(todo)} to answer", flush=True)
    if not todo:
        return
    model, tokenizer = load(MODEL, revision=REVISION)
    greedy = make_sampler(temp=0.0)
    start = time.time()
    with ANSWERS.open("a") as out:
        out.write("\n")  # a crash can leave a partial line
        for index, key in enumerate(todo, 1):
            message = [{"role": "user", "content": prompts[key]}]
            prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
            begun = time.perf_counter()
            text, last = "", None
            for last in stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=greedy):
                text += last.text
            entry = {"key": key, "text": text.strip(), "prompt_tokens": last.prompt_tokens, "tokens": last.generation_tokens,
                     "finish": last.finish_reason, "seconds": round(time.perf_counter() - begun, 3)}
            out.write(json.dumps(entry) + "\n")
            out.flush()
            os.fsync(out.fileno())
            if index % 100 == 0 or index == len(todo):
                pace = (time.time() - start) / index
                print(f"{index}/{len(todo)} {pace:.2f}s each, {(len(todo) - index) * pace / 60:.0f} min left, "
                      f"peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)


def judge(row: dict, entry: dict) -> dict:
    answer = entry["text"]
    refused = said_unknown(answer)
    out = {"unknown": float(refused), "invented": float(not refused), "cut": float(entry["finish"] == "length")}
    if row["gold"]:
        forms = [(form, row["question"], row["code"]) for form in row["gold"]]
        out["contain"] = float(any(contains(answer, *form) for form in forms))
        out["f1"] = max(token_f1(answer, *form) for form in forms)
    if row["old"]:
        version = version_given(answer, row["gold"][0], row["old"])
        out |= {"version": version, "gave_old": float(version == "old"), "gave_both": float(version == "both"),
                "gave_current": float(version == "current")}
    return out


def mean(rows: list[dict], metric: str) -> float | None:
    values = [r[metric] for r in rows]
    return round(sum(values) / len(values), 4) if values else None


def by_method(rows: list[dict]) -> dict[str, list[dict]]:
    table: dict[str, list[dict]] = {}
    for r in rows:
        table.setdefault(f"{r['method']}@{r['budget']}", []).append(r)
    return dict(sorted(table.items()))


def answers_summary(rows: list[dict]) -> dict:
    out = {}
    for name, part in by_method(rows).items():
        hit = [r for r in part if r["any"]]
        miss = [r for r in part if not r["any"]]
        out[name] = {"n": len(part), "contain": mean(part, "contain"), "f1": mean(part, "f1"), "unknown": mean(part, "unknown"),
                     "recall": mean(part, "recall"), "tokens": mean(part, "tokens"), "cut": mean(part, "cut"),
                     "contain_with_evidence": mean(hit, "contain"), "contain_without": mean(miss, "contain")}
    return out


def compare(rows: list[dict], unit: str, a: str, b: str, metric: str, rng, budgets: tuple = BUDGETS) -> dict:
    """Per-budget tests with Holm correction within this comparison, plus the mean over all budgets when there are several."""
    from experiments.final_test import per_query
    from experiments.stats import holm, paired

    def at(method: str, budgets: tuple) -> tuple:
        return (0,) if method == FLOOR else budgets

    def test(budgets: tuple) -> dict | None:
        left, right = per_query(rows, a, metric, at(a, budgets)), per_query(rows, b, metric, at(b, budgets))
        return paired(left, right, rng, unit) if set(left) & set(right) else None

    out = {str(budget): test((budget,)) for budget in budgets}
    holm([entry for entry in out.values() if entry])
    if len(budgets) > 1:
        out["mean"] = test(budgets)
    return out


def costs(rows: list[dict], answers: dict) -> dict:
    """Reader time per query, with each prompt charged once to the kind of query that first needs it."""
    seen: dict[str, str] = {}
    for r in rows:
        seen.setdefault(r["key"], r["kind"])
    out = {}
    for kind in ("own", "foreign"):
        entries = [answers[k] for k, v in seen.items() if v == kind and k in answers]
        queries = len({(r["stream"], r["qid"]) for r in rows if r["kind"] == kind})
        if entries:
            seconds = sum(e["seconds"] for e in entries)
            out[kind] = {"queries": queries, "prompts": len(entries), "seconds": round(seconds, 1),
                         "seconds_per_query": round(seconds / queries, 2),
                         "prompt_tokens": round(sum(e["prompt_tokens"] for e in entries) / len(entries), 1),
                         "answer_tokens": round(sum(e["tokens"] for e in entries) / len(entries), 1)}
    return out


def table(summary: dict, metric: str) -> str:
    conditions = [f"{name}+{suffix}" for name in GATED for suffix in ("weight", "gate")]
    lines = ["method".ljust(20) + "".join(f"{b:>8}" for b in BUDGETS)]
    for method in (*METHODS, *conditions, FLOOR, CEILING):
        cells = [summary.get(f"{method}@{b}", {}).get(metric) for b in BUDGETS]
        if method == FLOOR:
            cells = [summary.get(f"{FLOOR}@0", {}).get(metric)] * len(BUDGETS)
        lines.append(method.ljust(20) + "".join("       -" if c is None else f"{c:8.3f}" for c in cells))
    return "\n".join(lines)


def score_set(name: str, tag: str, answers: dict) -> dict:
    import numpy as np

    from experiments.harness import git_commit

    with gzip.open(rows_path(tag, name), "rt") as fh:
        stored = json.load(fh)
    spec = SETS[name]
    rng = np.random.default_rng(0)
    rows = [r | judge(r, answers[r["key"]]) | {"tokens_any": float(r["tokens"] > 0)} for r in stored["rows"] if r["key"] in answers]
    own = [r for r in rows if r["kind"] == "own"]
    foreign = [r for r in rows if r["kind"] == "foreign"]
    changed = [r for r in own if r["old"]]
    meta = stored["meta"] | {"scored_commit": git_commit(), "rows": len(stored["rows"]), "unanswered": len(stored["rows"]) - len(rows)}
    report = {"meta": meta, "queries": len({(r["stream"], r["qid"]) for r in own}), "cost": costs(stored["rows"], answers)}
    report["answers"] = answers_summary(own)
    categories = sorted({r["category"] for r in own})
    report["categories"] = {c: answers_summary([r for r in own if r["category"] == c]) for c in categories}
    report["tests"] = {f"{a} vs {b}": {m: compare(own, spec.unit, a, b, m, rng) for m in ("contain", "f1")} for a, b in PAIRS}
    if changed:
        report["stale"] = {
            key: {"n": len(part), "old": mean(part, "gave_old"), "both": mean(part, "gave_both"), "current": mean(part, "gave_current"),
                  "old_when_injected": mean([r for r in part if r["stale_any"]], "gave_old"), "injected": mean(part, "stale_any")}
            for key, part in by_method(changed).items()
        }
        report["stale_tests"] = {f"{a} vs {b}": compare(changed, spec.unit, a, b, "gave_old", rng) for a, b in PAIRS}
    if foreign:
        report["noise"] = {key: {"n": len(part), "invented": mean(part, "invented"), "injected": mean(part, "tokens_any")}
                           for key, part in by_method(foreign).items()}
        single = (GATE_BUDGET,)
        report["noise_tests"] = {f"{m} vs {FLOOR}": compare(foreign, spec.unit, m, FLOOR, "invented", rng, single) for m in METHODS}
        report["gate_tests"] = {}
        for method in GATED:
            gated, plain = f"{method}+gate", f"{method}+weight"
            report["gate_tests"][f"{gated} vs {plain}"] = {
                "invented": compare(foreign, spec.unit, gated, plain, "invented", rng, single),
                "contain": compare(own, spec.unit, gated, plain, "contain", rng, single),
                "f1": compare(own, spec.unit, gated, plain, "f1", rng, single),
            }
    (RAW / f"answers_{tag}_{name}_rows.json.gz").write_bytes(gzip.compress(json.dumps(rows).encode()))
    print(f"== {name}: {report['queries']} queries, {report['meta']['unanswered']} rows unanswered, cost {report['cost']}")
    for metric in ("contain", "f1", "unknown"):
        print(f"{metric}\n{table(report['answers'], metric)}")
    if changed:
        print(f"answered with the old value\n{table(report['stale'], 'old')}")
    if foreign:
        print(f"invented on foreign questions\n{table(report['noise'], 'invented')}")
    return report


def score(names: list[str], tag: str) -> None:
    answers = load_answers()
    sets = {name: score_set(name, tag, answers) for name in names}
    meta = {"experiment": "answer-level test", "commit": next(iter(sets.values()))["meta"]["commit"], "model": MODEL, "revision": REVISION,
            "max_tokens": MAX_TOKENS, "budgets": BUDGETS, "prompt": PROMPT}
    report = {"meta": meta, "sets": sets}
    # Only the full run is a result. A pilot checks the method, so it stays with the raw files.
    path = RESULTS / "answers.json" if tag == "full" else RAW / f"answers_{tag}_summary.json"
    path.write_text(json.dumps(report, indent=1))
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "select", "read", "score"))
    parser.add_argument("sets", nargs="*", default=list(SAMPLE))
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--queries", type=int, help="queries per set, by default SAMPLE")
    parser.add_argument("--noise", type=int, default=NOISE)
    args = parser.parse_intermixed_args()
    names = args.sets or list(SAMPLE)
    os.nice(max(0, NICE - os.nice(0)))
    if args.command == "read":
        return read(names, args.tag)
    if args.command in ("run", "select"):
        for name in names:
            select(name, args.tag, args.queries or SAMPLE[name], args.noise)
    if args.command == "run":
        subprocess.run([*READER, str(Path(__file__).resolve()), "read", "--tag", args.tag, *names], check=True)
    if args.command in ("run", "score"):
        score(names, args.tag)


if __name__ == "__main__":
    sys.exit(main())
