"""Russian and cross-language search on XQuAD, and the multilingual model against all-MiniLM-L6-v2 on English data.

XQuAD gives each SQuAD paragraph and question in English and, by professional translation, in Russian. One store per
article holds its five paragraphs as notes, and a retrieval is correct if it returns the question's paragraph. Budgets
count whole notes by default, so the report also says how often a note fits at all and what a random order gets. Excerpts
and one store of all 240 paragraphs are there because five notes and small budgets say little by themselves.

Set KANERVA_TEST_MULTILINGUAL to a folder with the model, and KANERVA_TEST_MINILM_ONNX to a folder with an ONNX export
of all-MiniLM-L6-v2 for the timings. onnx/model.onnx in the first folder, if present, is scored for the quantised check.
Writes results/russian.json.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.datasets import ROOT, Stream, digest
from experiments.embed_cache import TextCache
from experiments.harness import RESULTS, apply_write, evaluate_query, git_commit, new_memory
from experiments.methods import Context, kanerva_variant, variants
from experiments.registry import load
from experiments.stats import holm, paired
from kanerva import Memory, RetrievalConfig, relevance, text
from kanerva.embed import (
    MINILM_FILES,
    MINILM_REPO,
    MINILM_REVISION,
    MULTILINGUAL_FILES,
    MULTILINGUAL_REPO,
    MULTILINGUAL_REVISION,
    MultilingualEmbedder,
    OnnxEmbedder,
    SentenceTransformerEmbedder,
)

MODEL_FOLDER = Path(os.environ.get("KANERVA_TEST_MULTILINGUAL", ROOT / "data" / "multilingual"))
MINILM_ONNX = os.environ.get("KANERVA_TEST_MINILM_ONNX")
WORKERS = 8
BUDGETS = (128, 256, 512, 1024)
SAMPLE_BUDGETS = (256, 512, 1024)
RANDOM_ORDERS = 50
OVERHEAD = 14  # as methods.Context
UNIT = "locomo"  # stats.unit_of takes the stream as the independent unit for this name: here the stream is the article
START = 1_700_000_000.0
BASE = RetrievalConfig(use_bandit=False)
LEXICAL = variants(BASE)["kanerva-lex"]
DENSE = variants(BASE)["kanerva"]
EXCERPTS = {"excerpt_threshold": 160, "excerpt_tokens": 128}  # the tested setting, as variants()["kanerva-x128"]
ENGLISH_ONLY = {
    "_DATE": (r"\b(19|20)\d\d\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d|"
              r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|yesterday|today|tomorrow|week|weekend)\b"),
    "_TEMPORAL": (r"\b(when|date|day|days|week|weeks|month|months|year|years|ago|last|first|recent|recently|latest|"
                  r"before|after|since|until|how long|earlier|later|currently|now)\b"),
    "_HOW_MANY": r"\bhow (many|much|long|often|old|far)\b|\bwhat (number|time|percentage)\b",
    "_WHEN": r"\bwhen\b|\bwhat (date|day|year|month|time)\b",
}


@dataclass
class Article:
    title: str
    notes: dict[str, list[str]]  # language to paragraphs
    questions: list[dict]  # id, paragraph index, and per language the question and its answers


def read_xquad() -> list[Article]:
    """Both languages, joined by question id. The files are parallel, and this checks that they are."""
    files = {lang: json.loads((ROOT / "data" / "xquad" / f"xquad.{lang}.json").read_text()) for lang in ("ru", "en")}
    articles = []
    for ru, en in zip(files["ru"]["data"], files["en"]["data"], strict=True):
        assert ru["title"] == en["title"]
        notes = {"ru": [p["context"].lstrip("﻿") for p in ru["paragraphs"]], "en": [p["context"] for p in en["paragraphs"]]}
        questions = []
        for index, (a, b) in enumerate(zip(ru["paragraphs"], en["paragraphs"], strict=True)):
            english = {q["id"]: q for q in b["qas"]}
            assert len(english) == len(b["qas"]) == len(a["qas"]) and set(english) == {q["id"] for q in a["qas"]}
            for q in a["qas"]:
                questions.append({"id": q["id"], "paragraph": index, "ru": q["question"].lstrip("﻿"), "en": english[q["id"]]["question"],
                                  "answers_ru": [x["text"] for x in q["answers"]], "answers_en": [x["text"] for x in english[q["id"]]["answers"]]})
        articles.append(Article(ru["title"], notes, questions))
    return articles


@lru_cache(maxsize=1)
def counter():
    return text.tiktoken_counter()


@lru_cache(maxsize=None)
def tokens(s: str) -> int:
    """cl100k_base tokens, as in the other benchmarks."""
    return counter()(s)


class Vectors:
    """Text to vector for one embedder."""

    def __init__(self, embedder, texts: list[str]):
        unique = list(dict.fromkeys(texts))
        start = time.perf_counter()
        matrix = embedder.encode(unique)
        self.seconds = time.perf_counter() - start
        self.table = {t: matrix[i] for i, t in enumerate(unique)}

    def __call__(self, s: str) -> np.ndarray:
        return self.table[s]


def no_vectors(_: str) -> None:
    return None


def run_stores(stores: list[tuple[str, list[str], list[dict]]], notes: str, questions: str, vectors, config: RetrievalConfig) -> dict:
    """``recall[budget][(store, question id)]``, and ``answer``: the paragraph came back with the answer text in view."""
    recall: dict = {b: {} for b in BUDGETS}
    shown: dict = {b: {} for b in BUDGETS}
    picks: dict = {b: {} for b in BUDGETS}
    method = kanerva_variant(config)
    for title, paragraphs, asked in stores:
        mem = Memory(":memory:", count_tokens=tokens, config=config)
        ids = [mem.write(p, at=START + j * 3600, kind="semantic", embedding=vectors(p)) for j, p in enumerate(paragraphs)]
        assert len(set(ids)) == len(paragraphs)
        now = START + len(paragraphs) * 3600
        for q in asked:
            qvec = vectors(q[questions])
            ctx = Context(mem, q[questions], now, np.zeros(1) if qvec is None else qvec)
            for budget in BUDGETS:
                ctx.cache.pop("texts", None)
                chosen = method(ctx, budget)
                texts = ctx.cache.get("texts")
                if texts is None:
                    rows = mem._fetch(chosen)
                    texts = [rows[i]["text"] for i in chosen]
                key = (q.get("article", title), q["id"])
                gold = ids[q["paragraph"]]
                recall[budget][key] = float(gold in chosen)
                shown[budget][key] = float(gold in chosen and any(a in t for a in q[f"answers_{notes}"] for t in texts))
                picks[budget][key] = len(chosen)
        mem.close()
    return {"recall": recall, "answer": shown, "picks": picks}


def by_article(articles: list[Article], notes: str) -> list:
    return [(a.title, a.notes[notes], a.questions) for a in articles]


def pooled(articles: list[Article], notes: str) -> list:
    """One store of every paragraph. The question's own article stays the unit of the paired test."""
    paragraphs, asked = [], []
    for a in articles:
        asked += [{**q, "paragraph": q["paragraph"] + len(paragraphs), "article": a.title} for q in a.questions]
        paragraphs += a.notes[notes]
    return [("all", paragraphs, asked)]


def table(run: dict) -> dict:
    return {str(b): {"recall": float(np.mean(list(run["recall"][b].values()))), "answer": float(np.mean(list(run["answer"][b].values()))),
                     "notes_returned": float(np.mean(list(run["picks"][b].values()))), "n": len(run["recall"][b])} for b in BUDGETS}


def tests(a: dict, b: dict, metric: str, rng: np.random.Generator) -> dict:
    """``a`` minus ``b`` at each budget, with intervals over articles and Holm correction across budgets."""
    rows = {str(k): paired(a[metric][k], b[metric][k], rng, UNIT) for k in BUDGETS}
    holm(list(rows.values()))
    return rows


def random_order(articles: list[Article], notes: str, rng: np.random.Generator) -> dict:
    """How often a random order, filled to the budget, holds the question's paragraph."""
    out: dict = {}
    for budget in BUDGETS:
        rows = []
        for a in articles:
            costs = [tokens(p) + OVERHEAD for p in a.notes[notes]]
            held = np.zeros(len(costs))
            for _ in range(RANDOM_ORDERS):
                spent = 0
                for i in rng.permutation(len(costs)):
                    if spent + costs[i] <= budget:
                        spent += costs[i]
                        held[i] += 1 / RANDOM_ORDERS
            rows += [held[q["paragraph"]] for q in a.questions]
        out[str(budget)] = float(np.mean(rows))
    return out


def fits(articles: list[Article], notes: str) -> dict:
    """Paragraph size in tokens, and the share of paragraphs that fit each budget at all."""
    sizes = np.array([tokens(p) + OVERHEAD for a in articles for p in a.notes[notes]])
    return {"tokens_mean": float(sizes.mean() - OVERHEAD), "tokens_median": float(np.median(sizes) - OVERHEAD),
            **{str(b): float((sizes <= b).mean()) for b in BUDGETS}}


class plain_russian:
    """The package before Russian, as far as these questions can tell: no stems, no Russian stopwords or query shapes."""

    def __enter__(self):
        self.saved = text.russian_stem, text.STOPWORDS, {k: getattr(relevance, k) for k in ENGLISH_ONLY}
        text.russian_stem = lambda word: word
        text.STOPWORDS = frozenset(t for t in text.STOPWORDS if t.isascii())
        for name, pattern in ENGLISH_ONLY.items():
            setattr(relevance, name, re.compile(pattern, re.I))

    def __exit__(self, *_):
        text.russian_stem, text.STOPWORDS, patterns = self.saved
        for name, pattern in patterns.items():
            setattr(relevance, name, pattern)


class no_stems:
    def __enter__(self):
        self.saved = text.russian_stem
        text.russian_stem = lambda word: word

    def __exit__(self, *_):
        text.russian_stem = self.saved


MODES = {"": contextlib.nullcontext, "plain": plain_russian, "no-stems": no_stems}


def job(spec: tuple) -> dict:
    stores, notes, questions, vectors, config, mode = spec
    with MODES[mode]():
        return run_stores(stores, notes, questions, vectors, config)


def xquad_jobs(articles: list[Article], minilm: Vectors, multi: Vectors) -> dict:
    jobs: dict = {}
    for label, stores, excerpts in (("article", by_article, True), ("pooled", pooled, False)):
        data = {lang: stores(articles, lang) for lang in ("ru", "en")}
        same = (data["ru"], "ru", "ru", no_vectors)
        jobs[label, "ru-ru", "before"] = (*same, LEXICAL, "plain")
        jobs[label, "ru-ru", "stem-off"] = (*same, LEXICAL, "no-stems")
        jobs[label, "ru-ru", "stem-on"] = (*same, LEXICAL, "")
        if excerpts:
            jobs[label, "ru-ru", "stem-off-excerpts"] = (*same, LEXICAL.with_(**EXCERPTS), "no-stems")
            jobs[label, "ru-ru", "stem-on-excerpts"] = (*same, LEXICAL.with_(**EXCERPTS), "")
        for notes, questions in (("en", "ru"), ("en", "en"), ("ru", "ru")):
            pair = f"{questions}-{notes}"
            if pair != "ru-ru":
                jobs[label, pair, "lexical"] = (data[notes], notes, questions, no_vectors, LEXICAL, "")
            for name, vectors in (("minilm", minilm), ("multilingual", multi)):
                jobs[label, pair, name] = (data[notes], notes, questions, vectors, DENSE, "")
                if excerpts:
                    jobs[label, pair, f"{name}-excerpts"] = (data[notes], notes, questions, vectors, DENSE.with_(**EXCERPTS), "")
    return jobs


PAIRS = (
    ("ru-ru", "stem-on", "stem-off"), ("ru-ru", "stem-on", "before"), ("ru-ru", "stem-on-excerpts", "stem-off-excerpts"),
    ("ru-en", "multilingual", "minilm"), ("ru-en", "multilingual", "lexical"), ("ru-en", "minilm", "lexical"),
    ("en-en", "multilingual", "minilm"), ("en-en", "multilingual", "lexical"), ("en-en", "minilm", "lexical"),
    ("ru-ru", "multilingual", "stem-on"), ("ru-ru", "multilingual", "minilm"),
    ("ru-en", "multilingual-excerpts", "minilm-excerpts"), ("en-en", "multilingual-excerpts", "minilm-excerpts"),
)


class FloatModel(MultilingualEmbedder):
    """The same model before quantisation, which the shipped file is checked against."""

    model_file = "onnx/model.onnx"


def quantisation_check(articles: list[Article], multi: Vectors, pool: ProcessPoolExecutor, rng: np.random.Generator) -> dict | None:
    """Vectors and recall of the shipped quantised file against the float one, if that file is here."""
    if not (MODEL_FOLDER / FloatModel.model_file).exists():
        return None
    unique = list(multi.table)
    full = Vectors(FloatModel(MODEL_FOLDER), unique)
    cosines = np.array([float(multi.table[t][:384] @ full.table[t][:384]) for t in unique])
    out: dict = {"texts": len(unique), "cosine_mean": float(cosines.mean()), "cosine_p01": float(np.percentile(cosines, 1)),
                 "cosine_min": float(cosines.min()), "seconds_quantised": multi.seconds, "seconds_float": full.seconds, "tests": {}}
    pairs = (("ru-en", "en", "ru"), ("en-en", "en", "en"), ("ru-ru", "ru", "ru"))
    specs = [(by_article(articles, notes), notes, questions, vectors, DENSE, "") for _, notes, questions in pairs for vectors in (multi, full)]
    results = list(pool.map(job, specs))
    for k, (name, _, _) in enumerate(pairs):
        out["tests"][name] = tests(results[2 * k], results[2 * k + 1], "recall", rng)
    return out


class SampleCache:
    """A TextCache whose vectors come from another embedder. Token counts stay those of the benchmark cache."""

    def __init__(self, cache: TextCache, table_: dict[str, np.ndarray]):
        self.cache = cache
        self.table = table_

    def vector(self, s: str) -> np.ndarray:
        return self.table[digest(s)]

    def count(self, s: str) -> int:
        return self.cache.count(s)


def replay(spec: tuple) -> list[dict]:
    """As harness.run_stream, for one method and one source of vectors, without keys."""
    stream, name, table_ = spec
    cache = TextCache(name)
    if table_ is not None:
        cache = SampleCache(cache, table_)
    mem = new_memory(cache, DENSE)
    id_of: dict[str, int] = {}
    methods = {"kanerva": kanerva_variant(DENSE)}
    records: list[dict] = []
    cursor = 0
    for query in stream.queries:
        while cursor < query.pos:
            apply_write(mem, cache, stream.writes[cursor], id_of, False)
            cursor += 1
        records.extend(evaluate_query(mem, cache, query, id_of, methods, SAMPLE_BUDGETS, stream.name))
    mem.close()
    return records


def per_query(records: list[dict], budget: int) -> dict:
    return {(r["stream"], r["qid"]): r["recall"] for r in records if r["budget"] == budget}


def sample_streams(name: str) -> list[Stream]:
    """LoCoMo's test conversations, and the first of the three replays of each RepoMem repository."""
    streams = load(name)
    return [s for s in streams if s.split == "test"] if name == "locomo" else [s for s in streams if s.name.endswith("-0")]


def english_sample(embedder: MultilingualEmbedder, pool: ProcessPoolExecutor, rng: np.random.Generator) -> dict:
    out: dict = {}
    for name in ("locomo", "repomem-new"):
        streams = sample_streams(name)
        texts = sorted({t.text for s in streams for t in s.writes} | {q.question for s in streams for q in s.queries})
        start = time.perf_counter()
        vectors = {digest(t): v for t, v in zip(texts, embedder.encode(texts))}
        seconds = time.perf_counter() - start
        print(f"{name}: embedded {len(texts)} texts in {seconds:.0f}s", flush=True)
        records = {"multilingual": [], "minilm": []}
        specs = [(s, name, vectors if kind == "multilingual" else None) for kind in records for s in streams]
        for k, rows in enumerate(pool.map(replay, specs)):
            records["multilingual" if k < len(streams) else "minilm"] += rows
        results = {str(b): paired(per_query(records["multilingual"], b), per_query(records["minilm"], b), rng, name) for b in SAMPLE_BUDGETS}
        holm(list(results.values()))
        out[name] = {"streams": [s.name for s in streams], "texts_embedded": len(texts), "seconds_embedding": seconds,
                     "queries": len({(r["stream"], r["qid"]) for r in records["minilm"]}), "multilingual_minus_minilm": results}
        print(f"{name}: replayed", flush=True)
    return out


def timing_child() -> None:
    """One cold start and one query, in a process of its own, as a hook does."""
    kind = sys.argv[2]
    start = time.perf_counter()
    embedder = MultilingualEmbedder(MODEL_FOLDER) if kind == "multilingual" else OnnxEmbedder(Path(MINILM_ONNX))
    loaded = time.perf_counter()
    embedder.encode(["как запустить тесты в этом проекте?"])
    print(json.dumps({"load_s": loaded - start, "first_query_s": time.perf_counter() - loaded}))


def timings(queries: list[str]) -> dict:
    rows: dict = {}
    for kind, folder in (("multilingual", MODEL_FOLDER), ("minilm", MINILM_ONNX)):
        if folder is None:
            continue
        embedder = MultilingualEmbedder(MODEL_FOLDER) if kind == "multilingual" else OnnxEmbedder(Path(folder))
        embedder.encode(queries[:8])
        spent = []
        for query in queries:
            start = time.perf_counter()
            embedder.encode([query])
            spent.append(1000 * (time.perf_counter() - start))
        cold = []
        for _ in range(7):
            done = subprocess.run([sys.executable, "-m", "experiments.russian_test", "timing", kind], capture_output=True, text=True, check=True)
            cold.append(json.loads(done.stdout.strip().splitlines()[-1]))
        rows[kind] = {"queries": len(queries), "query_ms_p50": float(np.median(spent)), "query_ms_p95": float(np.percentile(spent, 95)),
                      "load_s_median": float(np.median([c["load_s"] for c in cold])),
                      "first_query_s_median": float(np.median([c["first_query_s"] for c in cold])), "cold_runs": len(cold)}
    return rows


def sizes() -> dict:
    """Bytes of each pinned file, as the Hub reports them, so that neither folder has to be on this machine."""
    out: dict = {}
    for name, repo, revision, files in (("minilm", MINILM_REPO, MINILM_REVISION, MINILM_FILES),
                                        ("multilingual", MULTILINGUAL_REPO, MULTILINGUAL_REVISION, MULTILINGUAL_FILES)):
        out[name] = {}
        for path in files:
            request = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/{revision}/{path}", method="HEAD")
            with urllib.request.urlopen(request) as response:
                out[name][path] = int(response.headers["Content-Length"])
        out[name]["total_mb"] = sum(out[name].values()) / 1e6
    return out


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if sys.argv[1:2] == ["timing"]:
        timing_child()
        return
    rng = np.random.default_rng(0)
    articles = read_xquad()
    texts = [p for a in articles for lang in ("ru", "en") for p in a.notes[lang]]
    texts += [q[lang] for a in articles for q in a.questions for lang in ("ru", "en")]
    embedder = MultilingualEmbedder(MODEL_FOLDER)
    multi = Vectors(embedder, texts)
    minilm = Vectors(SentenceTransformerEmbedder(), texts)
    print(f"embedded {len(multi.table)} texts: multilingual {multi.seconds:.0f}s, all-MiniLM-L6-v2 {minilm.seconds:.0f}s", flush=True)
    source = "https://github.com/google-deepmind/xquad"
    checksums = {lang: hashlib.sha256((ROOT / "data" / "xquad" / f"xquad.{lang}.json").read_bytes()).hexdigest() for lang in ("ru", "en")}
    report: dict = {"meta": {"experiment": "Russian and cross-language search", "commit": git_commit(), "model_file": MultilingualEmbedder.model_file,
                             "dataset": {"source": source, "revision": "7d30520c717524000f0d9d2f9c10a069acd9d285", "sha256": checksums},
                             "budgets": BUDGETS, "excerpts": EXCERPTS, "workers": WORKERS}}
    report["xquad"] = {"articles": len(articles), "notes_per_store": 5, "questions": sum(len(a.questions) for a in articles),
                       "paragraphs": {lang: fits(articles, lang) for lang in ("ru", "en")},
                       "random_order": {lang: random_order(articles, lang, rng) for lang in ("ru", "en")}}
    with ProcessPoolExecutor(WORKERS) as pool:
        jobs = xquad_jobs(articles, minilm, multi)
        runs = dict(zip(jobs, pool.map(job, jobs.values())))
        print("xquad done", flush=True)
        for store in ("article", "pooled"):
            runs[store, "ru-ru", "lexical"] = runs[store, "ru-ru", "stem-on"]
        report["recall"] = {f"{store} {pair} {method}": table(run) for (store, pair, method), run in runs.items()}
        report["tests"] = {}
        for store in ("article", "pooled"):
            for pair, a, b in PAIRS:
                if (store, pair, a) in runs and (store, pair, b) in runs:
                    report["tests"][f"{store} {pair}: {a} vs {b}"] = {
                        metric: tests(runs[store, pair, a], runs[store, pair, b], metric, rng) for metric in ("recall", "answer")}
        report["quantised"] = quantisation_check(articles, multi, pool, rng)
        print("quantisation check done", flush=True)
        report["english_sample"] = english_sample(embedder, pool, rng)
    english = [q["en"] for a in articles for q in a.questions][:100]
    russian = [q["ru"] for a in articles for q in a.questions][:100]
    report["timing"] = timings(english + russian)
    report["size"] = sizes()
    (RESULTS / "russian.json").write_text(json.dumps(report, indent=1))
    print("wrote results/russian.json, commit", git_commit())


if __name__ == "__main__":
    sys.exit(main())
