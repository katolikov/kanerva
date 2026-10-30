"""Statistics, tables and figures from results/. All paper numbers are read from the JSON this writes."""

from __future__ import annotations

import gzip
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from experiments.datasets import ROOT

RESULTS = ROOT / "results"
FIGS = ROOT / "paper" / "figs"
BUDGETS = (128, 256, 512, 1024, 2048, 4096)
DATASETS = {"lme": "LongMemEval-S", "locomo": "LoCoMo", "repomem-keyed": "RepoMem (keys)", "repomem-plain": "RepoMem (no keys)",
            "repomem-holdout": "RepoMem holdout (keys)"}
BASELINES = ("hybrid", "bm25", "dense", "mmr", "facility", "gen-agents", "recency", "rerank", "hybrid-live")
ABLATIONS = ("-density", "-coverage", "-redundancy", "-validity", "-supersession", "-activation", "kanerva-first", "kanerva-ce")
BOOT = 2000

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e0"
STYLE = {
    "kanerva-ce": ("#2a78d6", "o", 2.4),
    "kanerva": ("#1baf7a", "D", 2.0),
    "hybrid": ("#eb6834", "s", 1.8),
    "rerank": ("#e87ba4", "v", 1.8),
    "bm25": ("#eda100", "^", 1.8),
    "mmr": ("#008300", "P", 1.8),
    "facility": ("#6250d6", "X", 1.8),
}
LABEL = {"kanerva": "Kanerva", "kanerva-ce": "Kanerva + reranker", "hybrid": "Hybrid RRF", "rerank": "Hybrid + reranker", "bm25": "BM25", "dense": "Dense",
         "mmr": "MMR", "facility": "Facility location"}


def load_raw(name: str) -> list[dict]:
    records = json.load(gzip.open(RESULTS / "raw" / f"retrieval_{name}.json.gz", "rt"))
    for prefix in ("rerank", "ce"):
        extra = RESULTS / "raw" / f"{prefix}_{name}.json.gz"
        if extra.exists():
            records += json.load(gzip.open(extra, "rt"))
    return records


def pivot(records: list[dict], metric: str) -> dict:
    table: dict = {}
    for r in records:
        table.setdefault((r["method"], r["budget"]), {})[(r["stream"], r["qid"])] = r[metric]
    return table


def paired(a: dict, b: dict, rng: np.random.Generator) -> dict:
    keys = sorted(set(a) & set(b))
    x = np.array([a[k] for k in keys], dtype=float)
    y = np.array([b[k] for k in keys], dtype=float)
    d = x - y
    n = len(d)
    samples = d[rng.integers(0, n, size=(BOOT, n))].mean(axis=1)
    clusters = sorted({k[0].rsplit("-", 1)[0] for k in keys})
    cluster_ci = None
    if 3 <= len(clusters) <= 30:
        groups = {c: d[[i for i, k in enumerate(keys) if k[0].rsplit("-", 1)[0] == c]] for c in clusters}
        boots = []
        for _ in range(BOOT):
            picks = rng.integers(0, len(clusters), size=len(clusters))
            boots.append(np.concatenate([groups[clusters[p]] for p in picks]).mean())
        cluster_ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    p = float(wilcoxon(d, zero_method="zsplit").pvalue) if np.any(d != 0) else 1.0
    return {"mean_a": float(x.mean()), "mean_b": float(y.mean()), "diff": float(d.mean()),
            "ci": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
            "cluster_ci": cluster_ci, "p": p, "n": n}


def holm(entries: list[dict]) -> None:
    order = sorted(range(len(entries)), key=lambda i: entries[i]["p"])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, entries[i]["p"] * (len(order) - rank)))
        entries[i]["p_holm"] = running


def compare() -> dict:
    rng = np.random.default_rng(0)
    out: dict = {}
    for name in DATASETS:
        if not (RESULTS / "raw" / f"retrieval_{name}.json.gz").exists():
            continue
        records = load_raw(name)
        recall = pivot(records, "recall")
        tokens = pivot(records, "tokens")
        stale = pivot(records, "stale_error")
        entry: dict = {"means": {}, "comparisons": {}, "ablations": {}, "tokens": {}, "stale_error": {}, "pareto": {}}
        methods = sorted({m for m, _ in recall})
        for m in methods:
            for b in sorted({b for mm, b in recall if mm == m}):
                entry["means"][f"{m}@{b}"] = float(np.mean(list(recall[(m, b)].values())))
                entry["tokens"][f"{m}@{b}"] = float(np.mean(list(tokens[(m, b)].values())))
                entry["stale_error"][f"{m}@{b}"] = float(np.mean(list(stale[(m, b)].values())))
        for group, members, key in ((BASELINES, "comparisons", "comparisons"), (ABLATIONS, "ablations", "ablations")):
            tests = []
            for other in group:
                for b in BUDGETS:
                    if (other, b) in recall and ("kanerva", b) in recall:
                        result = paired(recall[("kanerva", b)], recall[(other, b)], rng)
                        result["other"] = other
                        result["budget"] = b
                        tests.append(result)
            holm(tests)
            entry[key] = tests
        for other in BASELINES:
            tests = [t for t in entry["comparisons"] if t["other"] == other]
            wins = sum(t["diff"] > 0 and t["p_holm"] < 0.05 for t in tests)
            losses = sum(t["diff"] < 0 and t["p_holm"] < 0.05 for t in tests)
            entry["pareto"][other] = {"wins": wins, "losses": losses, "ties": len(tests) - wins - losses}
        categories = sorted({r["category"] for r in records})
        by_category: dict = {}
        for c in categories:
            for m in ("kanerva", "kanerva-ce", "hybrid", "rerank", "bm25"):
                rows = [r["recall"] for r in records if r["category"] == c and r["method"] == m and r["budget"] == 1024]
                if rows:
                    by_category.setdefault(c, {})[m] = float(np.mean(rows))
            by_category[c]["n"] = sum(1 for r in records if r["category"] == c and r["method"] == "kanerva" and r["budget"] == 1024)
        entry["by_category_1024"] = by_category
        latency = [r["latency_ms"] for r in records if r["method"] == "kanerva"]
        entry["latency_ms"] = {"p50": float(np.percentile(latency, 50)), "p95": float(np.percentile(latency, 95))}
        out[name] = entry
    (RESULTS / "comparison.json").write_text(json.dumps(out, indent=1))
    return out


def full_history() -> dict:
    from experiments.embed_cache import TextCache
    from experiments.registry import load

    out = {}
    for name in ("lme", "locomo"):
        cache = TextCache(name)
        out[name] = float(np.mean([sum(cache.count(t.text) + 8 for t in s.writes) for s in load(name) if s.split == "test"]))
    (RESULTS / "full_history.json").write_text(json.dumps({"meta": {"commit": "derived from data"}, "tokens": out}, indent=1))
    return out


def setup() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "font.family": "sans-serif", "pdf.fonttype": 42,
    })


def pareto_figure(data: dict) -> None:
    names = [n for n in ("lme", "locomo", "repomem-keyed") if n in data]
    fig, axes = plt.subplots(1, len(names), figsize=(7.1, 2.5), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        means, tokens = data[name]["means"], data[name]["tokens"]
        for method, (color, marker, width) in reversed(list(STYLE.items())):
            points = [(tokens[f"{method}@{b}"], means[f"{method}@{b}"]) for b in BUDGETS if f"{method}@{b}" in means]
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(xs, ys, color=color, marker=marker, markersize=3.5, linewidth=width, label=LABEL[method],
                    markeredgecolor="white", markeredgewidth=0.4, zorder=3 if method.startswith("kanerva") else 2)
        topk = [(tokens[f"hybrid-k{k}@{k}"], means[f"hybrid-k{k}@{k}"]) for k in (3, 5, 10, 20) if f"hybrid-k{k}@{k}" in means]
        if topk:
            ax.scatter(*zip(*topk), s=14, facecolors="none", edgecolors=MUTED, linewidths=0.8, label="Hybrid top-$k$ (3, 5, 10, 20)", zorder=2)
        ax.set_xscale("log", base=2)
        ax.set_xticks([128, 512, 2048, 8192])
        ax.set_xticklabels(["128", "512", "2k", "8k"])
        ax.set_xlabel("Tokens injected per query")
        ax.set_title(DATASETS[name], fontsize=8.5, color=INK)
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("Evidence recall")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIGS / "pareto.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "pareto.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def online_figure() -> None:
    path = RESULTS / "raw" / "online.json"
    if not path.exists():
        return
    runs = json.loads(path.read_text())
    datasets = [d for d in ("lme", "locomo", "repomem") if any(r["dataset"] == d for r in runs)]
    colors = {"static": "#eb6834", "online": "#2a78d6", "cold-static": "#eda100", "cold-online": "#1baf7a"}
    labels = {"static": "Fitted prior, static", "online": "Fitted prior, online", "cold-static": "Flat prior, static", "cold-online": "Flat prior, online"}
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.1, 2.2))
    for ax, dataset in zip(np.atleast_1d(axes), datasets):
        for setting, color in reversed(list(colors.items())):
            series = [np.array(r["recall"]) for r in runs if r["dataset"] == dataset and r["setting"] == setting]
            if not series:
                continue
            n = min(len(s) for s in series)
            window = max(n // 10, 1)
            smooth = np.array([np.convolve(s[:n], np.ones(window) / window, mode="valid") for s in series])
            x = np.arange(smooth.shape[1]) + window
            ax.plot(x, smooth.mean(axis=0), color=color, linewidth=1.8, label=labels[setting])
            ax.fill_between(x, smooth.min(axis=0), smooth.max(axis=0), color=color, alpha=0.12, linewidth=0)
        ax.set_title({"lme": "LongMemEval-S", "locomo": "LoCoMo", "repomem": "RepoMem"}[dataset], fontsize=8.5, color=INK)
        ax.set_xlabel("Queries seen")
    np.atleast_1d(axes)[0].set_ylabel("Recall at $B=512$ (moving mean)")
    handles, labels_ = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels_[::-1], loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGS / "online.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    setup()
    data = compare()
    full_history()
    if data:
        pareto_figure(data)
    online_figure()
    for name, entry in data.items():
        print(name, {k: round(v, 3) for k, v in entry["means"].items() if k.startswith(("kanerva@", "hybrid@", "rerank@"))})
        print(" pareto", entry["pareto"])


if __name__ == "__main__":
    main()
