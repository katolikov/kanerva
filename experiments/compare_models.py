"""Offline comparison of relevance models and selection rules on the dumped development pools.

Models are cross-fitted: each query is scored by a model that never saw its group (question, conversation or
repository). Training pools all datasets with equal weight per dataset and per query.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from experiments.embed_cache import CACHE
from experiments.harness import BUDGETS, RESULTS, git_commit
from kanerva.relevance import FEATURES, RERANK_FEATURES

FOLDS = 5


class Data:
    def __init__(self, name: str):
        d = np.load(CACHE / f"features_{name}.npz")
        self.name = name
        self.offsets = d["offsets"]
        for key in ("x1", "x2", "gold", "stale", "cost", "answer", "nans", "ngold", "old", "qid", "category", "stream", "split"):
            setattr(self, key, d[key])
        self.n = len(self.offsets) - 1
        groups = [self.group(s) for s in self.stream]
        self.fold = np.array([int(hashlib.sha1(g.encode()).hexdigest(), 16) % FOLDS for g in groups])
        sizes = np.diff(self.offsets)
        self.row_query = np.repeat(np.arange(self.n), sizes)

    def group(self, stream: str) -> str:
        if self.name.startswith("repomem"):
            return stream.rsplit("-", 1)[0]  # all seeds of one repository share a fold
        return stream

    def rows(self, q: int) -> slice:
        return slice(self.offsets[q], self.offsets[q + 1])


def fit_logistic(x: np.ndarray, y: np.ndarray, w: np.ndarray, c: float = 1.0):
    from sklearn.linear_model import LogisticRegression

    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1.0
    model = LogisticRegression(C=c, max_iter=2000)
    model.fit((x - mean) / scale, y, sample_weight=w * len(w) / w.sum())
    return lambda z: model.predict_proba((z - mean) / scale)[:, 1], (model, mean, scale)


DENSE = {"dense_rank", "dense_score", "rank_agreement", "dense_top", "neighbour_dense", "neighbour2_dense", "session_dense", "dense_gap"}


def fit_gbdt(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    model.fit(x, y, sample_weight=w * len(w) / w.sum())
    return lambda z: model.predict_proba(z)[:, 1], model


def columns(use: tuple[str, ...]) -> list[int]:
    return [FEATURES.index(f) for f in use]


def training_weights(data: list[Data], masks: list[np.ndarray]) -> list[np.ndarray]:
    """Each dataset weighs the same in total, and each query the same within its dataset."""
    out = []
    for d, mask in zip(data, masks):
        sizes = np.diff(d.offsets).astype(float)
        per_row = 1.0 / sizes[d.row_query]
        w = np.where(mask[d.row_query], per_row, 0.0)
        out.append(w / max(w.sum(), 1e-12))
    return out


def in_pool(d: Data, pool: str) -> np.ndarray:
    return d.x2[:, FEATURES.index("lex_top")] > 0 if pool == "lexical" else np.ones(len(d.gold), dtype=bool)


def cross_fit(data: list[Data], use: tuple[str, ...], kind: str = "logistic", c: float = 1.0, pool: str = "all") -> list[np.ndarray]:
    """Out-of-fold scores for every candidate row of every dataset; rows outside ``pool`` score -1."""
    cols = columns(use)
    preds = [np.full(len(d.gold), -1.0) for d in data]
    for fold in range(FOLDS):
        train = [d.fold != fold for d in data]
        weights = training_weights(data, train)
        keep = [(w > 0) & in_pool(d, pool) for w, d in zip(weights, data)]
        x = np.vstack([d.x2[k][:, cols] for d, k in zip(data, keep)])
        y = np.concatenate([d.gold[k] for d, k in zip(data, keep)])
        w = np.concatenate([wt[k] for wt, k in zip(weights, keep)])
        if kind == "logistic":
            model = fit_logistic(x, y, w, c)[0]
        else:
            model = fit_gbdt(x, y, w)[0]
        for d, p in zip(data, preds):
            test = (d.fold == fold)[d.row_query] & in_pool(d, pool)
            if test.any():
                p[test] = model(d.x2[test][:, cols])
    return preds


def knapsack(p: np.ndarray, cost: np.ndarray, budget: int, top: int = 60) -> np.ndarray:
    """Indices maximising sum(p) subject to sum(cost) <= budget, by dynamic programming over capacity.

    Only the ``top`` items by p enter the program; the rest of the budget is then filled by density.
    """
    keep = np.argsort(-p, kind="stable")[:top]
    chosen = keep[_knapsack(p[keep], cost[keep], budget)]
    spent = int(cost[chosen].sum())
    rest = np.setdiff1d(np.arange(len(p)), chosen)
    extra = fill(rest[np.argsort(-p[rest] / cost[rest], kind="stable")], cost, budget - spent)
    return np.concatenate([chosen, extra]).astype(int)


def _knapsack(p: np.ndarray, cost: np.ndarray, budget: int) -> np.ndarray:
    n = len(p)
    best = np.zeros(budget + 1)
    take = np.zeros((n, budget + 1), dtype=bool)
    for i in range(n):
        c = int(cost[i])
        if c > budget or p[i] <= 0:
            continue
        cand = np.full(budget + 1, -1.0)
        cand[c:] = best[: budget + 1 - c] + p[i]
        better = cand > best
        take[i] = better
        best = np.where(better, cand, best)
    chosen, cap = [], budget
    for i in range(n - 1, -1, -1):
        if take[i, cap]:
            chosen.append(i)
            cap -= int(cost[i])
    return np.array(chosen[::-1], dtype=int)


def fill(order: np.ndarray, cost: np.ndarray, budget: int) -> np.ndarray:
    chosen, spent = [], 0
    for i in order:
        if spent + cost[i] <= budget:
            chosen.append(i)
            spent += cost[i]
    return np.array(chosen, dtype=int)


def density(p: np.ndarray, cost: np.ndarray, budget: int) -> np.ndarray:
    greedy = fill(np.argsort(-p / cost, kind="stable"), cost, budget)
    fits = np.flatnonzero(cost <= budget)
    if fits.size:
        single = fits[np.argmax(p[fits])]
        if p[single] > p[greedy].sum():
            return np.array([single])
    return greedy


def select(rule: str, p: np.ndarray, cost: np.ndarray, budget: int) -> np.ndarray:
    """Candidates with a negative score are outside the pool and never selected."""
    if (p < 0).any():
        inside = np.flatnonzero(p >= 0)
        return inside[select(rule, p[inside], cost[inside], budget)]
    if rule == "rank":
        return fill(np.argsort(-p, kind="stable"), cost, budget)
    if rule == "density":
        return density(p, cost, budget)
    if rule == "dp":
        return knapsack(p, cost, budget)
    raise ValueError(rule)


def rrf(d: Data, rows: slice) -> np.ndarray:
    x = d.x2[rows]
    with np.errstate(divide="ignore"):
        lex = np.where(x[:, 7] > 0, 1.0 / x[:, 7] - 1.0, np.inf)
        dense = np.where(x[:, 8] > 0, 1.0 / x[:, 8] - 1.0, np.inf)
    return 1.0 / (61.0 + lex) + 1.0 / (61.0 + dense)


EVAL_CAP = 1500


def eval_queries(d: Data) -> np.ndarray:
    """All queries, or a fixed random subset of EVAL_CAP for the largest datasets."""
    if d.n <= EVAL_CAP:
        return np.arange(d.n)
    return np.sort(np.random.default_rng(0).choice(d.n, EVAL_CAP, replace=False))


def evaluate(d: Data, scores: np.ndarray | None, rule: str, baseline: str | None = None) -> dict:
    """Recall and answer-term recall per budget, averaged over the evaluated queries."""
    queries = eval_queries(d)
    recall = np.full((d.n, len(BUDGETS)), np.nan)
    answer = np.full((d.n, len(BUDGETS)), np.nan)
    for q in queries:
        rows = d.rows(q)
        cost = d.cost[rows]
        if baseline == "hybrid":
            order = np.argsort(-rrf(d, rows), kind="stable")
        elif baseline == "bm25":
            order = np.argsort(-d.x2[rows][:, FEATURES.index("lex_top")], kind="stable")
            order = order[d.x2[rows][order, FEATURES.index("lex_top")] > 0]
        elif baseline == "rerank":
            x = d.x2[rows]
            fused = rrf(d, rows)
            order = np.lexsort((-fused, -x[:, 32]))  # cross-encoder order on its pool, then the hybrid order
        for b, budget in enumerate(BUDGETS):
            if baseline is not None:
                chosen = fill(order, cost, budget)
            else:
                chosen = select(rule, scores[rows], cost, budget)
            recall[q, b] = d.gold[rows][chosen].sum() / d.ngold[q]
            if d.nans[q]:
                mask = np.bitwise_or.reduce(d.answer[rows][chosen]) if chosen.size else 0
                answer[q, b] = bin(int(mask)).count("1") / d.nans[q]
    return {"recall": recall[queries], "answer": answer[queries]}


def summary(result: dict) -> tuple[list[float], float]:
    per_budget = result["recall"].mean(0).tolist()
    return per_budget, float(np.mean(per_budget))


BASE = tuple(f for f in FEATURES if f not in RERANK_FEATURES)
LEXICAL = tuple(f for f in BASE if f not in DENSE)
MODELS = {
    "kanerva": ("gbdt", BASE, "all"),
    "kanerva-ce": ("gbdt", BASE + RERANK_FEATURES, "all"),
    "kanerva-lex": ("gbdt", LEXICAL, "lexical"),
    "logistic": ("logistic", BASE, "all"),
}


def main(names: list[str], tag: str, models: tuple[str, ...], rules: tuple[str, ...]) -> None:
    data = [Data(n) for n in names]
    report: dict = {}
    scores = {name: cross_fit(data, MODELS[name][1], MODELS[name][0], pool=MODELS[name][2]) for name in models}
    for index, d in enumerate(data):
        rows = {}
        old = d.old[eval_queries(d)]
        rows["first-version"] = (old.mean(0).tolist(), float(old.mean()))
        for baseline in ("hybrid", "rerank", "bm25"):
            rows[baseline] = summary(evaluate(d, None, "", baseline))
        for name in models:
            for rule in rules:
                rows[f"{name}/{rule}"] = summary(evaluate(d, scores[name][index], rule))
        report[d.name] = rows
        print(f"== {d.name} ({d.n} queries)")
        print("method".ljust(26) + "".join(f"{b:>7}" for b in BUDGETS) + "   mean")
        for name, (per, mean) in rows.items():
            print(name.ljust(26) + "".join(f"{v:7.3f}" for v in per) + f"  {mean:.3f}")
    meta = {"experiment": "cross-fitted model comparison", "folds": FOLDS, "commit": git_commit()}
    (RESULTS / f"dev_models{tag}.json").write_text(json.dumps({"meta": meta, "results": report}, indent=1))


if __name__ == "__main__":
    main(["lme", "locomo", "repomem", "beam-dev"], "", ("kanerva", "kanerva-ce", "kanerva-lex"), ("density",))
    main(["lme-m2v", "locomo-m2v", "repomem-m2v", "beam-dev-m2v"], "_static", ("kanerva",), ("density",))
    main(["lme", "locomo", "repomem", "beam-dev"], "_rules", ("kanerva",), ("density", "rank", "dp"))
