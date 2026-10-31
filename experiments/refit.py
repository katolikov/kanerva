"""How the relevance model should be trained, by leave-repository-out recall on everything except the final test.

Each recipe below is scored out of fold: a repository or conversation is only scored by a model that never saw it.
Negatives are subsampled and reweighted so that a recipe takes minutes. Writes results/refit.json.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from experiments.compare_models import BASE, FOLDS, LEXICAL, Data, columns, evaluate, in_pool
from experiments.harness import BUDGETS, RESULTS, git_commit

CHATS = ("lme", "locomo", "beam-dev")
KEYED = ("repomem", "repomem-holdout", "repomem-fresh", "repomem-new")
PLAIN = ("repomem-plain", "repomem-holdout-plain", "repomem-fresh-plain", "repomem-new-plain", "repomem-other")
C_FAMILY = ("repomem-c-dev-plain", "repomem-c-test-plain", "repomem-c-dev", "repomem-c-test")
SCORED = CHATS + PLAIN + KEYED + C_FAMILY
KEEP_NEGATIVES = 0.2


@dataclass(frozen=True)
class Recipe:
    train: tuple[str, ...]
    leaves: int = 31
    min_leaf: int = 20
    l2: float = 1.0
    rounds: int = 300
    rate: float = 0.05
    families: bool = False  # chats and code weigh the same, however many sets of each there are


RECIPES = {
    "as-shipped": Recipe(CHATS + ("repomem",)),
    "add-plain": Recipe(CHATS + ("repomem", "repomem-plain")),
    "all": Recipe(CHATS + PLAIN + KEYED),
    "all-families": Recipe(CHATS + PLAIN + KEYED, families=True),
    "with-c": Recipe(CHATS + PLAIN + KEYED + C_FAMILY, families=True),
    "all-small": Recipe(CHATS + PLAIN + KEYED, leaves=15, min_leaf=100, families=True),
    "all-smaller": Recipe(CHATS + PLAIN + KEYED, leaves=7, min_leaf=200, l2=5.0, families=True),
}


def family_share(name: str, recipe: Recipe, names: tuple[str, ...]) -> float:
    if not recipe.families:
        return 1.0
    return 1.0 / sum(n.startswith("repomem") == name.startswith("repomem") for n in names)


def training_rows(data: dict[str, Data], recipe: Recipe, fold: int, use: tuple[str, ...], pool: str, rng):
    cols = columns(use)
    xs, ys, ws = [], [], []
    for name in recipe.train:
        d = data[name]
        keep = (d.fold != fold)[d.row_query] & in_pool(d, pool)
        sizes = np.diff(d.offsets).astype(float)
        weight = 1.0 / sizes[d.row_query]
        weight = weight / weight[keep].sum() * family_share(name, recipe, recipe.train)
        sampled = d.gold | (rng.random(len(d.gold)) < KEEP_NEGATIVES)
        keep &= sampled
        weight = np.where(d.gold, weight, weight / KEEP_NEGATIVES)
        xs.append(d.x2[keep][:, cols])
        ys.append(d.gold[keep])
        ws.append(weight[keep])
    w = np.concatenate(ws)
    return np.vstack(xs), np.concatenate(ys), w * len(w) / w.sum()


def fit(recipe: Recipe, x: np.ndarray, y: np.ndarray, w: np.ndarray):
    model = HistGradientBoostingClassifier(max_iter=recipe.rounds, learning_rate=recipe.rate, max_leaf_nodes=recipe.leaves,
                                           min_samples_leaf=recipe.min_leaf, l2_regularization=recipe.l2, random_state=0)
    return model.fit(x, y, sample_weight=w)


def out_of_fold(data: dict[str, Data], recipe: Recipe, use: tuple[str, ...], pool: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    cols = columns(use)
    scores = {name: np.full(len(d.gold), -1.0) for name, d in data.items()}
    for fold in range(FOLDS):
        model = fit(recipe, *training_rows(data, recipe, fold, use, pool, rng))
        for name, d in data.items():
            test = (d.fold == fold)[d.row_query] & in_pool(d, pool)
            if test.any():
                scores[name][test] = model.predict_proba(d.x2[test][:, cols])[:, 1]
    return scores


def recall(d: Data, scores: np.ndarray | None, baseline: str | None = None) -> list[float]:
    result = evaluate(d, scores, "density", baseline)
    return [round(float(v), 4) for v in result["recall"].mean(0)]


def main(names: list[str], kind: str) -> None:
    use, pool = (BASE, "all") if kind == "dense" else (LEXICAL, "lexical")
    data = {name: Data(name) for name in SCORED}
    path = RESULTS / ("refit_c.json" if "with-c" in names else "refit.json")
    report = json.loads(path.read_text()) if path.exists() else {}
    report["meta"] = {"experiment": "refit", "commit": git_commit(), "budgets": list(BUDGETS)}
    section = report.setdefault(kind, {"baseline": {}, "recipes": {}})
    baseline = "hybrid" if kind == "dense" else "bm25"
    for name, d in data.items():
        section["baseline"][name] = recall(d, None, baseline)
    for label in names:
        scores = out_of_fold(data, RECIPES[label], use, pool)
        section["recipes"][label] = {name: recall(d, scores[name]) for name, d in data.items()}
        path.write_text(json.dumps(report, indent=1))
        line = {name: round(float(np.mean(v)), 3) for name, v in section["recipes"][label].items()}
        print(kind, label, line, flush=True)


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "dense"
    sys.exit(main(sys.argv[2:] or [name for name in RECIPES if name != "with-c"], kind))
