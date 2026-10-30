"""Fit the shipped relevance models on all development dumps and write src/kanerva/relevance_model.json.

Three models: ``dense`` (hybrid candidates), ``lexical`` (FTS5 candidates only, no embedding features) and
``rerank`` (with cross-encoder features). Each dataset weighs the same, and each query the same within it.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from experiments.r2_offline import BASE, DENSE, Data, columns, fit_gbdt, in_pool, training_weights
from experiments.tree_export import export
from kanerva.relevance import MODEL_PATH, RERANK_FEATURES

DEV = ("lme", "locomo", "repomem", "beam-dev")
LEXICAL = tuple(f for f in BASE if f not in DENSE)
SPECS = {"dense": (BASE, "all"), "lexical": (LEXICAL, "lexical"), "rerank": (BASE + RERANK_FEATURES, "all")}


def fit(data: list[Data], use: tuple[str, ...], pool: str):
    weights = training_weights(data, [np.ones(d.n, dtype=bool) for d in data])
    keep = [in_pool(d, pool) & (w > 0) for d, w in zip(data, weights)]
    cols = columns(use)
    x = np.vstack([d.x2[k][:, cols] for d, k in zip(data, keep)])
    y = np.concatenate([d.gold[k] for d, k in zip(data, keep)])
    w = np.concatenate([wt[k] for wt, k in zip(weights, keep)])
    return fit_gbdt(x, y, w, use)[1]


def main(names: tuple[str, ...]) -> None:
    data = [Data(n) for n in names]
    out = {}
    for name, (use, pool) in SPECS.items():
        model = fit(data, use, pool)
        tree = export(model, use)
        check = data[0].x2[:2000]
        assert np.allclose(tree.predict(check), model.predict_proba(check[:, columns(use)])[:, 1], atol=1e-9)
        out[name] = tree.to_json() | {"trained_on": list(names), "suffix": os.environ.get("R2_SUFFIX", ""),
                                      "trees": int(tree.feature.shape[0])}
        print(name, "trees", tree.feature.shape[0], "depth", tree.depth, flush=True)
    MODEL_PATH.write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", MODEL_PATH, f"{MODEL_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or DEV)
