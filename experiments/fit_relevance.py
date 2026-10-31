"""Fit the shipped relevance models and write src/kanerva/relevance_model.json.

Three models: ``dense`` (hybrid candidates), ``lexical`` (FTS5 candidates only, no embedding features) and
``rerank`` (with cross-encoder features), trained as the winning recipe in experiments/refit.py says.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from experiments.compare_models import BASE, LEXICAL, Data, columns
from experiments.refit import RECIPES, fit, training_rows
from experiments.tree_export import export
from kanerva.relevance import MODEL_PATH, RERANK_FEATURES

RECIPE = RECIPES["all-families"]
SPECS = {"dense": (BASE, "all"), "lexical": (LEXICAL, "lexical"), "rerank": (BASE + RERANK_FEATURES, "all")}


def main() -> None:
    data = {name: Data(name) for name in RECIPE.train}
    out = json.loads(MODEL_PATH.read_text())  # the static model keeps its own dumps
    rng = np.random.default_rng(0)
    for name, (use, pool) in SPECS.items():
        model = fit(RECIPE, *training_rows(data, RECIPE, -1, use, pool, rng))
        tree = export(model, use)
        check = next(iter(data.values())).x2[:2000]
        assert np.allclose(tree.predict(check), model.predict_proba(check[:, columns(use)])[:, 1], atol=1e-9)
        out[name] = tree.to_json() | {"trained_on": list(RECIPE.train), "trees": int(tree.feature.shape[0])}
        print(name, "trees", tree.feature.shape[0], "depth", tree.depth, flush=True)
    MODEL_PATH.write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", MODEL_PATH, f"{MODEL_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
