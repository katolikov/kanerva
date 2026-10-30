"""Convert a fitted scikit-learn HistGradientBoostingClassifier into kanerva's dependency-free TreeModel."""

from __future__ import annotations

import numpy as np

from kanerva.relevance import TreeModel


def export(model, use: tuple[str, ...]) -> TreeModel:
    trees = [predictors[0].nodes for predictors in model._predictors]
    width = max(len(t) for t in trees)
    shape = (len(trees), width)
    feature = np.zeros(shape, dtype=np.int64)
    threshold = np.zeros(shape)
    left = np.tile(np.arange(width), (len(trees), 1))
    right = left.copy()
    value = np.zeros(shape)
    depth = 0
    for t, nodes in enumerate(trees):
        for i, node in enumerate(nodes):
            if node["is_leaf"]:
                value[t, i] = node["value"]
                threshold[t, i] = np.inf
            else:
                feature[t, i] = node["feature_idx"]
                threshold[t, i] = node["num_threshold"]
                left[t, i], right[t, i] = node["left"], node["right"]
            depth = max(depth, int(node["depth"]))
    baseline = float(np.ravel(model._baseline_prediction)[0])
    return TreeModel(feature, threshold, left, right, value, baseline, depth + 1, use)
