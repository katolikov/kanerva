"""How well the cross-fitted probabilities match observed rates, on development data."""

from __future__ import annotations

import json

import numpy as np

from experiments.compare_models import BASE, Data, cross_fit
from experiments.harness import RESULTS, git_commit

BINS = 10


def reliability(p: np.ndarray, y: np.ndarray) -> dict:
    """Expected calibration error over equal-count bins, and the top bin, where selection happens."""
    order = np.argsort(p)
    rows = []
    for chunk in np.array_split(order, BINS):
        rows.append({"predicted": float(p[chunk].mean()), "observed": float(y[chunk].mean()), "n": int(chunk.size)})
    ece = sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in rows) / len(p)
    top = order[int(0.99 * len(order)):]
    return {"ece": float(ece), "base_rate": float(y.mean()), "bins": rows,
            "top_percent": {"predicted": float(p[top].mean()), "observed": float(y[top].mean())}}


def main() -> None:
    data = [Data(n) for n in ("lme", "locomo", "repomem", "beam-dev")]
    preds = cross_fit(data, BASE, "gbdt")
    results = {d.name: reliability(p, d.gold.astype(float)) for d, p in zip(data, preds)}
    (RESULTS / "dev_calibration.json").write_text(json.dumps({"meta": {"commit": git_commit()}, "results": results}, indent=1))
    for name, row in results.items():
        print(name, round(row["ece"], 4), row["top_percent"], round(row["base_rate"], 4))


if __name__ == "__main__":
    main()
