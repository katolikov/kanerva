"""Which feature groups carry the gain: cross-fitted on development data, with one group removed at a time."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from experiments.harness import RESULTS
from experiments.r2_offline import BASE, Data, cross_fit, evaluate, summary

GROUPS = {
    "item": ("log_tokens", "short", "content_density", "is_question", "has_number"),
    "role": ("role_user", "role_assistant", "first_person_user", "speaker_in_query"),
    "query-item": ("entity_overlap", "number_overlap", "when_date", "count_number"),
    "neighbourhood": ("neighbour_dense", "neighbour2_dense", "session_dense", "session_hits"),
    "time": ("recency", "temporal_recency", "activation"),
}


def main(names: list[str]) -> None:
    data = [Data(n) for n in names]
    report = {"all": BASE} | {f"-{g}": tuple(f for f in BASE if f not in fs) for g, fs in GROUPS.items()}
    out: dict = {}
    for label, use in report.items():
        preds = cross_fit(data, use, "gbdt")
        out[label] = {d.name: summary(evaluate(d, p, "density")) for d, p in zip(data, preds)}
        print(label.ljust(16) + "  ".join(f"{d.name} {out[label][d.name][1]:.3f}" for d in data), flush=True)
    base = {d.name: out["all"][d.name][1] for d in data}
    print("mean drop:", {k: round(float(np.mean([base[n] - v[n][1] for n in base])), 4) for k, v in out.items()})
    (RESULTS / "r2_ablation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("R2_SUFFIX", "-wide")
    main(sys.argv[1:] or ["lme", "locomo", "repomem", "beam-dev"])
