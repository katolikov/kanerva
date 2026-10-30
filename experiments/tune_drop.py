"""Dev-only choice of the near-duplicate removal threshold for text without keys."""

from __future__ import annotations

import numpy as np

from experiments.harness import run_dataset
from experiments.registry import load
from kanerva import RetrievalConfig


def main() -> None:
    plain = [s for s in load("repomem") if s.split == "dev"]
    updates = [s for s in load("lme-updates") if s.split == "dev"]
    for drop in (0.90, 0.93, 0.95, 0.97, 0.99, 1.01):
        cfg = RetrievalConfig(supersession_drop=drop)
        rows = run_dataset(plain, "repomem", cfg, selected={"kanerva"}, budgets=(256, 1024))
        changed = [r for r in rows if "changed" in r["category"]]
        nl = run_dataset(updates, "lme", cfg, selected={"kanerva"}, budgets=(1024,))
        print(drop, "recall@1024", round(float(np.mean([r["recall"] for r in rows if r["budget"] == 1024])), 4),
              "recall@256", round(float(np.mean([r["recall"] for r in rows if r["budget"] == 256])), 4),
              "inject", round(float(np.mean([r["stale_any"] for r in changed if r["budget"] == 1024])), 3),
              "error", round(float(np.mean([r["stale_error"] for r in changed if r["budget"] == 1024])), 3),
              "| updates recall", round(float(np.mean([r["recall"] for r in nl])), 3),
              "inject", round(float(np.mean([r["stale_any"] for r in nl])), 3), flush=True)


if __name__ == "__main__":
    main()
