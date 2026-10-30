"""Store sizes in tokens, and how item length relates to being evidence."""

from __future__ import annotations

import json

import numpy as np

from experiments.embed_cache import TextCache
from experiments.harness import RESULTS
from experiments.registry import load

SPLITS = {"lme": ("lme", "test"), "locomo": ("locomo", "test"), "beam": ("beam", "test")}


def main() -> None:
    out: dict = {}
    for name, (dataset, split) in SPLITS.items():
        cache = TextCache(dataset)
        streams = [s for s in load(dataset) if s.split == split]
        gold = {t for s in streams for q in s.queries for t in q.gold}
        texts = {t.text for s in streams for t in s.writes}
        out[name] = {
            "full_tokens": float(np.mean([sum(cache.count(t.text) + 14 for t in s.writes) for s in streams])),
            "gold_tokens": float(np.mean([cache.count(t) for t in gold])),
            "other_tokens": float(np.mean([cache.count(t) for t in texts - gold])),
        }
    (RESULTS / "dataset_stats.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
