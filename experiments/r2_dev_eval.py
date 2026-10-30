"""Round-2 development runs through the full harness. Dev splits only; never the fresh test."""

from __future__ import annotations

import json
import os
import sys

from experiments.harness import RESULTS, run_dataset, table
from experiments.r2_dump import DATASETS
from experiments.registry import load
from kanerva import RetrievalConfig


def main(names: list[str]) -> None:
    selected = set(os.environ.get("R2_METHODS", "hybrid,bm25,kanerva,kanerva-r1,kanerva-x64,kanerva-x128,kanerva-x256").split(","))
    base = RetrievalConfig(**json.loads(os.environ.get("R2_BASE", "{}")))
    out = {}
    for name in names:
        loader, cache_name, keyed, splits = DATASETS[name]
        streams = [s for s in load(loader) if s.split in splits]
        if os.environ.get("R2_DEV_ONLY"):
            streams = [s for s in streams if s.split == "dev"]
        records = run_dataset(streams, cache_name, base, keyed=keyed, selected=selected)
        out[name] = records
        print(f"== {name} recall\n{table(records)}\n== {name} answer-term recall\n{table(records, 'answer')}", flush=True)
    tag = os.environ.get("R2_TAG", "dev")
    with open(RESULTS / f"r2_{tag}.json", "w") as fh:
        json.dump({k: v for k, v in out.items()}, fh)


if __name__ == "__main__":
    main(sys.argv[1:])
