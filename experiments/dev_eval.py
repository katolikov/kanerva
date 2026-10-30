"""Run chosen methods through the full harness on development data, for tuning and side checks."""

from __future__ import annotations

import json
import os
import sys

from experiments.dump_features import DATASETS
from experiments.harness import RESULTS, aggregate, git_commit, run_dataset, table
from experiments.registry import load
from kanerva import RetrievalConfig


def main(names: list[str]) -> None:
    selected = set(os.environ.get("METHODS", "hybrid,bm25,kanerva,kanerva-x128").split(","))
    base = RetrievalConfig()
    out = {}
    for name in names:
        loader, cache_name, keyed, splits = DATASETS[name]
        streams = [s for s in load(loader) if s.split in splits]
        records = run_dataset(streams, cache_name, base, keyed=keyed, selected=selected)
        out[name] = records
        print(f"== {name} recall\n{table(records)}\n== {name} answer-term recall\n{table(records, 'answer')}", flush=True)
    summary = {name: aggregate(records) for name, records in out.items()}
    (RESULTS / f"dev_{os.environ.get('TAG', 'eval')}.json").write_text(json.dumps({"meta": {"commit": git_commit()}, "summary": summary}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
