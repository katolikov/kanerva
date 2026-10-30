"""Quick dev-split comparison used during design; not part of the reported results."""

from __future__ import annotations

import sys
import time

from experiments.harness import run_dataset, table
from experiments.registry import load
from kanerva import RetrievalConfig


def main() -> None:
    name = sys.argv[1]
    keyed = len(sys.argv) > 2 and sys.argv[2] == "keyed"
    streams = [s for s in load(name) if s.split == "dev"]
    start = time.time()
    records = run_dataset(streams, name, RetrievalConfig(), keyed=keyed)
    print(f"{name} dev: {len(records)} records, {time.time() - start:.0f}s")
    for metric in sys.argv[3].split(",") if len(sys.argv) > 3 else ("recall", "all"):
        print(f"\n[{metric}]\n" + table(records, metric))


if __name__ == "__main__":
    main()
