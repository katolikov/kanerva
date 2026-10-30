"""Run every experiment with a wall-clock cap per step. ``--fast`` uses the dev split as a smoke test."""

from __future__ import annotations

import subprocess
import sys
import time

STEPS = [
    ("retrieval", ["experiments.run_retrieval"], 3600),
    ("rerank", ["experiments.run_rerank"], 3600),
    ("online", ["experiments.run_online"], 3600),
    ("synthetic", ["experiments.run_synthetic"], 1800),
    ("store", ["experiments.run_store"], 3600),
    ("baseline-systems", ["experiments.run_systems"], 1800),
    ("analysis", ["experiments.analysis"], 600),
]


def main() -> None:
    fast = "--fast" in sys.argv
    if fast:
        code = subprocess.run([sys.executable, "-m", "experiments.run_dev", "lme"], timeout=600).returncode
        sys.exit(code)
    for name, module, cap in STEPS:
        start = time.time()
        print(f"== {name} (cap {cap}s)", flush=True)
        subprocess.run([sys.executable, "-m", *module], check=True, timeout=cap)
        print(f"== {name} done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
