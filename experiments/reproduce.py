"""Run every experiment with a wall-clock cap per step. ``--fast`` uses the dev split as a smoke test."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYSTEMS_PYTHON = ROOT / ".venv-systems" / "bin" / "python"
STEPS = [
    ("synthetic", sys.executable, ["experiments.run_synthetic"], 1800),
    ("store", sys.executable, ["experiments.run_store"], 3600),
    ("rerank", sys.executable, ["experiments.run_rerank"], 3600),
    ("retrieval", sys.executable, ["experiments.run_retrieval"], 3600),
    ("online", sys.executable, ["experiments.run_online"], 3600),
    ("baseline-systems", str(SYSTEMS_PYTHON), ["experiments.run_systems"], 1800),
    ("analysis", sys.executable, ["experiments.analysis"], 600),
]


def main() -> None:
    fast = "--fast" in sys.argv
    if fast:
        code = subprocess.run([sys.executable, "-m", "experiments.run_dev", "lme"], timeout=600).returncode
        sys.exit(code)
    only = {a for a in sys.argv[1:] if not a.startswith("-")}
    for name, python, module, cap in STEPS:
        if only and name not in only:
            continue
        if not Path(python).exists():
            print(f"== {name} skipped: {python} is missing (see experiments/run_systems.py)", flush=True)
            continue
        start = time.time()
        print(f"== {name} (cap {cap}s)", flush=True)
        subprocess.run([python, "-m", *module], check=True, timeout=cap)
        print(f"== {name} done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
