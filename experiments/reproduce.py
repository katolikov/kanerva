"""Run every experiment in order, with a wall-clock cap per step."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYSTEMS_PYTHON = ROOT / ".venv-systems" / "bin" / "python"
STEPS = [
    ("dataset-stats", sys.executable, ["experiments.dataset_stats"], 600),
    ("pool", sys.executable, ["experiments.pool_recall", "beam-dev", "locomo"], 3600),
    ("features", sys.executable, ["experiments.dump_features"], 7200),
    ("models", sys.executable, ["experiments.compare_models"], 7200),
    ("ablation", sys.executable, ["experiments.feature_ablation"], 7200),
    ("fit", sys.executable, ["experiments.fit_relevance"], 3600),
    ("fresh-test", sys.executable, ["experiments.final_test"], 7200),
    ("second-test", sys.executable, ["experiments.new_repos_test"], 7200),
    ("static-twins", sys.executable, ["experiments.static_twins"], 3600),
    ("languages", sys.executable, ["experiments.languages_test"], 3600),
    ("hook-latency", sys.executable, ["experiments.hook_latency"], 1800),
    ("updates", sys.executable, ["experiments.dev_eval", "lme-updates"], 1800),
    ("store", sys.executable, ["experiments.run_store"], 3600),
    ("online", sys.executable, ["experiments.run_online"], 3600),
    ("synthetic", sys.executable, ["experiments.run_synthetic"], 1800),
    ("regret-scale", sys.executable, ["experiments.regret_scale"], 300),
    ("baseline-systems", str(SYSTEMS_PYTHON), ["experiments.run_systems"], 1800),
    ("figures", sys.executable, ["experiments.figures"], 300),
]


def main() -> None:
    only = set(sys.argv[1:])
    for name, python, module, cap in STEPS:
        if only and name not in only:
            continue
        if not Path(python).exists():
            print(f"== {name} skipped: {python} is missing (see experiments/run_systems.py)", flush=True)
            continue
        start = time.time()
        print(f"== {name} (cap {cap}s)", flush=True)
        env = {**os.environ, "TAG": "updates", "METHODS": "hybrid,hybrid-newest,hybrid-dedup,kanerva"} if name == "updates" else None
        subprocess.run([python, "-m", *module], check=True, timeout=cap, env=env)
        print(f"== {name} done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
