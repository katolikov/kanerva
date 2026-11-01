"""Run every experiment in order, with a wall-clock cap per step.

The first and second held-out tests, and the checks built on them, are the record of the earlier method. Each result file
names its commit, and rerunning them here would score the current method, so they run only when named, as in
``python -m experiments.reproduce fresh-test``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYSTEMS_PYTHON = ROOT / ".venv-systems" / "bin" / "python"
ON_REQUEST = {"fresh-test", "second-test", "static-twins", "languages", "updates", "answers", "c-dev", "c-test", "agent-write-path", "agent-write-path-c", "real-sessions"}
STEPS = [
    ("dataset-stats", sys.executable, ["experiments.dataset_stats"], 600),
    ("pool", sys.executable, ["experiments.pool_recall", "beam-dev", "locomo"], 3600),
    ("features", sys.executable, ["experiments.dump_features"], 7200),
    ("models", sys.executable, ["experiments.compare_models"], 7200),
    ("ablation", sys.executable, ["experiments.feature_ablation"], 7200),
    ("refit-dense", sys.executable, ["experiments.refit", "dense"], 7200),
    ("refit-lexical", sys.executable, ["experiments.refit", "lexical", "as-shipped", "all-families"], 3600),
    ("fit", sys.executable, ["experiments.fit_relevance"], 3600),
    ("fresh-test", sys.executable, ["experiments.final_test"], 7200),
    ("second-test", sys.executable, ["experiments.new_repos_test"], 7200),
    ("static-twins", sys.executable, ["experiments.static_twins"], 3600),
    ("languages", sys.executable, ["experiments.languages_test"], 3600),
    ("abstain", sys.executable, ["experiments.abstain"], 7200),
    ("abstain-report", sys.executable, ["experiments.abstain", "report"], 600),
    ("third-test", sys.executable, ["experiments.third_test"], 7200),
    ("beam-check", sys.executable, ["experiments.third_test", "beam"], 3600),
    ("abstain-final", sys.executable, ["experiments.abstain", "final"], 3600),
    ("file-recall", sys.executable, ["experiments.file_recall"], 3600),
    ("file-recall-final", sys.executable, ["experiments.file_recall", "repomem-final"], 3600),
    ("write-path", sys.executable, ["experiments.write_path"], 7200),
    ("russian", sys.executable, ["experiments.russian_test"], 7200),
    ("c-dev", sys.executable, ["experiments.third_test", "c-dev"], 7200),
    ("c-test", sys.executable, ["experiments.third_test", "c-test"], 7200),
    ("c-final", sys.executable, ["experiments.third_test", "c-final"], 7200),
    ("c-last", sys.executable, ["experiments.third_test", "c-last"], 7200),
    ("fact-losses", sys.executable, ["experiments.third_test", "losses"], 300),
    ("twin-rule", sys.executable, ["experiments.third_test", "twins"], 3600),
    ("refit-c", sys.executable, ["experiments.refit", "dense", "all-families", "with-c"], 7200),
    ("refit-c-lexical", sys.executable, ["experiments.refit", "lexical", "all-families", "with-c"], 7200),
    ("file-recall-c", sys.executable, ["experiments.file_recall", "repomem-c-dev", "repomem-c-test", "repomem-c-final", "repomem-c-last"], 3600),
    ("abstain-c", sys.executable, ["experiments.abstain", "c-test"], 3600),
    ("abstain-c-final", sys.executable, ["experiments.abstain", "c-final"], 3600),
    ("abstain-c-last", sys.executable, ["experiments.abstain", "c-last"], 3600),
    ("abstain-static", sys.executable, ["experiments.abstain", "static"], 3600),
    ("data-flaws", sys.executable, ["experiments.data_flaws"], 600),
    ("answers", sys.executable, ["experiments.answer_test", "run", "--tag", "full"], 14400),
    ("agent-write-path", sys.executable, ["experiments.agent_write_path"], 3600),
    ("agent-write-path-c", sys.executable, ["experiments.agent_write_path", "12", "repomem-c-final"], 3600),
    ("real-sessions", sys.executable, ["experiments.real_sessions"], 3600),
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
        skip = name not in only if only else name in ON_REQUEST
        if skip:
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
