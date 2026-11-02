"""Rerun chains of ``agent_chain`` and keep what each session saw and wrote, to read a result that looks wrong.

Usage: python -m experiments.chain_trace out.json repo,repo fixed=shadow,learned=auto
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import experiments.agent_chain as chain
from experiments.agent_signal import MODELS_HOME
from kanerva.embed import OnnxEmbedder, fetch_minilm, minilm_folder


def main(out: str, repos: str, arms: str) -> None:
    chain.ARMS.clear()
    chain.ARMS.update(dict(a.split("=") for a in arms.split(",")))
    os.environ["KANERVA_HOME"] = str(MODELS_HOME)
    fetch_minilm()
    embedder = OnnxEmbedder(minilm_folder())
    chains = [chain.make_chain(repo, random.Random(repo), chain.SESSIONS) for repo in sorted(repos.split(","))]
    folder = Path(tempfile.mkdtemp(prefix="kanerva-trace-")).resolve()
    jobs = [(c, "default", arm) for c in chains for arm in chain.ARMS]
    with ThreadPoolExecutor(chain.WORKERS) as pool:
        rows = list(pool.map(lambda pair: chain.run_chain(*pair[1], folder / f"chain-{pair[0]}", embedder), enumerate(jobs)))
    shutil.rmtree(folder, ignore_errors=True)
    for path in (Path.home() / ".claude" / "projects").glob(f"*{folder.name.replace('_', '-')}*"):
        shutil.rmtree(path, ignore_errors=True)
    Path(out).write_text(json.dumps(rows))


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
