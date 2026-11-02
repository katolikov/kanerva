"""Why right answers seemed to fall in the run of release 0.6.0.

That run counted a signature as right only if the answer held its whole parameter list word for word. Two of its repositories were
rerun with the same chores and with answers kept, in results/chain_trace.json, and here each answer is scored both ways. Writes
results/chain_rescore.json.

Usage: python -m experiments.chain_rescore
"""

from __future__ import annotations

import json
import os
import random
import re
import sys

import experiments.agent_chain as chain
from experiments.agent_signal import MODELS_HOME
from experiments.harness import RESULTS, git_commit

ORDER = ("dateutil/dateutil", "PyCQA/isort", "Pylons/pyramid", "jemalloc/jemalloc", "ToruNiina/toml11", "fastfloat/fast_float")  # the run's own order sets its chores


def main(path: str = str(RESULTS / "chain_trace.json")) -> None:
    os.environ["KANERVA_HOME"] = str(MODELS_HOME)
    rows = json.load(open(path))
    rng = random.Random("agent-chain-1")
    chains = {repo: chain.make_chain(repo, rng, chain.SESSIONS) for repo in ORDER}
    totals: dict = {arm: {"word_for_word": 0, "named": 0, "sessions": 0} for arm in ("fixed", "learned")}
    for row in rows:
        for task, session in zip(chains[row["repo"]]["tasks"][12:], row["sessions"][12:]):
            answer = session["trace"]["answer"]
            value = task["values"][0].replace(" ", "") in answer.replace(" ", "").replace("\n", "")
            named = value if task["kinds"][0] == "constant" else re.split(r"::|\.", task["names"][0])[-1] in answer
            totals[row["arm"]]["word_for_word"] += session.get("quoted", session["right"])
            totals[row["arm"]]["named"] += int(named)
            totals[row["arm"]]["sessions"] += 1
    report = {"meta": {"experiment": "scoring of the chains of release 0.6.0", "commit": git_commit(), "repos": sorted({r["repo"] for r in rows})}, **totals}
    (RESULTS / "chain_rescore.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
