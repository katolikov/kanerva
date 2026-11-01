"""What the gate and the weight floor hide on plain questions, with no agent.

The chains of ``agent_chain`` ask for a fact in plain words, with no identifier from it. This asks the same questions of the same
stores at several settings of the gate and of the floor under note weights, and counts how often the needed note reaches the
context and how many tokens that costs. Writes results/plain_questions.json.

Usage: python -m experiments.plain_questions
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

from experiments.agent_chain import DEV_REPOS, REPOS, SESSIONS, SPENT_REPOS, make_chain
from experiments.agent_signal import MODELS_HOME
from experiments.harness import RESULTS, git_commit
from kanerva import Memory
from kanerva.embed import OnnxEmbedder, fetch_minilm, minilm_folder
from kanerva.learning import Tuning

SETTINGS = ((0.0, 0.02), (-3.0, 0.02), (-3.0, 0.002))
PROMPT_BUDGET = 600


def measure(repo: str, sessions: int, embedder: OnnxEmbedder) -> dict:
    chain = make_chain(repo, random.Random(repo), sessions)
    mem = Memory(Path(tempfile.mkdtemp()) / "memory.sqlite", embedder=embedder)
    ids = {}
    for note in chain["notes"]:
        item = mem.write(note["text"], at=note["at"], scope="project", kind=note["kind"], key=note["memory_key"])
        if note["key"]:
            ids[note["key"]] = item
    mem.config = mem.config.with_(use_bandit=False, learn=False)
    now, out = time.time(), {}
    in_pool = sum(ids[task["keys"][0]] in mem.prepare(task["prompt"], now, "project").candidates.ids for task in chain["tasks"])
    for shift, floor in SETTINGS:
        mem.tuning = Tuning(shift=shift)
        found, tokens = 0, 0
        for task in chain["tasks"]:
            result = mem.retrieve(task["prompt"], PROMPT_BUDGET, scope="project", now=now, min_weight=floor, gated=True, record=False)
            found += ids[task["keys"][0]] in {item.id for item in result.items}
            tokens += result.tokens
        out[f"{shift}/{floor}"] = {"found": found, "n": len(chain["tasks"]), "tokens": tokens / len(chain["tasks"])}
    mem.close()
    return {"settings": out, "in_pool": in_pool, "n": len(chain["tasks"])}


def main() -> None:
    os.environ["KANERVA_HOME"] = str(MODELS_HOME)
    fetch_minilm()
    embedder = OnnxEmbedder(minilm_folder())
    rows = {repo: measure(repo, SESSIONS, embedder) for repo in (*DEV_REPOS, *SPENT_REPOS, *REPOS)}
    totals = {}
    for setting in rows[next(iter(rows))]["settings"]:
        parts = [r["settings"][setting] for r in rows.values()]
        found, n = sum(p["found"] for p in parts), sum(p["n"] for p in parts)
        totals[setting] = {"found": found, "n": n, "share": found / n, "tokens": sum(p["tokens"] for p in parts) / len(parts)}
    report = {"meta": {"experiment": "what the gate and the floor hide on plain questions", "commit": git_commit(), "prompt_budget": PROMPT_BUDGET, "repos": list(rows), "probes": "off"},
              "settings": totals, "in_pool": sum(r["in_pool"] for r in rows.values()), "n": sum(r["n"] for r in rows.values()), "repos": rows}
    (RESULTS / "plain_questions.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(totals, indent=1))


if __name__ == "__main__":
    sys.exit(main())
