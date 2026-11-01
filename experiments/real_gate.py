"""The learning gate on real prompts: this machine's Claude Code transcripts, replayed with verdicts stood in for.

As in ``real_sessions``, each project's sessions are replayed through the hooks in time order, into a throwaway store. No agent
judged these notes, so a verdict is stood in for: a prompt's notes helped when one that is not an edit trail names a file the turn
then touched, and it is missed with the rate measured on a real agent. That label is weak, so read the numbers as a check
on direction. One prompt from another project follows every ``FOREIGN_EVERY`` own prompts, and those notes never help.

Prompts stay in memory. results/real_gate.json holds counts only, with no text and no project names.
Usage: python -m experiments.real_gate [transcripts folder]
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from experiments.harness import RESULTS, git_commit
from experiments.real_sessions import MIN_PROMPTS, SKIP, TRANSCRIPTS, sessions_of, touched, turns
from kanerva import hooks
from kanerva.embed import fetch_minilm
from kanerva.learning import GATE_SESSIONS, GATE_WINDOW, PROBE_RATE, best_shift, probes

FOREIGN_EVERY = 4
MISS = 0.213  # share of needed notes the agent left unmarked on experiments/agent_signal.py


def hit(items, files: set[str]) -> bool:
    names = {os.path.basename(t) for item in items if not item.text.startswith("Edited ") for t in re.findall(r"[\w./-]+\.\w{1,6}\b", item.text)}
    return bool(names & files)


def replay(project: list[tuple[str, list[dict]]], foreign: list[str], rng: random.Random, learn: bool) -> list[dict]:
    cwd = next(e["cwd"] for _, events in project for e in events if e.get("cwd"))
    mem, scope = hooks._open(cwd)
    log: list[tuple[float, int, float, str]] = []
    rows = []
    turn = 0

    def ask(text: str, files: set[str] | None, session: str) -> None:
        prepared = mem.prepare(text, time.time(), scope)
        if prepared.problem is None:
            return
        margin = mem.gate_margin(prepared)
        free = mem.select(prepared, hooks.PROMPT_BUDGET, record=False, min_weight=hooks.MIN_WEIGHT)
        recent = log[-GATE_WINDOW:]
        ready = len({row[3] for row in recent}) >= GATE_SESSIONS
        shift = best_shift(recent) if learn and ready else 0.0
        rate = PROBE_RATE[ready]
        spoke = margin >= shift or (learn and probes(text, margin - shift, rate))
        useful = files is not None and hit(free.items, files)
        if spoke and free.items:
            log.append((margin, int(useful and rng.random() > MISS), 1.0 if margin >= shift else 1.0 / rate, session))
        rows.append({"own": files is not None, "useful": useful, "spoke": spoke and bool(free.items), "tokens": free.tokens if spoke else 0, "shift": shift})

    for session, events in project:
        for event in events:
            if event["kind"] == "summary" and event["text"]:
                mem.write(event["text"], scope=scope, kind="semantic", source={"session": session})
        for prompt, tools in turns(events):
            ask(prompt["text"], touched(tools, prompt.get("cwd")), session)
            turn += 1
            if turn % FOREIGN_EVERY == 0 and foreign:
                ask(rng.choice(foreign), None, session)
            for event in tools:
                if event["kind"] == "tool":
                    hooks.on_tool(mem, session, scope, event["name"], event["input"], event["response"], event.get("cwd"))
            replies = [e["text"] for e in tools if e["kind"] == "reply"]
            if replies and hooks.wrapup_request(mem, session, {}):
                mem.write(" ".join(replies[-1].split()[:60]), scope=scope, kind="semantic", source={"session": session})
            mem.commit()
    mem.close()
    return rows


def later_half(rows: list[dict]) -> list[dict]:
    return rows[len(rows) // 2:]


def summary(late: list[dict]) -> dict:
    useful = [r for r in late if r["useful"]]
    idle = [r for r in late if r["own"] and not r["useful"]]
    other = [r for r in late if not r["own"]]
    return {"rows": len(late), "useful": len(useful), "spoke_useful": round(float(np.mean([r["spoke"] for r in useful])), 3) if useful else None,
            "spoke_idle": round(float(np.mean([r["spoke"] for r in idle])), 3) if idle else None, "spoke_foreign": round(float(np.mean([r["spoke"] for r in other])), 3) if other else None,
            "tokens": round(float(np.mean([r["tokens"] for r in late])), 1), "last_shift": late[-1]["shift"] if late else 0.0}


def recurrence(projects: dict) -> dict:
    """How often a turn that touches files returns to a file from an earlier session, which sets the repeats in ``agent_chain``."""
    turns_with_files, returning = 0, 0
    for sessions in projects.values():
        seen: set[str] = set()
        for _, events in sessions:
            now: set[str] = set()
            for prompt, tools in turns(events):
                files = touched(tools, prompt.get("cwd"))
                if files:
                    turns_with_files += 1
                    returning += bool(files & seen)
                    now |= files
            seen |= now
    return {"turns": turns_with_files, "returning": returning, "share": round(returning / max(turns_with_files, 1), 3)}


def main(folder: Path = TRANSCRIPTS) -> None:
    projects = {}
    for path in sorted(folder.iterdir()):
        if path.is_dir() and not SKIP.search(path.name):
            sessions = sessions_of(path)
            if sum(e["kind"] == "prompt" for _, events in sessions for e in events) >= MIN_PROMPTS:
                projects[path.name] = sessions
    prompts = {name: [e["text"] for _, events in s for e in events if e["kind"] == "prompt"] for name, s in projects.items()}
    work = Path(tempfile.mkdtemp(prefix="kanerva-gate-"))
    os.environ["KANERVA_HOME"] = str(work / "models-home")
    models = fetch_minilm().parent
    report: dict = {"meta": {"experiment": "the learning gate on real prompts", "commit": git_commit(), "projects": len(projects), "foreign_every": FOREIGN_EVERY, "miss": MISS},
                    "recurrence": recurrence(projects)}
    for arm, learn in (("fixed", False), ("learned", True)):
        everything, per_project = [], []
        for index, (name, sessions) in enumerate(projects.items()):
            home = work / f"{arm}-{index}"
            home.mkdir()
            (home / "models").symlink_to(models)
            os.environ["KANERVA_HOME"] = str(home)
            foreign = [p for other, texts in prompts.items() if other != name for p in texts]
            rows = replay(sessions, foreign, random.Random(index), learn)
            everything += later_half(rows)
            per_project.append(summary(later_half(rows)))
        report[arm] = {"all": summary(everything), "projects": per_project}
    shutil.rmtree(work, ignore_errors=True)
    (RESULTS / "real_gate.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else TRANSCRIPTS))
