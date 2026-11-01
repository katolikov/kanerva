"""Can a real agent say which notes helped? The signal any learning would rest on.

Each trial is a small documentation chore on a repository the method has never seen. The store holds true facts about that
repository, the prompt needs two of them, and in a third of the trials the fact a prompt needs is held only in an older
version, while the code shows the current one. Claude Code runs with the plugin, does the chore, and is asked at the stop for its
verdict, as any session with real work is. The labels come from how the trial was built: a shown note is ``needed``, ``stale``,
``related`` when it names the same symbol, or ``other``.

The same sessions also score the signal the plugins used before, ``item_used``, which matches a note's identifiers against
later tool calls. The thresholds in ``PASS`` were set before the first run.

Needs the claude CLI signed in, and spends real tokens. Sessions load no user settings and write no transcripts, and the store
and the project live in a temporary folder. Writes results/agent_signal.json.

Usage: python -m experiments.agent_signal [trials per repository]
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.agent_write_path import SESSION_VARIABLES, share_interval
from experiments.harness import RESULTS, git_commit
from experiments.real_sessions import item_used
from experiments.repomem import (
    QUERY_TEMPLATES,
    c_snapshot_facts,
    ensure_clone,
    extract_c_facts,
    extract_cpp_facts,
    git,
    render_fact,
    snapshot_facts,
)
from kanerva import Memory
from kanerva.embed import OnnxEmbedder, fetch_minilm, minilm_folder
from kanerva.paths import project_scope

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "plugins" / "claude-code"
RAW = RESULTS / "raw" / "agent_signal.json"
MODELS_HOME = ROOT / "experiments" / "cache" / "agent-models"
MODEL = "claude-sonnet-5-5"
WORKERS = 4
MAX_USD = "1.0"
NOTES = 60
STALE_EVERY = 3
YEARS_BACK = 3
PASS = {"coverage": 0.8, "precision": 0.8, "recall": 0.6, "false_mark": 0.1, "misled": 0.5}
# Repositories no earlier test has read: (source root, reader of its facts).
REPOS = {
    "python-humanize/humanize": ("src/humanize", "python"),
    "more-itertools/more-itertools": ("more_itertools", "python"),
    "madler/zlib": ("", "c"),
    "leethomason/tinyxml2": ("", "cpp"),
    "zeux/pugixml": ("src", "cpp"),
}
TOOLS = ["Read", "Write", "Edit", "Glob", "Grep", "Bash(grep:*)", "Bash(ls:*)", "Bash(cat:*)", "Bash(find:*)", "Bash(git log:*)",
         *(f"mcp__plugin_kanerva_memory__memory_{name}" for name in ("feedback", "remember", "forget", "edit", "search", "expand"))]
PROMPT = (
    "A documentation chore for this repository. Create four short files under docs/handover/, each under five lines:\n"
    "1. first.md: answer this: {first}\n"
    "2. second.md: answer this: {second}\n"
    "3. audience.md: who this project is for, in one line.\n"
    "4. index.md: one line per file above.\n"
    "Do not build or run anything."
)


def facts_of(repo: str) -> tuple[Path, str, dict, dict]:
    root, reader = REPOS[repo]
    clone = ensure_clone(repo)
    head = git(clone, "rev-parse", "HEAD").strip()
    older = git(clone, "log", "-1", "--first-parent", f"--before={YEARS_BACK} years ago", "--format=%H", head).strip()
    if reader == "python":
        return clone, head, snapshot_facts(clone, head, root), snapshot_facts(clone, older, root)
    read = extract_c_facts if reader == "c" else extract_cpp_facts
    return clone, head, c_snapshot_facts(clone, head, root, read), c_snapshot_facts(clone, older, root, read)


def question(rng: random.Random, fact: tuple[str, str, str, str]) -> str:
    kind, name, _, path = fact
    return rng.choice(QUERY_TEMPLATES[kind]).format(name=name, path=path)


def make_trials(repo: str, n: int, rng: random.Random) -> list[dict]:
    clone, head, now, before = facts_of(repo)
    answerable = sorted(k for k, f in now.items() if f[0] in ("constant", "signature"))
    changed = sorted(k for k in answerable if k in before and before[k][2] != now[k][2] and before[k][0] == now[k][0])
    trials = []
    for index in range(n):
        stale = index % STALE_EVERY == 0 and bool(changed)
        second = rng.choice(changed) if stale else rng.choice(answerable)
        first = rng.choice([k for k in answerable if k != second])
        rest = rng.sample(sorted(set(now) - {first, second}), min(NOTES, len(now) - 2))
        notes = [{"text": render_fact(*now[first][:3], now[first][3]), "label": "needed", "name": now[first][1]}]
        fact = before[second] if stale else now[second]
        notes.append({"text": render_fact(*fact[:3], fact[3]), "label": "stale" if stale else "needed", "name": now[second][1]})
        notes += [{"text": render_fact(*now[k][:3], now[k][3]), "label": "other", "name": now[k][1]} for k in rest]
        rng.shuffle(notes)
        trials.append({"repo": repo, "index": index, "clone": str(clone), "head": head, "stale": stale, "notes": notes,
                       "prompt": PROMPT.format(first=question(rng, now[first]), second=question(rng, now[second])),
                       "values": {"first": now[first][2], "second": now[second][2], "second_old": before[second][2] if stale else ""}})
    return trials


def prepare(trial: dict, folder: Path, embedder: OnnxEmbedder) -> tuple[Path, Path, dict[int, dict]]:
    home, project = folder / "kanerva", folder / "project"
    home.mkdir(parents=True)
    project.mkdir()
    archive = subprocess.run(["git", "-C", trial["clone"], "archive", trial["head"]], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(project)], input=archive, check=True)
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    (home / "models").symlink_to(MODELS_HOME / "models")
    mem = Memory(home / "memory.sqlite", embedder=embedder)
    scope = project_scope(str(project))
    labels = {}
    for at, note in enumerate(trial["notes"]):
        labels[mem.write(note["text"], at=1_000_000.0 + at, scope=scope, kind="semantic")] = note
    mem.commit()
    mem.close()
    return home, project, labels


def run_agent(prompt: str, project: Path, home: Path, extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in SESSION_VARIABLES} | {"KANERVA_HOME": str(home), "KANERVA_WRAPUP": "1"} | (extra or {})
    command = ["claude", "-p", prompt, "--plugin-dir", str(PLUGIN), "--model", MODEL, "--setting-sources", "project", "--no-session-persistence",
               "--allowedTools", *TOOLS, "--max-budget-usd", MAX_USD, "--output-format", "stream-json", "--verbose"]
    out = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=420).stdout
    calls, observed, cost, reply = [], [], 0.0, ""
    for line in out.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            for block in event["message"].get("content", []):
                if block.get("type") == "tool_use":
                    calls.append(block["name"].split("__")[-1])
                    if block["name"] in ("Bash", "Write", "Edit", "Read"):  # what the old signal's hook could see
                        observed.append(str(block["input"].get("command") or json.dumps(block["input"]))[:4000])
        elif event.get("type") == "result":
            cost, reply = event.get("total_cost_usd", 0.0), event.get("result", "")
    return {"calls": calls, "observed": "\n".join(observed), "cost": cost, "reply": reply}


def shown_notes(home: Path) -> set[int]:
    with sqlite3.connect(home / "memory.sqlite") as db:
        ids = {i for (row,) in db.execute("SELECT item_ids FROM retrievals") for i in json.loads(row)}
        ids |= {i for (row,) in db.execute("SELECT items FROM session_files") for i in json.loads(row)}
    return ids


def verdicts(home: Path) -> tuple[set[int], set[int]]:
    with sqlite3.connect(home / "memory.sqlite") as db:
        rows = db.execute("SELECT item_id, helped FROM verdicts WHERE item_id IS NOT NULL").fetchall()
    return {i for i, h in rows if h}, {i for i, h in rows if not h}


def trial_row(trial: dict, folder: Path, embedder: OnnxEmbedder) -> dict:
    home, project, labels = prepare(trial, folder, embedder)
    run = run_agent(trial["prompt"], project, home)
    shown = shown_notes(home)
    helped, misled = verdicts(home)
    names = {note["name"] for note in labels.values() if note["label"] != "other"}
    rows = []
    for i in sorted(shown & set(labels)):
        note = labels[i]
        label = "related" if note["label"] == "other" and note["name"] in names else note["label"]
        rows.append({"label": label, "helped": i in helped, "misled": i in misled, "old_signal": item_used(note["text"], run["observed"])})
    second = project / "docs" / "handover" / "second.md"
    written = second.read_text(errors="replace") if second.exists() else ""
    values = trial["values"]
    return {"repo": trial["repo"], "index": trial["index"], "stale": trial["stale"], "notes": rows, "calls": run["calls"], "cost": run["cost"], "reply": run["reply"][:300],
            "judged": "memory_feedback" in run["calls"], "unseen_marks": len((helped | misled) - shown),
            "wrote_new": bool(values["second"]) and re.sub(r"\s", "", values["second"]) in re.sub(r"\s", "", written),
            "wrote_old": bool(values["second_old"]) and re.sub(r"\s", "", values["second_old"]) in re.sub(r"\s", "", written),
            "shown_needed": sum(r["label"] == "needed" for r in rows), "shown_stale": sum(r["label"] == "stale" for r in rows)}


def rate(rows: list[dict], key: str) -> dict:
    hits = sum(r[key] for r in rows)
    return {"hits": hits, "n": len(rows), "share": round(hits / max(len(rows), 1), 3), "interval": share_interval(hits, len(rows))}


def summary(rows: list[dict]) -> dict:
    judged = [r for r in rows if r["judged"]]
    notes = [note for r in judged for note in r["notes"]]
    by = {label: [n for n in notes if n["label"] == label] for label in ("needed", "stale", "related", "other")}
    marked = [n for n in notes if n["helped"]]
    old_marked = [n for n in notes if n["old_signal"]]
    helped_right = sum(n["label"] == "needed" for n in marked)
    old_right = sum(n["label"] == "needed" for n in old_marked)
    out = {
        "trials": len(rows), "coverage": round(len(judged) / max(len(rows), 1), 3), "shown": {k: len(v) for k, v in by.items()},
        "verdict": {"precision": round(helped_right / max(len(marked), 1), 3), "precision_interval": share_interval(helped_right, len(marked)),
                    "recall": rate(by["needed"], "helped"), "false_mark": rate(by["other"], "helped"), "related_marked": rate(by["related"], "helped"),
                    "stale_marked_misled": rate(by["stale"], "misled"), "stale_marked_helped": rate(by["stale"], "helped"), "other_marked_misled": rate(by["other"], "misled")},
        "old_signal": {"precision": round(old_right / max(len(old_marked), 1), 3), "precision_interval": share_interval(old_right, len(old_marked)),
                       "recall": rate(by["needed"], "old_signal"), "false_mark": rate(by["other"], "old_signal"), "stale_marked_helped": rate(by["stale"], "old_signal")},
        "unseen_marks": sum(r["unseen_marks"] for r in rows),
        "stale_trials": {"n": sum(r["stale"] for r in rows), "wrote_new": sum(r["wrote_new"] for r in rows if r["stale"]), "wrote_old": sum(r["wrote_old"] for r in rows if r["stale"])},
    }
    v = out["verdict"]
    out["passed"] = {"coverage": out["coverage"] >= PASS["coverage"], "precision": v["precision"] >= PASS["precision"], "recall": v["recall"]["share"] >= PASS["recall"],
                     "false_mark": v["false_mark"]["share"] <= PASS["false_mark"], "misled": v["stale_marked_misled"]["share"] >= PASS["misled"]}
    return out


def main(per_repo: int = 8, repos: tuple[str, ...] = tuple(REPOS)) -> None:
    rng = random.Random("agent-signal-1")
    version = subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
    os.environ["KANERVA_HOME"] = str(MODELS_HOME)
    fetch_minilm()
    embedder = OnnxEmbedder(minilm_folder())
    trials = [t for repo in repos for t in make_trials(repo, per_repo, rng)]
    folder = Path(tempfile.mkdtemp(prefix="kanerva-signal-")).resolve()
    start = time.time()

    def one(pair: tuple[int, dict]) -> dict:
        try:
            row = trial_row(pair[1], folder / f"trial-{pair[0]}", embedder)
        except Exception as exc:  # a timeout or a crash counts as a session with no verdict
            row = {"repo": pair[1]["repo"], "index": pair[1]["index"], "stale": pair[1]["stale"], "notes": [], "calls": [], "cost": 0.0, "reply": repr(exc)[:300],
                   "judged": False, "unseen_marks": 0, "wrote_new": False, "wrote_old": False, "shown_needed": 0, "shown_stale": 0}
        shutil.rmtree(folder / f"trial-{pair[0]}", ignore_errors=True)
        print(json.dumps({k: row[k] for k in ("repo", "index", "stale", "judged", "shown_needed", "shown_stale", "cost")} | {"marks": [(n["label"], n["helped"], n["misled"]) for n in row["notes"] if n["helped"] or n["misled"] or n["label"] != "other"]}), flush=True)
        return row

    with ThreadPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(one, enumerate(trials)))
    shutil.rmtree(folder, ignore_errors=True)
    for path in (Path.home() / ".claude" / "projects").glob(f"*{folder.name.replace('_', '-')}*"):
        shutil.rmtree(path, ignore_errors=True)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(json.dumps(rows))
    report = {"meta": {"experiment": "can a real agent say which notes helped", "commit": git_commit(), "agent": version, "model": MODEL, "pass": PASS, "prompt": PROMPT,
                       "repos": list(repos), "cost_usd": round(sum(r["cost"] for r in rows), 2), "seconds": round(time.time() - start)},
              **summary(rows), "by_repo": {repo: summary([r for r in rows if r["repo"] == repo]) for repo in repos}}
    (RESULTS / "agent_signal.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "by_repo"}, indent=1))


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8, tuple(sys.argv[2:]) or tuple(REPOS)))
