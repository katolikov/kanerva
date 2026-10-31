"""A real host agent on the write path: what Claude Code does with the notes ``memory_remember`` lists.

experiments/write_path.py lets an oracle stand in for the agent. Here Claude Code runs with the plugin, a store replayed up
to a note, and one instruction: remember this note. Each trial runs on its own copy of the store. The agent sees the list and
decides, and the store afterwards says what it did. On a changed note the right move is to retire or correct the older version,
and on any other note the right move is to touch nothing.

Two kinds of note. ``verbatim`` stores a text word for word, so a conversation turn with several subjects stays one note. ``own``
is the ordinary case on conversations: a first session writes the older turn as a note in the agent's own words, and a second,
fresh session does the same with the newer turn.

Needs the claude CLI signed in, and spends real tokens. Sessions load no user settings and write no transcripts, and the
store and the project live in a temporary folder. Writes results/agent_write_path.json.

Usage: python -m experiments.agent_write_path [trials per dataset and kind] [sets]
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from experiments.embed_cache import TextCache
from experiments.harness import RESULTS, git_commit, new_memory
from experiments.write_path import REPORT_ON, SCOPE, config, notes_of, store, streams_for
from kanerva import Memory
from kanerva.embed import OnnxEmbedder, fetch_minilm, minilm_folder
from kanerva.paths import project_scope

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "plugins" / "claude-code"
RAW = RESULTS / "raw" / "agent_write_path.json"
MODEL = "claude-sonnet-5-5"
SETS = {"repomem-final": ("verbatim", 16), "lme-updates": ("verbatim", 12), "lme-own-notes": ("own", 12)}  # trials of each kind
WORKERS = 4
MAX_USD = "0.4"
PROMPTS = {"verbatim": "Remember this note for later, word for word:\n\n{note}",
           "own": "Save what matters in this message with memory_remember, as one short note in your own words:\n\n{note}"}
TOOLS = " ".join(f"mcp__plugin_kanerva_memory__memory_{name}" for name in ("remember", "forget", "edit", "search", "expand"))
SESSION_VARIABLES = ("CLAUDECODE", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_ENTRYPOINT",
                     "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_PROJECT_DIR", "CLAUDE_PLUGIN_ROOT")


def candidates(job: tuple) -> list[dict]:
    """The notes of one stream that a list could change: changed notes with the older version listed, others with any list.

    Own notes need no list yet, since the agent has not written them: each comes with the earlier turn its first session writes.
    """
    stream, dataset, cache_name, mode = job
    cache = TextCache(cache_name)
    mem = new_memory(cache, config(True))
    id_of: dict[str, int] = {}
    out = []
    users: list[str] = []
    for cursor, (turn, older) in enumerate(notes_of(stream, dataset)):
        store(mem, cache, turn, id_of, True)
        item = id_of.get(turn.text)
        if older is None or item is None:
            if turn.text.startswith("user:"):
                users.append(turn.text)
            continue
        old = {id_of[t] for t in older if t in id_of} - {item}
        if older and not old:
            continue
        row = {"stream": stream.name, "cursor": cursor, "changed": bool(old), "old": sorted(old), "note": turn.text}
        if mode == "own":
            first = older[-1] if older else users[-1] if users else None
            if first is not None:
                out.append(row | {"first": first})
        else:
            listed = {i.id for i in mem.similar(turn.text, SCOPE, exclude=(item,), query_vector=cache.vector(turn.text))}
            if listed & old if old else listed:
                out.append(row)
    mem.close()
    return out


def snapshots(job: tuple) -> None:
    """The store of one stream just before each trial's note, each in a file of its own. Own notes start without the older turns."""
    stream, dataset, cache_name, trials = job
    cache = TextCache(cache_name)
    mem = new_memory(cache, config(True))
    id_of: dict[str, int] = {}
    by_cursor = {trial["cursor"]: trial for trial in trials}
    for cursor, (turn, _) in enumerate(notes_of(stream, dataset)):
        if cursor in by_cursor:
            mem.commit()
            mem.db.execute("VACUUM INTO ?", (by_cursor[cursor]["store"],))
            if "first" in by_cursor[cursor]:
                copy = Memory(by_cursor[cursor]["store"])
                copy.delete(by_cursor[cursor]["old"] + [i for t, i in id_of.items() if t == by_cursor[cursor]["first"]])
                copy.commit()
                copy.close()
        if cursor >= max(by_cursor):
            break
        store(mem, cache, turn, id_of, True)
    mem.close()


def pick(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """``n`` trials spread evenly over streams."""
    by_stream: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_stream[row["stream"]].append(row)
    ranked = []
    for group in by_stream.values():
        rng.shuffle(group)
        ranked.extend((rank, rng.random(), row) for rank, row in enumerate(group))
    return [row for _, _, row in sorted(ranked, key=lambda r: r[:2])[:n]]


def agent(prompt: str, project: Path, home: Path, model: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in SESSION_VARIABLES} | {"KANERVA_HOME": str(home)}
    command = ["claude", "-p", prompt, "--plugin-dir", str(PLUGIN), "--model", model, "--setting-sources", "project", "--no-session-persistence",
               "--allowedTools", TOOLS, "--max-budget-usd", MAX_USD, "--output-format", "stream-json", "--verbose"]
    out = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=240).stdout
    calls, listed, cost, reply = [], [], 0.0, ""
    for line in out.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            calls += [c["name"].split("__")[-1] for c in event["message"].get("content", []) if c.get("type") == "tool_use"]
        elif event.get("type") == "user":
            listed += similar_ids(event)
        elif event.get("type") == "result":
            cost, reply = event.get("total_cost_usd", 0.0), event.get("result", "")
    return {"calls": calls, "listed": listed, "cost": cost, "reply": reply}


def session(trial: dict, folder: Path, models: Path, model: str) -> dict:
    home, project = folder / "kanerva", folder / "project"
    home.mkdir(parents=True)
    project.mkdir()
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    (home / "models").symlink_to(models)
    shutil.move(trial["store"], home / "memory.sqlite")
    with sqlite3.connect(home / "memory.sqlite") as db:
        db.execute("UPDATE items SET scope = ?", (project_scope(str(project)),))
    cost, old = 0.0, list(trial["old"])
    prompt = PROMPTS[trial["mode"]]
    if "first" in trial:
        known = set(texts(home))
        first = agent(prompt.format(note=trial["first"]), project, home, model)
        cost += first["cost"]
        old = sorted(set(texts(home)) - known)  # what the agent wrote is what the next session may replace
    before, closed_before = texts(home), closed(home)
    run = agent(prompt.format(note=trial["note"]), project, home, model)
    after = texts(home)
    retired = sorted(closed(home) - closed_before)
    edited = sorted(i for i, text in before.items() if after.get(i, text) != text)
    written = [after[i] for i in sorted(set(after) - set(before))]
    return {k: trial[k] for k in ("set", "dataset", "stream", "cursor", "changed")} | {"mode_own": "first" in trial} | {
        "old": old, "remembered": "memory_remember" in run["calls"], "listed": run["listed"], "retired": retired, "retired_text": [before[i] for i in retired if i in before],
        "edited": edited, "calls": run["calls"], "cost": cost + run["cost"], "reply": run["reply"],
        "changes": [{"id": i, "before": before[i], "after": after[i]} for i in edited], "older_text": [before[i] for i in old if i in before], "written": written,
    }


def similar_ids(event: dict) -> list[int]:
    ids: list[int] = []
    for block in event.get("message", {}).get("content", []):
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        content = block.get("content")
        texts = [c.get("text", "") for c in content if isinstance(c, dict)] if isinstance(content, list) else [str(content)]
        for text in texts:
            try:
                reply = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(reply, dict):
                ids += [note["id"] for note in reply.get("similar", [])]
    return ids


def texts(home: Path) -> dict[int, str]:
    with sqlite3.connect(home / "memory.sqlite") as db:
        return dict(db.execute("SELECT id, text FROM items"))


def closed(home: Path) -> set[int]:
    with sqlite3.connect(home / "memory.sqlite") as db:
        return {row[0] for row in db.execute("SELECT id FROM items WHERE valid_to IS NOT NULL")}


def judged(row: dict) -> str:
    """What happened to the older version. A note outside the labels is counted apart, by ``extra``."""
    old, touched = set(row["old"]), set(row["retired"]) | set(row["edited"])
    if touched & old:
        return "replaced" if set(row["retired"]) & old else "edited"
    return "kept" if not old or set(row["listed"]) & old else "not_listed"


def extra(row: dict) -> bool:
    """The agent touched a note the labels do not mark. Read these by hand: a label can miss a rename or a second copy."""
    return bool((set(row["retired"]) | set(row["edited"])) - (set(row["old"]) if row["changed"] else set()))


def share_interval(hits: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson interval."""
    if n == 0:
        return [0.0, 0.0]
    p, shift = hits / n, z * z / n
    middle, half = (p + shift / 2) / (1 + shift), z * (p * (1 - p) / n + shift / (4 * n)) ** 0.5 / (1 + shift)
    return [round(middle - half, 3), round(middle + half, 3)]


def summary(rows: list[dict]) -> dict:
    out: dict = {"trials": len(rows), "remembered": sum(r["remembered"] for r in rows)}
    for kind, label in ((True, "changed"), (False, "other")):
        group = [r for r in rows if r["changed"] == kind]
        counts = {v: sum(judged(r) == v for r in group) for v in ("replaced", "edited", "kept", "not_listed")}
        good = counts["replaced"] + counts["edited"] if kind else sum(not extra(r) for r in group)
        out[label] = {"n": len(group), **counts, "extra": sum(extra(r) for r in group), "right": good / max(len(group), 1), "interval": share_interval(good, len(group))}
    return out


def cosines(rows: list[dict]) -> dict:
    """How close an agent's two notes are in the own-words trials: the distance ``similar`` has to bridge."""
    embedder = OnnxEmbedder(minilm_folder())
    out: dict = {"changed": [], "other": []}
    for row in rows:
        if row["mode_own"] and row["older_text"] and row["written"]:
            vectors = embedder.encode([row["older_text"][0], row["written"][0]])
            out["changed" if row["changed"] else "other"].append(round(float(vectors[0] @ vectors[1]), 3))
    return {kind: {"n": len(values), "min": min(values), "median": float(np.median(values)), "max": max(values)} for kind, values in out.items() if values}


def main(per_kind: int | None = None, names: tuple[str, ...] = tuple(SETS)) -> None:
    rng = random.Random("agent-write-path-5")  # earlier samples set the hint and the cosine threshold, so this one is new
    version = subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
    folder = Path(tempfile.mkdtemp(prefix="kanerva-agent-"))
    os.environ["KANERVA_HOME"] = str(folder / "models-home")
    models = folder / "models-home" / "models"
    fetch_minilm()
    trials = []
    with ProcessPoolExecutor(6) as pool:
        for label in names:
            mode, n = SETS[label]
            dataset = "lme-updates" if label == "lme-own-notes" else label
            cache_name, split = REPORT_ON[dataset]
            streams = streams_for(dataset, split)
            rows = [r | {"set": label, "dataset": dataset, "mode": mode} for part in pool.map(candidates, [(s, dataset, cache_name, mode) for s in streams]) for r in part]
            picked = pick([r for r in rows if r["changed"]], per_kind or n, rng) + pick([r for r in rows if not r["changed"]], per_kind or n, rng)
            print(label, "candidates", len(rows), "trials", len(picked), flush=True)
            for index, trial in enumerate(picked):
                trial["store"] = str(folder / f"{label}-{index}.sqlite")
            by_name = {s.name: s for s in streams}
            jobs = defaultdict(list)
            for trial in picked:
                jobs[trial["stream"]].append(trial)
            list(pool.map(snapshots, [(by_name[n], dataset, cache_name, group) for n, group in jobs.items()]))
            trials += picked
    start = time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(lambda pair: session(pair[1], folder / f"trial-{pair[0]}", models, MODEL), enumerate(trials)))
    own = cosines(rows)
    shutil.rmtree(folder, ignore_errors=True)
    for path in (Path.home() / ".claude" / "projects").glob(f"*{folder.name.replace('_', '-')}*"):  # the host's folders for these sessions
        shutil.rmtree(path, ignore_errors=True)
    RAW.write_text(json.dumps(rows))
    report = {
        "meta": {"experiment": "real agent on the write path", "commit": git_commit(), "agent": version, "model": MODEL, "prompts": PROMPTS,
                 "trials": len(rows), "cost_usd": round(sum(r["cost"] for r in rows), 2), "seconds": round(time.time() - start)},
        "sets": {name: summary([r for r in rows if r["set"] == name]) for name in names}, "own_note_cosines": own,
    }
    (RESULTS / "agent_write_path.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else None, tuple(sys.argv[2:]) or tuple(SETS)))
