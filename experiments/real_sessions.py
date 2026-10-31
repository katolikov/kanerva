"""Real prompts: the gate and the implicit use signal, on the Claude Code transcripts of this machine.

Every benchmark prompt comes from a template or a dataset, and the labels of ``abstain`` come from foreign questions. Here each
project's real sessions are replayed through the hooks in time order, into a throwaway store, and each real prompt asks that
store for notes, with and without the gate. Prompts from other projects stand in for prompts the store cannot answer.

The sessions never had these notes, which gives two things no benchmark has:

- A weak label. The store could have helped when one of its notes names a file that this turn then touched.
- A clean false-positive rate for the use signal. ``item_used`` cannot be right on a note the agent never saw, so each time
  it fires here it is wrong, and a random note of the same store shows how much of that is chance.

The store of a real session holds what the hooks write. Host notes are stood in for: where the wrap-up rule asks for a note, the
turn's last reply, cut to the 60 words the request allows, is written instead. Notes all carry the replay's own time.

Prompts stay in memory. results/real_sessions.json holds counts only, with no text and no project names.
Usage: python -m experiments.real_sessions [transcripts folder]
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.harness import RESULTS, git_commit
from kanerva import hooks
from kanerva.embed import fetch_minilm

TRANSCRIPTS = Path.home() / ".claude" / "projects"
SKIP = re.compile(r"^-private-|observer-sessions")  # scratch folders, and sessions run by tools
MIN_PROMPTS = 10
MAX_CHARS = 4000
FOREIGN = 60
SESSION_QUERY = "project conventions decisions commands architecture bugs"
_CYRILLIC = re.compile(r"[а-яё]", re.I)
_LATIN = re.compile(r"[a-z]", re.I)


def prompt_text(entry: dict) -> str | None:
    if entry.get("type") != "user" or entry.get("isMeta") or entry.get("isCompactSummary") or entry.get("isSidechain"):
        return None
    content = entry.get("message", {}).get("content")
    if isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return None
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    if not isinstance(content, str) or len(content.strip()) < 8 or content.lstrip().startswith(("<", "/")):
        return None
    return content.strip()[:MAX_CHARS]


def events_of(path: Path) -> list[dict]:
    events, pending = [], {}
    with open(path, errors="replace") as lines:
        for line in lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            cwd = entry.get("cwd")
            if entry.get("isCompactSummary"):
                content = entry.get("message", {}).get("content")
                events.append({"kind": "summary", "text": content if isinstance(content, str) else ""})
            if (text := prompt_text(entry)) is not None:
                events.append({"kind": "prompt", "text": text, "cwd": cwd})
            elif entry.get("type") == "assistant" and not entry.get("isSidechain"):
                for block in entry.get("message", {}).get("content") or []:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use":
                        pending[block.get("id")] = (block.get("name", ""), block.get("input") or {}, cwd)
                    elif block.get("type") == "text":
                        events.append({"kind": "reply", "text": block.get("text", "")})
            elif entry.get("type") == "user":
                content = entry.get("message", {}).get("content")
                for block in content if isinstance(content, list) else []:
                    if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id") in pending:
                        name, tool_input, at = pending.pop(block["tool_use_id"])
                        events.append({"kind": "tool", "name": name, "input": tool_input, "response": entry.get("toolUseResult"), "cwd": at})
    return events


def sessions_of(folder: Path) -> list[tuple[str, list[dict]]]:
    found = []
    for path in folder.glob("*.jsonl"):
        events = events_of(path)
        if any(e["kind"] == "prompt" for e in events):
            found.append((path.stat().st_mtime, path.stem, events))
    return [(name, events) for _, name, events in sorted(found, key=lambda s: s[0])]


def language(text: str) -> str:
    return "ru" if len(_CYRILLIC.findall(text)) > len(_LATIN.findall(text)) else "en"


def turns(events: list[dict]) -> list[tuple[dict, list[dict]]]:
    out, current = [], None
    for event in events:
        if event["kind"] == "prompt":
            current = (event, [])
            out.append(current)
        elif current is not None:
            current[1].append(event)
    return out


def named_files(mem, scope: str) -> dict[str, str]:
    """The files the store's notes name: base name to one note's id, only for names of some length."""
    names: dict[str, str] = {}
    for row in mem.db.execute("SELECT text FROM items WHERE scope = ? AND tier = 'active'", (scope,)):
        for token in re.findall(r"[\w./-]+\.\w{1,6}\b", row["text"]):
            names[os.path.basename(token)] = token
    return names


def touched(tools: list[dict], cwd: str | None) -> set[str]:
    files: set[str] = set()
    for event in tools:
        if event["kind"] == "tool":
            files.update(os.path.basename(p) for p in hooks.touched_files(event["name"], event["input"]))
    return files


def observed_text(tools: list[dict]) -> str:
    parts = []
    for event in tools:
        if event["kind"] == "tool":
            parts.append(f"{event['name']} {json.dumps(event['input'], ensure_ascii=False)[:4000]}")
        elif event["kind"] == "reply":
            parts.append(event["text"])
    return "\n".join(parts)


def replay(project: list[tuple[str, list[dict]]], foreign: list[str], rng: random.Random) -> dict:
    """One project in time order. Returns one row per own prompt and one per foreign prompt."""
    cwd = next(e["cwd"] for _, events in project for e in events if e.get("cwd"))
    mem, scope = hooks._open(cwd)
    own, others, starts = [], [], []
    for session, events in project:
        opening = mem.retrieve(SESSION_QUERY, hooks.SESSION_BUDGET, scope=scope, min_weight=hooks.MIN_WEIGHT, record=False)
        if opening.items:
            starts.append(sum(i.text.startswith("Edited ") for i in opening.items) / len(opening.items))
        for event in events:
            if event["kind"] == "summary" and event["text"]:
                mem.write(event["text"], scope=scope, kind="semantic", source={"session": session})
        for prompt, tools in turns(events):
            text = prompt["text"]
            free = mem.retrieve(text, hooks.PROMPT_BUDGET, scope=scope, min_weight=hooks.MIN_WEIGHT, record=False)
            gated = mem.retrieve(text, hooks.PROMPT_BUDGET, scope=scope, min_weight=hooks.MIN_WEIGHT, gated=True, record=False)
            files, names = touched(tools, prompt.get("cwd")), named_files(mem, scope)
            known = files & set(names)
            observed = observed_text(tools)
            ids = [i.id for i in free.items]
            active = [r["id"] for r in mem.db.execute("SELECT id FROM items WHERE scope = ? AND tier = 'active'", (scope,))]
            random_ids = rng.sample(active, min(len(ids), len(active)))

            def names_a_file(items) -> bool:
                return any(os.path.basename(t) in known for item in items for t in re.findall(r"[\w./-]+\.\w{1,6}\b", item.text))

            own.append({
                "language": language(text), "known": bool(known), "free_tokens": free.tokens, "gated_tokens": gated.tokens,
                "free_hit": names_a_file(free.items), "gated_hit": names_a_file(gated.items),
                "used": sum(hooks.item_used(i.text, observed) for i in free.items), "shown": len(free.items),
                "random_used": sum(hooks.item_used(mem.get(i).text, observed) for i in random_ids), "random_shown": len(random_ids),
            })
            for event in tools:
                if event["kind"] == "tool":
                    hooks.on_tool(mem, session, scope, event["name"], event["input"], event["response"], event.get("cwd"))
            replies = [e["text"] for e in tools if e["kind"] == "reply"]
            if replies and hooks.wrapup_request(mem, session, {}):
                mem.write(" ".join(replies[-1].split()[:60]), scope=scope, kind="semantic", source={"session": session})
            mem.commit()
    for text in rng.sample(foreign, min(FOREIGN, len(foreign))):
        free = mem.retrieve(text, hooks.PROMPT_BUDGET, scope=scope, min_weight=hooks.MIN_WEIGHT, record=False)
        gated = mem.retrieve(text, hooks.PROMPT_BUDGET, scope=scope, min_weight=hooks.MIN_WEIGHT, gated=True, record=False)
        others.append({"language": language(text), "free_tokens": free.tokens, "gated_tokens": gated.tokens})
    notes = mem.stats()["active"]
    mem.close()
    return {"own": own, "foreign": others, "starts": starts, "notes": notes}


def spoke(rows: list[dict], key: str) -> float | None:
    return round(float(np.mean([r[key] > 0 for r in rows])), 3) if rows else None


def summary(parts: list[dict]) -> dict:
    own = [r for p in parts for r in p["own"]]
    others = [r for p in parts for r in p["foreign"]]
    out: dict = {"notes": sum(p["notes"] for p in parts)}
    for name, rows in (("all", own), ("ru", [r for r in own if r["language"] == "ru"]), ("en", [r for r in own if r["language"] == "en"])):
        known = [r for r in rows if r["known"]]
        out[name] = {
            "prompts": len(rows), "free_spoke": spoke(rows, "free_tokens"), "gated_spoke": spoke(rows, "gated_tokens"),
            "free_tokens": round(float(np.mean([r["free_tokens"] for r in rows])), 1) if rows else None,
            "gated_tokens": round(float(np.mean([r["gated_tokens"] for r in rows])), 1) if rows else None,
            "known": len(known), "known_free_spoke": spoke(known, "free_tokens"), "known_gated_spoke": spoke(known, "gated_tokens"),
            "known_free_hit": round(float(np.mean([r["free_hit"] for r in known])), 3) if known else None,
            "known_gated_hit": round(float(np.mean([r["gated_hit"] for r in known])), 3) if known else None,
        }
    shown, random_shown = sum(r["shown"] for r in own), sum(r["random_shown"] for r in own)
    out["signal"] = {
        "notes": shown, "fired": round(sum(r["used"] for r in own) / max(shown, 1), 3),
        "random_notes": random_shown, "random_fired": round(sum(r["random_used"] for r in own) / max(random_shown, 1), 3),
    }
    out["foreign"] = {name: {"prompts": len(rows), "free_spoke": spoke(rows, "free_tokens"), "gated_spoke": spoke(rows, "gated_tokens")}
                      for name, rows in (("all", others), ("ru", [r for r in others if r["language"] == "ru"]), ("en", [r for r in others if r["language"] == "en"]))}
    starts = [v for p in parts for v in p["starts"]]
    out["session_start_share_of_edit_lines"] = round(float(np.mean(starts)), 3) if starts else None
    return out


def main(folder: Path = TRANSCRIPTS) -> None:
    projects = {}
    for path in sorted(folder.iterdir()):
        if path.is_dir() and not SKIP.search(path.name):
            sessions = sessions_of(path)
            if sum(e["kind"] == "prompt" for _, events in sessions for e in events) >= MIN_PROMPTS:
                projects[path.name] = sessions
    prompts = {name: [e["text"] for _, events in s for e in events if e["kind"] == "prompt"] for name, s in projects.items()}
    work = Path(tempfile.mkdtemp(prefix="kanerva-real-"))
    os.environ["KANERVA_HOME"] = str(work / "models-home")
    models = fetch_minilm().parent
    report: dict = {"meta": {"experiment": "real sessions", "commit": git_commit(), "projects": len(projects), "sessions": sum(len(s) for s in projects.values()),
                             "prompts": sum(len(p) for p in prompts.values()), "prompt_budget": hooks.PROMPT_BUDGET, "foreign_per_project": FOREIGN}}
    for arm, embedder in (("vectors", ""), ("lexical", "off")):
        parts = defaultdict(list)
        for index, (name, sessions) in enumerate(projects.items()):
            home = work / f"{arm}-{index}"
            home.mkdir()
            (home / "models").symlink_to(models)
            os.environ["KANERVA_HOME"] = str(home)
            os.environ["KANERVA_EMBEDDER"] = embedder
            foreign = [p for other, texts in prompts.items() if other != name for p in texts]
            parts[arm].append(replay(sessions, foreign, random.Random(f"{arm}-{index}")))
            print(arm, index, len(prompts[name]), flush=True)
        report[arm] = summary(parts[arm])
    shutil.rmtree(work, ignore_errors=True)
    (RESULTS / "real_sessions.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else TRANSCRIPTS))
