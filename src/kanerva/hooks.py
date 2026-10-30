"""Agent-host integration: turn hook payloads into memory operations.

``handle(event, payload)`` is host-neutral and returns text to inject, or an empty string. The Claude Code
and OpenCode plugins are thin adapters over it. Hooks must never break a session, so callers catch every
exception and exit cleanly.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

from .consolidate import consolidate
from .memory import Memory
from .paths import db_path, project_scope
from .text import content_terms

SESSION_BUDGET = int(os.environ.get("KANERVA_SESSION_BUDGET", "400"))
PROMPT_BUDGET = int(os.environ.get("KANERVA_PROMPT_BUDGET", "600"))
MIN_WEIGHT = float(os.environ.get("KANERVA_MIN_WEIGHT", "0.04"))
LEARN = os.environ.get("KANERVA_LEARN") == "1"  # policy learning is opt-in: noisy rewards hurt in the experiments
CONSOLIDATE_ON_COMPACT = os.environ.get("KANERVA_CONSOLIDATE") == "1"
MAX_CONTEXT_CHARS = 9500  # Claude Code truncates injected strings at 10,000 characters

_SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_retrievals (session TEXT NOT NULL, retrieval_id INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS session_events (session TEXT NOT NULL, at REAL NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS session_events_s ON session_events(session);
CREATE INDEX IF NOT EXISTS session_retrievals_s ON session_retrievals(session);
"""

_TEST_COMMAND = re.compile(r"\b(pytest|npm (run )?test|pnpm test|yarn test|cargo test|go test|make test|tox|vitest|jest)\b")
_DISTINCTIVE = re.compile(r"[A-Za-z_][\w./-]*[_./-][\w./-]+|[a-z]+[A-Z]\w+|[A-Z][a-z]+[A-Z]\w+|\b\d{3,}\b")


def _open(cwd: str | None) -> tuple[Memory, str]:
    mem = Memory(db_path())
    mem.config = mem.config.with_(learn=LEARN)
    mem.db.executescript(_SESSION_SCHEMA)
    return mem, project_scope(cwd)


def format_context(result, header: str) -> str:
    if not result.items:
        return ""
    items = sorted(result.items, key=lambda item: -item.created_at)  # newest first, so that updates read as updates
    lines = [header] + [item.render() for item in items]
    return "\n".join(lines)[:MAX_CONTEXT_CHARS]


def _inject(mem: Memory, session: str, query: str, scope: str, budget: int, header: str) -> str:
    result = mem.retrieve(query, budget - mem.count_tokens(header), scope=scope, min_weight=MIN_WEIGHT)
    if result.retrieval_id is not None:
        mem.db.execute("INSERT INTO session_retrievals VALUES (?, ?)", (session, result.retrieval_id))
    return format_context(result, header)


def _note(mem: Memory, session: str, kind: str, text: str) -> None:
    mem.db.execute("INSERT INTO session_events VALUES (?, ?, ?, ?)", (session, time.time(), kind, text[:4000]))


def distinctive_tokens(text: str) -> set[str]:
    tokens = set(_DISTINCTIVE.findall(text))
    if not tokens:
        tokens = {t for t in content_terms(text) if len(t) >= 6}
    return tokens


def item_used(item_text: str, observed: str) -> bool:
    """An item counts as used when its distinctive tokens show up in later tool calls or the reply."""
    tokens = distinctive_tokens(item_text)
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in observed)
    return hits >= 1 if len(tokens) <= 2 else hits >= 2


def resolve_feedback(mem: Memory, session: str) -> int:
    """Turn the session's execution trace into per-item rewards for its logged retrievals."""
    events = mem.db.execute("SELECT kind, text FROM session_events WHERE session=? ORDER BY at", (session,)).fetchall()
    observed = "\n".join(e["text"] for e in events if e["kind"] in ("edit", "command", "tool"))
    tests = [e["text"] for e in events if e["kind"] == "test"]
    failed_last = bool(tests) and tests[-1].startswith("fail")
    rows = mem.db.execute("SELECT retrieval_id FROM session_retrievals WHERE session=?", (session,)).fetchall()
    updated = 0
    for row in rows:
        retrieval = mem.db.execute("SELECT * FROM retrievals WHERE id=?", (row["retrieval_id"],)).fetchone()
        if retrieval is None or retrieval["resolved"]:
            continue
        rewards: dict[int, float] = {}
        for item_id in json.loads(retrieval["item_ids"]):
            item = mem.get(item_id)
            if item is not None and item_used(item.text, observed):
                rewards[item_id] = 0.5 if failed_last else 1.0
        updated += mem.feedback(row["retrieval_id"], rewards)
    mem.db.execute("DELETE FROM session_retrievals WHERE session=?", (session,))
    mem.db.execute("DELETE FROM session_events WHERE session=?", (session,))
    return updated


def on_tool(mem: Memory, session: str, scope: str, tool: str, tool_input: dict, response) -> None:
    if tool in {"Write", "Edit", "MultiEdit", "NotebookEdit", "write", "edit", "patch"}:
        path = tool_input.get("file_path") or tool_input.get("filePath") or tool_input.get("path") or ""
        _note(mem, session, "edit", json.dumps(tool_input)[:4000])
        if path:
            mem.write(f"Edited {path}", scope=scope, kind="episodic", key=None, source={"session": session})
    elif tool in {"Bash", "bash", "shell"}:
        command = str(tool_input.get("command", ""))
        _note(mem, session, "command", command)
        if _TEST_COMMAND.search(command):
            failed = isinstance(response, dict) and (response.get("interrupted") or response.get("exit_code") not in (0, None))
            stderr = str(response.get("stderr", "")) if isinstance(response, dict) else ""
            failed = bool(failed or re.search(r"\b(failed|error)\b", stderr, re.I))
            _note(mem, session, "test", ("fail: " if failed else "pass: ") + command)
            mem.write(
                f"Test command: {command}", scope=scope, kind="semantic",
                key=f"cmd:test:{command.split()[0]}", source={"session": session},
            )
    else:
        _note(mem, session, "tool", f"{tool} {json.dumps(tool_input)[:1000]}")


def handle(event: str, payload: dict) -> str:
    session = str(payload.get("session_id", "default"))
    mem, scope = _open(payload.get("cwd"))
    try:
        if event == "session_start":
            source = payload.get("source", "startup")
            query = "project conventions decisions commands architecture bugs"
            return _inject(mem, session, query, scope, SESSION_BUDGET,
                           f"Relevant notes from earlier sessions ({source}); each line shows its id and date:")
        if event == "prompt":
            prompt = str(payload.get("prompt", ""))
            if len(prompt.strip()) < 8 or prompt.lstrip().startswith("/"):
                return ""
            _note(mem, session, "prompt", prompt)
            return _inject(mem, session, prompt, scope, PROMPT_BUDGET, "Possibly relevant notes from earlier sessions:")
        if event == "tool":
            on_tool(mem, session, scope, str(payload.get("tool_name", "")), payload.get("tool_input") or {},
                    payload.get("tool_response"))
            return ""
        if event == "stop":
            message = str(payload.get("last_assistant_message", ""))
            if message:
                _note(mem, session, "reply", message)
            resolve_feedback(mem, session)
            return ""
        if event == "compact":
            if CONSOLIDATE_ON_COMPACT:
                consolidate(mem, scope=scope)
            return ""
        if event == "compact_summary":
            summary = str(payload.get("compact_summary", ""))
            if summary:
                mem.write(summary, scope=scope, kind="semantic", source={"session": session, "compact": True})
            return ""
        return ""
    finally:
        mem.close()


EVENT_NAMES = {
    "SessionStart": "session_start",
    "UserPromptSubmit": "prompt",
    "PostToolUse": "tool",
    "Stop": "stop",
    "PreCompact": "compact",
    "PostCompact": "compact_summary",
}


def main(argv: list[str] | None = None) -> int:
    """Claude Code command hook: read the payload on stdin, print additionalContext JSON."""
    try:
        payload = json.load(sys.stdin)
        name = payload.get("hook_event_name") or (argv or sys.argv[1:] or [""])[0]
        event = EVENT_NAMES.get(name, name)
        context = handle(event, payload)
        if context and name in ("SessionStart", "UserPromptSubmit"):
            print(json.dumps({"hookSpecificOutput": {"hookEventName": name, "additionalContext": context}}))
    except Exception as exc:  # a memory failure must never block the agent
        print(f"kanerva hook: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
