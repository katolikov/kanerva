"""Agent-host integration: turn hook payloads into memory operations.

``handle(event, payload)`` is host-neutral and returns text to inject, or an empty string. The plugins are
thin adapters over it, and ``hosts`` maps the hook formats of other agents onto it. Hooks must never break a
session, so callers catch every exception and exit cleanly.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

from .consolidate import consolidate
from .memory import Memory, open_store
from .paths import project_scope
from .secrets_filter import redact
from .text import content_terms

SESSION_BUDGET = int(os.environ.get("KANERVA_SESSION_BUDGET", "400"))
PROMPT_BUDGET = int(os.environ.get("KANERVA_PROMPT_BUDGET", "600"))
MIN_WEIGHT = float(os.environ.get("KANERVA_MIN_WEIGHT", "0.02"))
LEARN = os.environ.get("KANERVA_LEARN") == "1"  # policy learning is opt-in: noisy rewards hurt in the experiments
CONSOLIDATE_ON_COMPACT = os.environ.get("KANERVA_CONSOLIDATE") == "1"
WRAPUP = os.environ.get("KANERVA_WRAPUP") != "0"
MAX_CONTEXT_CHARS = 9500  # Claude Code truncates injected strings at 10,000 characters

WRAPUP_EDITS = 3  # below this, with no test run and few commands, a session counts as trivial
WRAPUP_COMMANDS = 12
WRAPUP_AGAIN_EDITS = 20  # a long session is asked again only after this many further edits
WRAPUP_KEEP_DAYS = 30
WRAPUP_PROMPT = (
    "This session did real work. Call memory_remember once with one short note for a later session: what was tried, "
    "what worked and why. Under 60 words, no secrets, no key. If nothing here is worth keeping, skip it. "
    "Then stop, with no further reply."
)

_SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_retrievals (session TEXT NOT NULL, retrieval_id INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS session_events (session TEXT NOT NULL, at REAL NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS session_events_s ON session_events(session);
CREATE INDEX IF NOT EXISTS session_retrievals_s ON session_retrievals(session);
CREATE TABLE IF NOT EXISTS session_work (
    session TEXT PRIMARY KEY,
    edits INTEGER NOT NULL DEFAULT 0,
    commands INTEGER NOT NULL DEFAULT 0,
    tests INTEGER NOT NULL DEFAULT 0,
    asked INTEGER NOT NULL DEFAULT 0,
    at REAL NOT NULL
);
"""

_TEST_COMMAND = re.compile(r"\b(pytest|npm (run )?test|pnpm test|yarn test|cargo test|go test|make test|tox|vitest|jest)\b")
_DISTINCTIVE = re.compile(r"[A-Za-z_][\w./-]*[_./-][\w./-]+|[a-z]+[A-Z]\w+|[A-Z][a-z]+[A-Z]\w+|\b\d{3,}\b")
_PATCH_FILE = re.compile(r"^\*\*\* (?:Add|Update) File: (.+)$", re.M)

EDIT_TOOLS = {
    "Write", "Edit", "MultiEdit", "NotebookEdit", "StrReplace", "write", "edit", "patch", "write_file", "replace",
    "edit_file", "editor", "create", "str_replace", "write_to_file", "replace_file_content", "multi_replace_file_content",
    "fs_write",
}
SHELL_TOOLS = {
    "Bash", "Shell", "bash", "shell", "exec", "terminal", "run_shell_command", "run_command", "execute_bash",
    "execute_cmd",
}


def _open(cwd: str | None) -> tuple[Memory, str]:
    mem = open_store()
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
    mem.db.execute("INSERT INTO session_events VALUES (?, ?, ?, ?)", (session, time.time(), kind, redact(text)[0][:4000]))


def _count_work(mem: Memory, session: str, edits: int = 0, commands: int = 0, tests: int = 0) -> None:
    mem.db.execute(
        "INSERT INTO session_work (session, edits, commands, tests, at) VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(session) DO UPDATE SET edits = edits + excluded.edits, commands = commands + excluded.commands,"
        " tests = tests + excluded.tests, at = excluded.at",
        (session, edits, commands, tests, time.time()),
    )


def _restart_work(mem: Memory, session: str, asked: bool = False) -> None:
    mem.db.execute(
        "UPDATE session_work SET edits = 0, commands = 0, tests = 0, asked = max(asked, ?) WHERE session = ?",
        (int(asked), session),
    )


def wrapup_request(mem: Memory, session: str, payload: dict) -> str:
    """The text that asks the host agent for an outcome note, or "" when this stop should stay quiet.

    Only the agent knows what was tried and why, so the note goes through its own ``memory_remember`` call and meets
    the same secret filter and ranking as any other note. Asking restarts the count, so one stop cannot ask twice.
    """
    mem.db.execute("DELETE FROM session_work WHERE at < ?", (time.time() - WRAPUP_KEEP_DAYS * 86400,))
    unfinished = payload.get("stop_hook_active") or payload.get("background_tasks")
    if not WRAPUP or unfinished or payload.get("status", "completed") != "completed":
        return ""
    row = mem.db.execute("SELECT * FROM session_work WHERE session = ?", (session,)).fetchone()
    if row is None:
        return ""
    if row["asked"]:
        due = row["edits"] >= WRAPUP_AGAIN_EDITS
    else:
        due = row["edits"] >= WRAPUP_EDITS or (row["edits"] > 0 and row["tests"] > 0) or row["commands"] >= WRAPUP_COMMANDS
    if not due:
        return ""
    _restart_work(mem, session, asked=True)
    return WRAPUP_PROMPT


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
    if tool == "apply_patch":
        patch = str(tool_input.get("command") or tool_input.get("input") or tool_input.get("patch") or "")
        _note(mem, session, "edit", patch[:4000])
        paths = _PATCH_FILE.findall(patch)
        _count_work(mem, session, edits=max(len(paths), 1))
        for path in paths:
            mem.write(f"Edited {path.strip()}", scope=scope, kind="episodic", key=None, source={"session": session})
    elif tool in EDIT_TOOLS:
        path = (
            tool_input.get("file_path") or tool_input.get("filePath") or tool_input.get("path")
            or tool_input.get("TargetFile") or ""
        )
        _note(mem, session, "edit", json.dumps(tool_input)[:4000])
        _count_work(mem, session, edits=1)
        if path:
            mem.write(f"Edited {path}", scope=scope, kind="episodic", key=None, source={"session": session})
    elif tool in SHELL_TOOLS:
        command = str(tool_input.get("command") or tool_input.get("CommandLine") or "")
        _note(mem, session, "command", command)
        is_test = bool(_TEST_COMMAND.search(command))
        _count_work(mem, session, commands=1, tests=int(is_test))
        if is_test:
            failed = isinstance(response, dict) and (response.get("interrupted") or response.get("exit_code") not in (0, None))
            stderr = str(response.get("stderr", "")) if isinstance(response, dict) else ""
            failed = bool(failed or re.search(r"\b(failed|error)\b", stderr, re.I))
            _note(mem, session, "test", ("fail: " if failed else "pass: ") + command)
            mem.write(
                f"Test command: {command}", scope=scope, kind="semantic",
                key=f"cmd:test:{command.split()[0]}", source={"session": session},
            )
    elif tool.endswith("memory_remember"):
        _restart_work(mem, session)  # the agent has just written a note of its own
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
            return wrapup_request(mem, session, payload)
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


def stop_output(text: str) -> dict:
    """Keeps the agent going for one more turn. Claude Code labels this feedback, and ``decision`` an error."""
    if os.environ.get("CLAUDECODE"):
        return {"hookSpecificOutput": {"hookEventName": "Stop", "additionalContext": text}}
    return {"decision": "block", "reason": text}


def main(argv: list[str] | None = None) -> int:
    """Claude Code command hook: read the payload on stdin, print the hook's JSON answer."""
    try:
        payload = json.load(sys.stdin)
        name = payload.get("hook_event_name") or (argv or sys.argv[1:] or [""])[0]
        event = EVENT_NAMES.get(name, name)
        context = handle(event, payload)
        if context and name in ("SessionStart", "UserPromptSubmit"):
            print(json.dumps({"hookSpecificOutput": {"hookEventName": name, "additionalContext": context}}))
        elif context and event == "stop":
            print(json.dumps(stop_output(context)))
    except Exception as exc:  # a memory failure must never block the agent
        print(f"kanerva hook: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
