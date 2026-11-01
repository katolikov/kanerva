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
from .learning import Tuning, learning_on, tuning
from .memory import Memory, open_store
from .paths import project_root, project_scope

SESSION_BUDGET = int(os.environ.get("KANERVA_SESSION_BUDGET", "400"))
PROMPT_BUDGET = int(os.environ.get("KANERVA_PROMPT_BUDGET", "600"))
MIN_WEIGHT = float(os.environ.get("KANERVA_MIN_WEIGHT", "0.02"))
GATE = os.environ.get("KANERVA_GATE", "1") != "0"  # say nothing when the store looks unable to answer the prompt
FILE_BUDGET = int(os.environ.get("KANERVA_FILE_BUDGET", "150"))  # tokens of notes when a session first touches a file
FILE_SESSION_BUDGET = 8 * FILE_BUDGET
REEMBED_AT_START = 32
LEARN = os.environ.get("KANERVA_LEARN", "auto")  # "auto" acts once the shadow check of kanerva.learning allows it, "1" forces it, "shadow" only records, "0" stops it
CONSOLIDATE_ON_COMPACT = os.environ.get("KANERVA_CONSOLIDATE") == "1"
WRAPUP = os.environ.get("KANERVA_WRAPUP") != "0"
MAX_CONTEXT_CHARS = 9500  # Claude Code truncates injected strings at 10,000 characters

WRAPUP_EDITS = 3  # below this, with no test run and few commands, a session counts as trivial
WRAPUP_COMMANDS = 12
WRAPUP_AGAIN_EDITS = 20  # a long session is asked again only after this many further edits
WRAPUP_KEEP_DAYS = 30
WRAPUP_VERDICT = (
    "First call memory_feedback once with the ids of notes shown to you in this session: helped for notes that changed what "
    "you did or saved you work, misled for notes that turned out wrong or out of date. Notes you did not rely on go in neither "
    "list, and both lists may be empty. Then "
)
WRAPUP_NOTE = (
    "Call memory_remember once with one short note for a later session: what was tried, what worked and why. Under 60 words, "
    "no secrets, no key. If nothing here is worth keeping, skip it. If the result lists a note yours replaces, do what its hint "
    "says. Then stop, with no further reply."
)
WRAPUP_PROMPT = "This session did real work. " + (WRAPUP_NOTE if LEARN == "0" else WRAPUP_VERDICT + WRAPUP_NOTE[0].lower() + WRAPUP_NOTE[1:])

_SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_retrievals (session TEXT NOT NULL, retrieval_id INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS session_retrievals_s ON session_retrievals(session);
CREATE TABLE IF NOT EXISTS session_credits (session TEXT NOT NULL, item_id INTEGER NOT NULL, at REAL NOT NULL, PRIMARY KEY (session, item_id));
CREATE TABLE IF NOT EXISTS session_files (
    session TEXT NOT NULL,
    path TEXT NOT NULL,
    at REAL NOT NULL,
    tokens INTEGER NOT NULL DEFAULT 0,
    items TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (session, path)
);
CREATE TABLE IF NOT EXISTS session_work (
    session TEXT PRIMARY KEY,
    edits INTEGER NOT NULL DEFAULT 0,
    commands INTEGER NOT NULL DEFAULT 0,
    tests INTEGER NOT NULL DEFAULT 0,
    asked INTEGER NOT NULL DEFAULT 0,
    at REAL NOT NULL
);
"""

_TEST_COMMAND = re.compile(
    r"\b(pytest|npm (run )?test|pnpm test|yarn test|cargo test|go test|make (test|check)|ninja (test|check)|ctest|meson test|"
    r"bazel test|tox|vitest|jest)\b|--target (test|check)\b"
)
_PATCH_FILE = re.compile(r"^\*\*\* (?:Add|Update) File: (.+)$", re.M)
_NOT_RUN = re.compile(r"command not found|No module named")
_BUILD_COMMAND = re.compile(r"(?:^|[;&|(]\s*)(?:\w+=\S+\s+)*(cmake|ninja|make|meson|bazel|xcodebuild|msbuild|gcc|g\+\+|clang\+\+|clang)(?=\s|$)")
_BUILD_FAILED = re.compile(r"\berror:|\bError \d+|\bFAILED\b|CMake Error")

EDIT_TOOLS = {
    "Write", "Edit", "MultiEdit", "NotebookEdit", "StrReplace", "write", "edit", "patch", "write_file", "replace",
    "edit_file", "editor", "create", "str_replace", "write_to_file", "replace_file_content", "multi_replace_file_content",
    "fs_write",
}
READ_TOOLS = {"Read", "read", "read_file", "view", "view_file", "ReadFile", "fs_read"}
SHELL_TOOLS = {
    "Bash", "Shell", "bash", "shell", "exec", "terminal", "run_shell_command", "run_command", "execute_bash",
    "execute_cmd",
}


def _open(cwd: str | None, embed: bool = True) -> tuple[Memory, str]:
    mem = open_store(embed=embed)
    scope = project_scope(cwd)
    mem.config = mem.config.with_(use_bandit=False, learn=False, item_lift=LEARN == "1" or (LEARN == "auto" and learning_on(mem.db, scope)))
    mem.tuning = tuning(mem.db, scope, WRAPUP and not headless()) if LEARN in ("auto", "1") else Tuning()
    mem.db.executescript(_SESSION_SCHEMA)
    return mem, scope


def format_context(result, header: str) -> str:
    if not result.items:
        return ""
    items = sorted(result.items, key=lambda item: -item.created_at)  # newest first, so that updates read as updates
    lines = [header] + [item.render() for item in items]
    return "\n".join(lines)[:MAX_CONTEXT_CHARS]


def _inject(mem: Memory, session: str, query: str, scope: str, budget: int, header: str, gated: bool = False, opening: bool = False) -> str:
    result = mem.retrieve(query, budget - mem.count_tokens(header), scope=scope, min_weight=MIN_WEIGHT, gated=gated)
    if opening:  # with no prompt to go on, a list of edited files only repeats the git log
        result.items = [item for item in result.items if not item.text.startswith("Edited ")]
        if result.retrieval_id is not None and not mem.narrow_retrieval(result.retrieval_id, [item.id for item in result.items]):
            result.retrieval_id = None
    if result.retrieval_id is not None:
        mem.db.execute("INSERT INTO session_retrievals VALUES (?, ?)", (session, result.retrieval_id))
    return format_context(result, header)


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


def headless() -> bool:
    """`claude -p` hands its last message to a script, and a wrap-up turn would replace that message."""
    return os.environ.get("CLAUDE_CODE_ENTRYPOINT", "").startswith("sdk") and os.environ.get("KANERVA_WRAPUP") != "1"


def wrapup_request(mem: Memory, session: str, payload: dict) -> str:
    """The text that asks the host agent for an outcome note, or "" when this stop should stay quiet.

    Only the agent knows what was tried and why, so the note goes through its own ``memory_remember`` call and meets
    the same secret filter and ranking as any other note. Asking restarts the count, so one stop cannot ask twice.
    """
    old = time.time() - WRAPUP_KEEP_DAYS * 86400
    mem.db.execute("DELETE FROM session_work WHERE at < ?", (old,))
    mem.db.execute("DELETE FROM session_files WHERE at < ?", (old,))
    mem.db.execute("DELETE FROM session_retrievals WHERE retrieval_id IN (SELECT id FROM retrievals WHERE at < ?)", (old,))
    mem.db.execute("DELETE FROM session_credits WHERE at < ?", (old,))
    unfinished = payload.get("stop_hook_active") or payload.get("background_tasks")
    if not WRAPUP or headless() or unfinished or payload.get("status", "completed") != "completed":
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


def resolve_feedback(mem: Memory, session: str, scope: str) -> int:
    """Turn the agent's verdict into a record for each note this session saw. A session it did not judge says nothing.

    The MCP server does not know the session, so a verdict is tied to its project, and a session leaves one that names only what
    another open session saw. Each note counts once for a session, however many times the agent names it.
    """
    verdict = mem.open_verdict(scope) if LEARN != "0" else None
    if verdict is None:
        return 0
    at, helped, misled = verdict
    rows = mem.db.execute(
        "SELECT r.id, r.at, r.item_ids, r.resolved FROM retrievals r JOIN session_retrievals s ON s.retrieval_id = r.id WHERE s.session = ?", (session,),
    ).fetchall()
    shown = {i for row in rows for i in json.loads(row["item_ids"])}
    beside = {i for r in mem.db.execute("SELECT items FROM session_files WHERE session = ?", (session,)) for i in json.loads(r["items"])}
    others = {i for r in mem.db.execute(
        "SELECT r.item_ids FROM retrievals r JOIN session_retrievals s ON s.retrieval_id = r.id WHERE s.session != ? AND r.resolved = 0", (session,),
    ) for i in json.loads(r["item_ids"])}
    if (helped | misled) & others and not (helped | misled) & (shown | beside):
        return 0  # it names what another open session saw
    pending = [row for row in rows if not row["resolved"] and row["at"] <= at]
    mem.shadow_check(scope, [row["id"] for row in pending], helped)
    fresh = [i for i in helped - shown if mem.db.execute("INSERT OR IGNORE INTO session_credits VALUES (?, ?, ?)", (session, i, at)).rowcount]
    mem.credit(fresh, misled - shown)
    updated = sum(mem.feedback(row["id"], {i: 1.0 for i in json.loads(row["item_ids"]) if i in helped}, misled, session, save=False) for row in pending)
    mem.close_verdict(scope, at)
    mem.commit()
    return updated


def _patch_text(tool_input: dict) -> str:
    return str(tool_input.get("command") or tool_input.get("input") or tool_input.get("patch") or "")


def _tool_path(tool_input: dict) -> str:
    return (
        tool_input.get("file_path") or tool_input.get("filePath") or tool_input.get("path")
        or tool_input.get("TargetFile") or ""
    )


def project_file(path: str, cwd: str | None) -> str | None:
    """A path as the project names it, or None for a file outside it, such as a host's own memory folder."""
    if not cwd or not os.path.isabs(path):
        return path
    inside = os.path.relpath(os.path.realpath(path), os.path.realpath(project_root(cwd)))
    return None if inside.startswith("..") else inside


def touched_files(tool: str, tool_input: dict) -> list[str]:
    if tool == "apply_patch":
        return [path.strip() for path in _PATCH_FILE.findall(_patch_text(tool_input))]
    if tool in EDIT_TOOLS or tool in READ_TOOLS:
        return [path] if (path := _tool_path(tool_input)) else []
    return []


def file_notes(mem: Memory, session: str, scope: str, cwd: str | None, paths: list[str]) -> str:
    """Notes that name the files this session touches for the first time.

    The moment an agent opens a file is the moment notes about it help. Each file is looked up once per session, and
    the session's total stays bounded however many files it reads.
    """
    if FILE_BUDGET <= 0:
        return ""
    lines: list[str] = []
    row = mem.db.execute("SELECT coalesce(sum(tokens), 0) AS spent FROM session_files WHERE session = ?", (session,)).fetchone()
    spent = row["spent"]
    seen = {i for r in mem.db.execute("SELECT items FROM session_files WHERE session = ?", (session,)) for i in json.loads(r["items"])}
    injected = "SELECT item_ids FROM retrievals WHERE id IN (SELECT retrieval_id FROM session_retrievals WHERE session = ?)"
    seen.update(i for r in mem.db.execute(injected, (session,)) for i in json.loads(r["item_ids"]))
    for path in paths:
        name = project_file(path, cwd)
        if name is None:
            continue
        known = mem.db.execute("SELECT 1 FROM session_files WHERE session = ? AND path = ?", (session, name)).fetchone()
        if known:
            continue
        budget = min(FILE_BUDGET, FILE_SESSION_BUDGET - spent)
        items = mem.about_file(name, budget, scope, seen) if budget > 0 else []
        tokens = sum(item.tokens + mem.config.item_overhead for item in items)
        mem.db.execute("INSERT INTO session_files VALUES (?, ?, ?, ?, ?)",
                       (session, name, time.time(), tokens, json.dumps([item.id for item in items])))
        if items:
            lines.append(f"Notes that mention {name}:")
            lines.extend(item.render() for item in items)
            seen.update(item.id for item in items)
            spent += tokens
    return "\n".join(lines)[:MAX_CONTEXT_CHARS]


def on_tool(mem: Memory, session: str, scope: str, tool: str, tool_input: dict, response, cwd: str | None = None) -> None:
    if tool == "apply_patch":
        paths = [path for path in (project_file(p, cwd) for p in touched_files(tool, tool_input)) if path]
        _count_work(mem, session, edits=max(len(paths), 1))
        for path in paths:
            mem.write(f"Edited {path}", scope=scope, kind="episodic", key=None, source={"session": session})
    elif tool in EDIT_TOOLS:
        path = project_file(_tool_path(tool_input), cwd)
        if path is None:
            return
        _count_work(mem, session, edits=1)
        if path:
            mem.write(f"Edited {path}", scope=scope, kind="episodic", key=None, source={"session": session})
    elif tool in SHELL_TOOLS:
        command = str(tool_input.get("command") or tool_input.get("CommandLine") or "")
        output = str(response)
        is_test = bool(_TEST_COMMAND.search(command)) and not _NOT_RUN.search(output)
        _count_work(mem, session, commands=1, tests=int(is_test))
        if is_test:
            mem.write(
                f"Test command: {command}", scope=scope, kind="semantic",
                key=f"cmd:test:{command.split()[0]}", source={"session": session},
            )
        elif (build := _BUILD_COMMAND.search(command)) and not _NOT_RUN.search(output) and not _BUILD_FAILED.search(output):
            mem.write(f"Build command: {command}", scope=scope, kind="semantic", key=f"cmd:build:{build.group(1)}", source={"session": session})
    elif tool.endswith("memory_remember"):
        _restart_work(mem, session)  # the agent has just written a note of its own


def _needs_vectors(event: str, payload: dict) -> bool:
    """Loading the model takes a tenth of a second, which a tool call that writes no note should not pay."""
    if event in ("stop", "compact"):
        return CONSOLIDATE_ON_COMPACT
    if event != "tool":
        return True
    tool, tool_input = str(payload.get("tool_name", "")), payload.get("tool_input") or {}
    command = str(tool_input.get("command") or tool_input.get("CommandLine") or "")
    commands = _TEST_COMMAND.search(command) or _BUILD_COMMAND.search(command)
    return tool in EDIT_TOOLS or tool == "apply_patch" or (tool in SHELL_TOOLS and bool(commands))


def handle(event: str, payload: dict) -> str:
    session = str(payload.get("session_id", "default"))
    mem, scope = _open(payload.get("cwd"), embed=_needs_vectors(event, payload))
    try:
        if event == "session_start":
            mem.reembed(limit=REEMBED_AT_START)  # notes from a process that started before the model arrived
            source = payload.get("source", "startup")
            query = "project conventions decisions commands architecture bugs"
            return _inject(mem, session, query, scope, SESSION_BUDGET,
                           f"Relevant notes from earlier sessions ({source}); each line shows its id and date:", opening=True)
        if event == "prompt":
            prompt = str(payload.get("prompt", ""))
            if len(prompt.strip()) < 8 or prompt.lstrip().startswith("/"):
                return ""
            return _inject(mem, session, prompt, scope, PROMPT_BUDGET, "Possibly relevant notes from earlier sessions:", GATE)
        if event == "tool":
            tool, tool_input = str(payload.get("tool_name", "")), payload.get("tool_input") or {}
            on_tool(mem, session, scope, tool, tool_input, payload.get("tool_response"), payload.get("cwd"))
            return file_notes(mem, session, scope, payload.get("cwd"), touched_files(tool, tool_input))
        if event == "stop":
            resolve_feedback(mem, session, scope)
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


def stop_output(text: str, payload: dict) -> dict:
    """Keeps the agent going for one more turn. Claude Code labels this feedback, and ``decision`` an error.

    Codex rejects that form, and started from a Claude Code session it inherits ``CLAUDECODE``, so its ``turn_id`` decides.
    """
    if os.environ.get("CLAUDECODE") and "turn_id" not in payload:
        return {"hookSpecificOutput": {"hookEventName": "Stop", "additionalContext": text}}
    return {"decision": "block", "reason": text}


def main(argv: list[str] | None = None) -> int:
    """Claude Code command hook: read the payload on stdin, print the hook's JSON answer."""
    try:
        payload = json.load(sys.stdin)
        name = payload.get("hook_event_name") or (argv or sys.argv[1:] or [""])[0]
        event = EVENT_NAMES.get(name, name)
        context = handle(event, payload)
        if context and name in ("SessionStart", "UserPromptSubmit", "PostToolUse"):
            print(json.dumps({"hookSpecificOutput": {"hookEventName": name, "additionalContext": context}}))
        elif context and event == "stop":
            print(json.dumps(stop_output(context, payload)))
    except Exception as exc:  # a memory failure must never block the agent
        print(f"kanerva hook: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
