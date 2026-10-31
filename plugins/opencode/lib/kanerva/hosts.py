"""Hook formats of agent hosts.

Hosts name their events and fields differently and read different output. ``run`` translates one host event into a
``hooks.handle`` call and renders the result in that host's format. Hosts that use the Claude Code format
(Claude Code, Codex, Devin, Qwen Code) go through ``hooks.main`` instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable

from .hooks import handle
from .paths import home

EVENTS = {
    "copilot": {
        "sessionStart": "session_start", "userPromptTransformed": "prompt", "postToolUse": "tool",
        "agentStop": "stop", "preCompact": "compact",
    },
    "cursor": {
        "sessionStart": "session_start", "beforeSubmitPrompt": "prompt", "postToolUse": "tool",
        "stop": "stop", "preCompact": "compact",
    },
    "gemini": {
        "SessionStart": "session_start", "BeforeAgent": "prompt", "AfterTool": "tool",
        "AfterAgent": "stop", "PreCompress": "compact",
    },
    "antigravity": {"SessionStart": "session_start", "PostToolUse": "tool", "Stop": "stop"},
    "kiro": {"SessionStart": "session_start", "UserPromptSubmit": "prompt", "PostToolUse": "tool", "Stop": "stop"},
}


def _first(values) -> str | None:
    return values[0] if isinstance(values, list) and values else None


def _copilot(payload: dict) -> dict:
    arguments = payload.get("toolArgs")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {"command": arguments}
    result = payload.get("toolResult") or {}
    failed = isinstance(result, dict) and result.get("resultType") == "failure"
    return {
        "session_id": payload.get("sessionId", "default"), "cwd": payload.get("cwd"), "source": payload.get("source"),
        "prompt": payload.get("prompt"), "tool_name": payload.get("toolName"), "tool_input": arguments,
        "tool_response": {"exit_code": 1 if failed else 0},
    }


def _cursor(payload: dict) -> dict:
    return {
        "session_id": payload.get("conversation_id") or payload.get("session_id") or "default",
        "cwd": payload.get("cwd") or _first(payload.get("workspace_roots")), "source": payload.get("source"),
        "prompt": payload.get("prompt"), "tool_name": payload.get("tool_name"),
        "tool_input": payload.get("tool_input"), "tool_response": {},
    }


def _gemini(payload: dict) -> dict:
    response = payload.get("tool_response") or {}
    error = response.get("error") if isinstance(response, dict) else None
    return payload | {
        "tool_response": {"stderr": str(error or "")}, "last_assistant_message": payload.get("prompt_response"),
    }


def _antigravity(payload: dict) -> dict:
    call = payload.get("toolCall") or {}
    return {
        "session_id": payload.get("conversationId", "default"), "cwd": _first(payload.get("workspacePaths")),
        "tool_name": call.get("name"), "tool_input": call.get("args"),
        "tool_response": {"stderr": str(payload.get("error") or "")},
    }


def _kiro(payload: dict) -> dict:
    return payload | {"prompt": payload.get("prompt") or os.environ.get("USER_PROMPT", "")}


READERS: dict[str, Callable[[dict], dict]] = {
    "copilot": _copilot, "cursor": _cursor, "gemini": _gemini, "antigravity": _antigravity, "kiro": _kiro,
}


def _pending_path(session: str):
    return home() / "pending" / hashlib.sha1(session.encode()).hexdigest()


def _hold(session: str, text: str) -> None:
    path = _pending_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _release(session: str) -> str:
    path = _pending_path(session)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    path.unlink()
    return text


def _render(host: str, name: str, event: str, context: str, raw: dict, session: str) -> str:
    if host == "kiro":
        return context
    if host == "copilot":
        if event == "prompt":
            prompt = raw.get("transformedPrompt") or raw.get("prompt") or ""
            return json.dumps({"modifiedTransformedPrompt": f"{prompt}\n\n{context}"} if context else {})
        return json.dumps({"additionalContext": context} if context else {})
    if host == "cursor":
        # Cursor takes no context from beforeSubmitPrompt, so it is held for the first tool result of the turn.
        if event == "prompt":
            if context:
                _hold(session, context)
            return json.dumps({"continue": True})
        if event == "tool":
            context = _release(session)
        return json.dumps({"additional_context": context} if context else {})
    if host == "gemini":
        output = {"hookSpecificOutput": {"hookEventName": name, "additionalContext": context}}
        return json.dumps(output if context else {})
    return json.dumps({"injectSteps": [{"systemMessage": context}]} if context else {})


def run(host: str, name: str, raw: dict) -> str:
    """Handles one hook call and returns what the host should read on stdout."""
    payload = READERS[host](raw)
    event = EVENTS[host].get(name, "")
    return _render(host, name, event, handle(event, payload), raw, str(payload.get("session_id", "default")))


def main(argv: list[str]) -> int:
    """``hook-host <host> <event>``: payload on stdin, host output on stdout. Never fails the agent."""
    try:
        host, name = argv
        output = run(host, name, json.load(sys.stdin))
        if output:
            sys.stdout.write(output)
    except Exception as exc:
        print(f"kanerva hook: {exc}", file=sys.stderr)
    return 0
