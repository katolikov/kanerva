"""Hermes plugin. Notes are added to each turn; edits, commands and turn ends are recorded in the background."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUNNER = Path(__file__).resolve().parent / "scripts" / "run.sh"
TIMEOUT_SECONDS = float(os.environ.get("KANERVA_TIMEOUT_MS", "20000")) / 1000

logger = logging.getLogger(__name__)
writes = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kanerva")


def workspace() -> str:
    return os.environ.get("TERMINAL_CWD") or os.getcwd()


def call(event: str, session_id: str, **fields) -> str:
    """Runs one memory event. A failure here must never reach the agent, so it returns an empty string."""
    payload = {"session_id": session_id or "default", "cwd": workspace(), **fields}
    try:
        done = subprocess.run(
            ["sh", str(RUNNER), "hook-raw", event], input=json.dumps(payload, default=str), capture_output=True,
            text=True, timeout=TIMEOUT_SECONDS, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("kanerva %s failed: %s", event, exc)
        return ""
    return done.stdout if done.returncode == 0 else ""


def pre_llm_call(session_id="", user_message="", is_first_turn=False, **_):
    parts = [call("session_start", session_id, source="startup")] if is_first_turn else []
    parts.append(call("prompt", session_id, prompt=user_message or ""))
    context = "\n".join(part for part in parts if part)
    return {"context": context} if context else None


def post_tool_call(tool_name="", args=None, session_id="", status="", error_message="", **_):
    response = {"exit_code": 1 if status == "error" else 0, "stderr": error_message or ""}
    writes.submit(call, "tool", session_id, tool_name=tool_name, tool_input=args or {}, tool_response=response)


def on_session_end(session_id="", **_):
    writes.submit(call, "stop", session_id)


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_hook("post_tool_call", post_tool_call)
    ctx.register_hook("on_session_end", on_session_end)
