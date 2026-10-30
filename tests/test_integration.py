"""Simulated host payloads for the Claude Code hooks and the MCP server."""

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from kanerva import hooks
from kanerva.mcp_server import Server, serve
from kanerva.memory import Memory

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("KANERVA_HOME", str(tmp_path / "kanerva"))
    return tmp_path


def payload(event, **extra):
    return {"session_id": "s1", "cwd": "/tmp/project-x", "hook_event_name": event} | extra


def seed(store, text, **kwargs):
    from kanerva.paths import db_path, project_scope

    mem = Memory(db_path())
    item = mem.write(text, scope=project_scope("/tmp/project-x"), kind="semantic", **kwargs)
    mem.close()
    return item


def test_prompt_hook_injects_relevant_memory(store):
    seed(store, "Run the suite with make test-all from the repo root")
    context = hooks.handle("prompt", payload("UserPromptSubmit", prompt="how do I run the test suite?"))
    assert "make test-all" in context


def test_short_or_slash_prompts_are_ignored(store):
    seed(store, "Anything at all")
    assert hooks.handle("prompt", payload("UserPromptSubmit", prompt="ok")) == ""
    assert hooks.handle("prompt", payload("UserPromptSubmit", prompt="/compact everything now")) == ""


def test_injected_context_fits_the_host_limit(store):
    for i in range(300):
        seed(store, f"Note {i} about the payment service retry logic and its idempotency keys " * 8)
    context = hooks.handle("prompt", payload("UserPromptSubmit", prompt="payment service retry logic"))
    assert len(context) <= hooks.MAX_CONTEXT_CHARS


def test_session_start_uses_project_memory(store):
    seed(store, "Project convention: all architecture decisions live in docs/adr")
    assert "docs/adr" in hooks.handle("session_start", payload("SessionStart", source="startup"))


def test_tool_hook_records_edits_and_test_commands(store):
    hooks.handle("tool", payload("PostToolUse", tool_name="Edit", tool_input={"file_path": "/repo/src/app.py"}, tool_response={}))
    hooks.handle("tool", payload("PostToolUse", tool_name="Bash", tool_input={"command": "pytest -x tests"}, tool_response={"stdout": "ok", "stderr": ""}))
    from kanerva.paths import db_path

    mem = Memory(db_path())
    texts = [r["text"] for r in mem.db.execute("SELECT text FROM items")]
    mem.close()
    assert "Edited /repo/src/app.py" in texts
    assert "Test command: pytest -x tests" in texts


def test_stop_turns_execution_into_feedback(store):
    used = seed(store, "The retry helper lives in src/billing/retry_policy.py")
    hooks.handle("prompt", payload("UserPromptSubmit", prompt="fix the billing retry helper"))
    hooks.handle("tool", payload("PostToolUse", tool_name="Edit", tool_input={"file_path": "src/billing/retry_policy.py"}, tool_response={}))
    hooks.handle("stop", payload("Stop", last_assistant_message="Done."))
    from kanerva.paths import db_path

    mem = Memory(db_path())
    row = mem.db.execute("SELECT used, unused FROM items WHERE id=?", (used,)).fetchone()
    assert row["used"] == 2.0
    assert mem.db.execute("SELECT count(*) AS n FROM session_events").fetchone()["n"] == 0
    mem.close()


def test_failed_tests_halve_the_reward(store):
    used = seed(store, "The retry helper lives in src/billing/retry_policy.py")
    hooks.handle("prompt", payload("UserPromptSubmit", prompt="fix the billing retry helper"))
    hooks.handle("tool", payload("PostToolUse", tool_name="Edit", tool_input={"file_path": "src/billing/retry_policy.py"}, tool_response={}))
    hooks.handle("tool", payload("PostToolUse", tool_name="Bash", tool_input={"command": "pytest"}, tool_response={"stderr": "2 failed"}))
    hooks.handle("stop", payload("Stop", last_assistant_message=""))
    from kanerva.paths import db_path

    mem = Memory(db_path())
    assert mem.db.execute("SELECT used FROM items WHERE id=?", (used,)).fetchone()["used"] == 1.5
    mem.close()


def test_compaction_hooks(store):
    assert hooks.handle("compact", payload("PreCompact", trigger="auto")) == ""
    hooks.handle("compact_summary", payload("PostCompact", compact_summary="We migrated auth to OAuth2."))
    assert "OAuth2" in hooks.handle("prompt", payload("UserPromptSubmit", prompt="what did we do about auth and OAuth2?"))


def run_hook(store, body):
    env = {"KANERVA_HOME": str(store / "kanerva"), "PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
    return subprocess.run([sys.executable, "-m", "kanerva", "hook"], input=body, capture_output=True, text=True, env=env, cwd=store)


def test_hook_process_contract(store):
    seed(store, "Run the suite with make test-all from the repo root")
    out = run_hook(store, json.dumps(payload("UserPromptSubmit", prompt="how do I run the test suite?")))
    assert out.returncode == 0
    parsed = json.loads(out.stdout)
    assert parsed["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "make test-all" in parsed["hookSpecificOutput"]["additionalContext"]


def test_hook_process_survives_bad_input(store):
    out = run_hook(store, "not json")
    assert out.returncode == 0 and out.stdout == ""


def test_mcp_round_trip(tmp_path):
    server = Server(Memory(tmp_path / "m.sqlite"), scope="project:x")
    assert server.handle({"id": 1, "method": "initialize", "params": {}})["result"]["serverInfo"]["name"] == "kanerva"
    tools = server.handle({"id": 2, "method": "tools/list"})["result"]["tools"]
    assert {"memory_search", "memory_remember", "memory_forget", "memory_expand"} <= {t["name"] for t in tools}

    def call(name, **args):
        reply = server.handle({"id": 3, "method": "tools/call", "params": {"name": name, "arguments": args}})
        return json.loads(reply["result"]["content"][0]["text"])

    item = call("memory_remember", text="Deploys go through the staging branch", key="process:deploy")["id"]
    found = call("memory_search", query="how do deploys work")
    assert found["memories"][0]["id"] == item
    assert call("memory_feedback", retrieval_id=found["retrieval_id"], used_ids=[item])["updated"] == 1
    assert call("memory_edit", id=item, text="Deploys go through the release branch")["edited"]
    assert call("memory_expand", id=item)[0]["text"].endswith("release branch")
    assert call("memory_stats")["active"] == 1
    assert call("memory_forget", id=item)["removed"] == 1
    assert call("memory_stats")["active"] == 0


def test_mcp_error_is_reported_not_raised(tmp_path):
    server = Server(Memory(tmp_path / "m.sqlite"), scope="project:x")
    reply = server.handle({"id": 1, "method": "tools/call", "params": {"name": "nope", "arguments": {}}})
    assert reply["result"]["isError"] is True
    assert server.handle({"id": 2, "method": "bogus"})["error"]["code"] == -32601


def test_mcp_stdio_loop(store):
    lines = []
    reader = io.StringIO(json.dumps({"id": 1, "method": "ping"}) + "\nnot-json\n" + json.dumps({"method": "notifications/initialized"}) + "\n")
    serve(reader, lines.append)
    assert json.loads(lines[0])["result"] == {}
    assert json.loads(lines[1])["error"]["code"] == -32700
    assert len(lines) == 2


def test_plugin_manifests_are_valid_json_and_reference_real_files():
    plugin = ROOT / "plugins" / "claude-code"
    manifest = json.loads((plugin / ".claude-plugin" / "plugin.json").read_text())
    assert manifest["name"] == "kanerva"
    hook_config = json.loads((plugin / "hooks" / "hooks.json").read_text())
    assert set(hook_config["hooks"]) == {"SessionStart", "UserPromptSubmit", "PostToolUse", "Stop", "PreCompact", "PostCompact"}
    assert (plugin / "scripts" / "run.sh").stat().st_mode & 0o111
    assert "mcpServers" in json.loads((plugin / ".mcp.json").read_text())
    for skill in ("memory", "remember", "forget"):
        assert (plugin / "skills" / skill / "SKILL.md").read_text().startswith("---\n")
