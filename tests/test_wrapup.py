"""The end-of-session note: when the host agent is asked for it, and what happens to the note."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kanerva import hooks, hosts
from kanerva.mcp_server import Server
from kanerva.memory import Memory
from kanerva.paths import db_path, project_scope

ROOT = Path(__file__).resolve().parent.parent
CWD = "/tmp/project-x"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("KANERVA_HOME", str(tmp_path / "kanerva"))
    monkeypatch.setattr(hooks, "WRAPUP", True)
    return tmp_path


def event(name, session="s1", **extra):
    return hooks.handle(name, {"session_id": session, "cwd": CWD} | extra)


def edit(session="s1", times=1):
    for i in range(times):
        event("tool", session, tool_name="Edit", tool_input={"file_path": f"src/module_{i}.py"}, tool_response={})


def shell(command, session="s1", times=1):
    for _ in range(times):
        event("tool", session, tool_name="Bash", tool_input={"command": command}, tool_response={"stderr": ""})


def test_a_trivial_session_is_left_alone(store):
    event("prompt", prompt="what does this function do?")
    assert event("stop") == ""
    edit(times=2)
    shell("ls", times=3)
    assert event("stop") == ""


def test_several_edits_ask_for_one_note(store):
    edit(times=3)
    assert event("stop") == hooks.WRAPUP_PROMPT
    assert event("stop") == ""


def test_an_edit_with_a_test_run_counts_as_real_work(store):
    edit()
    shell("pytest -q")
    assert event("stop") == hooks.WRAPUP_PROMPT


def test_a_long_run_of_commands_counts_as_real_work(store):
    shell("kubectl get pods", times=hooks.WRAPUP_COMMANDS)
    assert event("stop") == hooks.WRAPUP_PROMPT


def test_work_adds_up_across_turns(store):
    for _ in range(2):
        edit()
        assert event("stop") == ""
    edit()
    assert event("stop") == hooks.WRAPUP_PROMPT


def test_a_long_session_is_asked_again_only_after_much_more_work(store):
    edit(times=3)
    assert event("stop") == hooks.WRAPUP_PROMPT
    edit(times=hooks.WRAPUP_AGAIN_EDITS - 1)
    assert event("stop") == ""
    edit()
    assert event("stop") == hooks.WRAPUP_PROMPT


def test_sessions_are_counted_apart(store):
    edit("a", times=2)
    edit("b", times=2)
    assert event("stop", "a") == "" and event("stop", "b") == ""


def test_the_switch_turns_it_off(store, monkeypatch):
    monkeypatch.setattr(hooks, "WRAPUP", False)
    edit(times=5)
    assert event("stop") == ""


def test_a_turn_already_kept_going_by_a_stop_hook_is_not_asked(store):
    edit(times=3)
    assert event("stop", stop_hook_active=True) == ""


def test_a_session_waiting_on_background_work_is_not_asked_yet(store):
    edit(times=3)
    assert event("stop", background_tasks=[{"id": "t1", "type": "shell"}]) == ""
    assert event("stop") == hooks.WRAPUP_PROMPT


def test_a_note_the_agent_wrote_itself_restarts_the_count(store):
    edit(times=2)
    event("tool", tool_name="mcp__plugin_kanerva_memory__memory_remember", tool_input={"text": "x"}, tool_response={})
    edit()
    assert event("stop") == ""


def test_opencode_tool_names_count_too(store):
    event("tool", tool_name="edit", tool_input={"filePath": "a.py"}, tool_response={})
    event("tool", tool_name="kanerva_memory_remember", tool_input={"text": "x"}, tool_response={})
    event("tool", tool_name="edit", tool_input={"filePath": "b.py"}, tool_response={})
    event("tool", tool_name="bash", tool_input={"command": "npm test"}, tool_response={})
    assert event("stop") == hooks.WRAPUP_PROMPT


def test_a_patch_counts_each_file(store):
    patch = "*** Begin Patch\n*** Update File: a.py\n*** Update File: b.py\n*** Add File: c.py\n*** End Patch"
    event("tool", tool_name="apply_patch", tool_input={"command": patch}, tool_response={})
    assert event("stop") == hooks.WRAPUP_PROMPT


def test_old_sessions_are_dropped(store):
    edit("old")
    memory = Memory(db_path())
    memory.db.executescript("UPDATE session_work SET at = 0")
    memory.close()
    edit("new")
    event("stop", "new")
    memory = Memory(db_path())
    left = [r["session"] for r in memory.db.execute("SELECT session FROM session_work")]
    memory.close()
    assert left == ["new"]


def test_the_note_meets_the_secret_filter_and_the_normal_ranking(store):
    secret = "ghp_" + "a1B2c3D4e5" * 4
    server = Server(Memory(db_path()), scope=project_scope(CWD))
    text = f"Flaky upload test: tried retries, the fix was a longer socket timeout. Token was {secret}"
    server.call("memory_remember", {"text": text})
    server.mem.close()

    memory = Memory(db_path())
    row = memory.db.execute("SELECT * FROM items").fetchone()
    assert secret not in row["text"] and "socket timeout" in row["text"]
    assert row["kind"] == "semantic" and row["scope"] == project_scope(CWD) and row["pinned"] == 0
    memory.close()

    context = event("prompt", prompt="why was the upload test flaky?")
    assert "socket timeout" in context and secret not in context


def run_hook(store, name, env=None, **extra):
    payload = {"session_id": "s1", "cwd": CWD, "hook_event_name": name} | extra
    base = {"KANERVA_HOME": str(store / "kanerva"), "PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
    return subprocess.run(
        [sys.executable, "-m", "kanerva", "hook"], input=json.dumps(payload), capture_output=True, text=True,
        env=base | (env or {}), cwd=store,
    )


def work_through_hooks(store, env=None):
    for i in range(3):
        run_hook(store, "PostToolUse", env, tool_name="Edit", tool_input={"file_path": f"a{i}.py"}, tool_response={})


def test_claude_code_gets_feedback_rather_than_an_error(store):
    work_through_hooks(store)
    out = run_hook(store, "Stop", {"CLAUDECODE": "1"})
    assert json.loads(out.stdout) == {
        "hookSpecificOutput": {"hookEventName": "Stop", "additionalContext": hooks.WRAPUP_PROMPT}
    }
    assert run_hook(store, "Stop", {"CLAUDECODE": "1"}).stdout == ""


def test_hosts_on_the_same_hook_format_get_a_block_decision(store):
    work_through_hooks(store)
    assert json.loads(run_hook(store, "Stop").stdout) == {"decision": "block", "reason": hooks.WRAPUP_PROMPT}


def test_the_environment_switch_reaches_the_hook(store):
    work_through_hooks(store)
    assert run_hook(store, "Stop", {"KANERVA_WRAPUP": "0"}).stdout == ""


def host_work(host, name, payload, times=3):
    for _ in range(times):
        hosts.run(host, name, payload)


def test_copilot_is_given_a_block_decision(store):
    host_work("copilot", "postToolUse", {"sessionId": "c1", "cwd": CWD, "toolName": "edit", "toolArgs": {"path": "a.py"}})
    answer = hosts.run("copilot", "agentStop", {"sessionId": "c1", "cwd": CWD})
    assert json.loads(answer) == {"decision": "block", "reason": hooks.WRAPUP_PROMPT}
    assert json.loads(hosts.run("copilot", "agentStop", {"sessionId": "c1", "cwd": CWD})) == {}


def test_gemini_is_given_a_deny_with_a_calm_message(store):
    payload = {"session_id": "g1", "cwd": CWD, "tool_name": "write_file", "tool_input": {"file_path": "a.py"}}
    host_work("gemini", "AfterTool", payload)
    answer = json.loads(hosts.run("gemini", "AfterAgent", {"session_id": "g1", "cwd": CWD, "prompt_response": "Done."}))
    assert answer["decision"] == "deny" and answer["reason"] == hooks.WRAPUP_PROMPT and "Kanerva" in answer["systemMessage"]


def test_cursor_is_given_a_follow_up_message(store):
    payload = {"conversation_id": "c2", "workspace_roots": [CWD], "tool_name": "Write", "tool_input": {"file_path": "a.py"}}
    host_work("cursor", "postToolUse", payload)
    answer = hosts.run("cursor", "stop", {"conversation_id": "c2", "workspace_roots": [CWD], "status": "completed", "loop_count": 0})
    assert json.loads(answer) == {"followup_message": hooks.WRAPUP_PROMPT}


def test_cursor_is_not_asked_after_an_aborted_run(store):
    payload = {"conversation_id": "c3", "workspace_roots": [CWD], "tool_name": "Write", "tool_input": {"file_path": "a.py"}}
    host_work("cursor", "postToolUse", payload)
    assert json.loads(hosts.run("cursor", "stop", {"conversation_id": "c3", "status": "aborted"})) == {}


@pytest.mark.parametrize("host", ["kiro", "antigravity"])
def test_hosts_without_a_route_stay_quiet(store, host):
    tool = {"session_id": "k1", "cwd": CWD, "tool_name": "fs_write", "tool_input": {"path": "a.py"}}
    tool |= {"conversationId": "k1", "workspacePaths": [CWD], "toolCall": {"name": "write_to_file", "args": {"TargetFile": "a.py"}}}
    host_work(host, "PostToolUse", tool)
    answer = hosts.run(host, "Stop", {"session_id": "k1", "conversationId": "k1", "cwd": CWD, "workspacePaths": [CWD]})
    assert hooks.WRAPUP_PROMPT not in answer


def test_the_shipped_hook_can_answer_stop():
    hook_file = json.loads((ROOT / "plugins" / "claude-code" / "hooks" / "hooks.json").read_text())
    stop = hook_file["hooks"]["Stop"][0]["hooks"][0]
    assert "async" not in stop  # a background hook cannot keep the agent going
    assert "memory_remember" in hook_file["hooks"]["PostToolUse"][0]["matcher"]


def test_a_session_with_no_recorded_work_stays_quiet(store):
    assert event("stop", "never-seen") == ""
