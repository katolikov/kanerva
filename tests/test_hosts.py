"""Hook formats of the agents that do not use the Claude Code format."""

import json
import subprocess
import sys

import pytest

from kanerva import hooks, hosts
from kanerva.memory import Memory
from kanerva.paths import db_path, project_scope

CWD = "/tmp/project-x"
NOTE = "The test suite runs with make test-all"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("KANERVA_HOME", str(tmp_path / "kanerva"))
    memory = Memory(db_path())
    memory.write(NOTE, scope=project_scope(CWD), kind="semantic")
    memory.close()
    return tmp_path


def texts():
    memory = Memory(db_path())
    found = [item.text for item in memory.list_items(project_scope(CWD), limit=50)]
    memory.close()
    return found


def test_copilot_appends_notes_to_the_transformed_prompt(store):
    raw = {"sessionId": "s", "cwd": CWD, "prompt": "how do I run the test suite?", "transformedPrompt": "<date/> how do I run the test suite?"}
    output = json.loads(hosts.run("copilot", "userPromptTransformed", raw))
    assert output["modifiedTransformedPrompt"].startswith("<date/> how do I run")
    assert "make test-all" in output["modifiedTransformedPrompt"]


def test_copilot_prompt_without_notes_stays_untouched(store):
    raw = {"sessionId": "s", "cwd": "/tmp/other-project", "prompt": "how do I run the test suite?", "transformedPrompt": "x"}
    assert json.loads(hosts.run("copilot", "userPromptTransformed", raw)) == {}


def test_copilot_tool_arguments_arrive_as_a_string(store):
    raw = {
        "sessionId": "s", "cwd": CWD, "toolName": "edit", "toolArgs": json.dumps({"path": "src/app.py"}),
        "toolResult": {"resultType": "success", "textResultForLlm": "ok"},
    }
    assert json.loads(hosts.run("copilot", "postToolUse", raw)) == {}
    assert "Edited src/app.py" in texts()


def test_cursor_holds_prompt_notes_for_the_first_tool_result(store):
    prompt = {"conversation_id": "c", "workspace_roots": [CWD], "prompt": "how do I run the test suite?"}
    assert json.loads(hosts.run("cursor", "beforeSubmitPrompt", prompt)) == {"continue": True}
    tool = {"conversation_id": "c", "workspace_roots": [CWD], "tool_name": "Shell", "tool_input": {"command": "ls"}}
    assert "make test-all" in json.loads(hosts.run("cursor", "postToolUse", tool))["additional_context"]
    assert json.loads(hosts.run("cursor", "postToolUse", tool)) == {}


def test_cursor_session_start_uses_the_first_workspace_root(store):
    memory = Memory(db_path())
    memory.write("Project conventions: architecture decisions live in docs/adr", scope=project_scope(CWD), kind="semantic")
    memory.close()
    output = hosts.run("cursor", "sessionStart", {"session_id": "s", "workspace_roots": [CWD]})
    assert "docs/adr" in json.loads(output)["additional_context"]


def test_gemini_names_the_event_it_answers(store):
    raw = {"session_id": "s", "cwd": CWD, "prompt": "how do I run the test suite?"}
    output = json.loads(hosts.run("gemini", "BeforeAgent", raw))["hookSpecificOutput"]
    assert output["hookEventName"] == "BeforeAgent" and "make test-all" in output["additionalContext"]


def test_gemini_failed_test_command_is_recorded(store):
    raw = {
        "session_id": "s", "cwd": CWD, "tool_name": "run_shell_command", "tool_input": {"command": "pytest -x"},
        "tool_response": {"llmContent": "", "error": "exit code 1: error"},
    }
    assert hosts.run("gemini", "AfterTool", raw) == "{}"
    assert any(text == "Test command: pytest -x" for text in texts())


def test_antigravity_injects_a_system_message(store):
    memory = Memory(db_path())
    memory.write("Project conventions: architecture decisions live in docs/adr", scope=project_scope(CWD), kind="semantic")
    memory.close()
    output = json.loads(hosts.run("antigravity", "SessionStart", {"conversationId": "c", "workspacePaths": [CWD]}))
    assert "docs/adr" in output["injectSteps"][0]["systemMessage"]


def test_antigravity_reads_the_tool_call(store):
    raw = {
        "conversationId": "c", "workspacePaths": [CWD],
        "toolCall": {"name": "run_command", "args": {"CommandLine": "pytest"}},
    }
    assert hosts.run("antigravity", "PostToolUse", raw) == "{}"
    assert "Test command: pytest" in texts()


def test_kiro_prints_plain_text(store):
    raw = {"session_id": "s", "cwd": CWD, "prompt": "how do I run the test suite?"}
    assert "make test-all" in hosts.run("kiro", "UserPromptSubmit", raw)


def test_kiro_falls_back_to_the_prompt_variable(store, monkeypatch):
    monkeypatch.setenv("USER_PROMPT", "how do I run the test suite?")
    assert "make test-all" in hosts.run("kiro", "UserPromptSubmit", {"session_id": "s", "cwd": CWD})


@pytest.mark.parametrize("host", sorted(hosts.EVENTS))
def test_unknown_events_write_nothing(store, host):
    before = texts()
    hosts.run(host, "SomethingNew", {"session_id": "s", "cwd": CWD})
    assert texts() == before


def test_codex_patches_record_each_file(store):
    patch = "*** Begin Patch\n*** Update File: src/a.py\n@@\n*** Add File: src/b.py\n+x\n*** End Patch"
    hooks.handle("tool", {"session_id": "s", "cwd": CWD, "tool_name": "apply_patch", "tool_input": {"command": patch}})
    assert {"Edited src/a.py", "Edited src/b.py"} <= set(texts())


def test_process_contract_stays_silent_on_bad_input(store):
    env = {"KANERVA_HOME": str(store / "kanerva"), "PYTHONPATH": "src", "PATH": "/usr/bin:/bin"}
    for arguments in (["copilot", "sessionStart"], ["nobody", "x"], ["copilot"]):
        done = subprocess.run(
            [sys.executable, "-m", "kanerva", "hook-host", *arguments], input="not json", capture_output=True, text=True, env=env,
        )
        assert done.returncode == 0 and done.stdout == ""
