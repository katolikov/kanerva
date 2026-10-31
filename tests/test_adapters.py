"""Manifests, config templates and the Python adapter of the agent plugins."""

import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from kanerva import __version__, hosts

ROOT = Path(__file__).resolve().parent.parent
PLUGINS = ROOT / "plugins"
sys.path.insert(0, str(ROOT / "scripts"))

import add_to_config  # noqa: E402

MANIFESTS = [
    "claude-code/.claude-plugin/plugin.json", "claude-code/.codex-plugin/plugin.json", "copilot/plugin.json",
    "gemini/gemini-extension.json", "qwen/qwen-extension.json", "openclaw/package.json", "pi/package.json",
    "cline/package.json", "amp/package.json", "opencode/package.json",
]
TEMPLATES = [
    "cursor/hooks.json", "cursor/mcp.json", "kiro/hooks.json", "kiro/mcp.json", "antigravity/hooks.json",
    "antigravity/mcp_config.json", "devin/hooks.json",
]


@pytest.mark.parametrize("path", MANIFESTS)
def test_manifests_carry_the_package_version(path):
    assert json.loads((PLUGINS / path).read_text())["version"] == __version__


@pytest.mark.parametrize("name", ["claude-code", "copilot", "gemini", "qwen", "hermes"])
def test_launchers_are_executable(name):
    assert (PLUGINS / name / "scripts" / "run.sh").stat().st_mode & stat.S_IXUSR


@pytest.mark.parametrize(("path", "host"), [("copilot/hooks/hooks.json", "copilot"), ("gemini/hooks/hooks.json", "gemini"), ("cursor/hooks.json", "cursor")])
def test_hook_files_name_events_the_host_table_knows(path, host):
    hook_file = json.loads((PLUGINS / path).read_text())
    assert set(hook_file["hooks"]) == set(hosts.EVENTS[host])
    for event, entries in hook_file["hooks"].items():
        assert f"hook-host {host} {event}" in json.dumps(entries)


def test_kiro_triggers_are_known():
    hook_file = json.loads((PLUGINS / "kiro" / "hooks.json").read_text())
    assert {hook["trigger"] for hook in hook_file["hooks"]} == set(hosts.EVENTS["kiro"])


def test_antigravity_events_are_known():
    assert set(json.loads((PLUGINS / "antigravity" / "hooks.json").read_text())["kanerva"]) == set(hosts.EVENTS["antigravity"])


@pytest.mark.parametrize("template", TEMPLATES)
def test_templates_fill_in_the_launcher(template, tmp_path):
    target = tmp_path / "config.json"
    add_to_config.install(PLUGINS / template, target, "/opt/kanerva/scripts/run.sh")
    text = target.read_text()
    assert "{{RUN}}" not in text and "/opt/kanerva/scripts/run.sh" in text


def test_merging_keeps_other_settings_and_replaces_old_entries(tmp_path):
    target = tmp_path / "hooks.json"
    other = {"command": "my-linter"}
    old = {"command": "/old/scripts/run.sh hook-host cursor stop"}
    target.write_text(json.dumps({"version": 1, "theme": "dark", "hooks": {"stop": [other, old]}}))
    add_to_config.install(PLUGINS / "cursor" / "hooks.json", target, "/new/scripts/run.sh")
    result = json.loads(target.read_text())
    assert result["theme"] == "dark"
    assert result["hooks"]["stop"][0] == other
    assert [entry["command"] for entry in result["hooks"]["stop"][1:]] == ["/new/scripts/run.sh hook-host cursor stop"]


def test_installing_twice_changes_nothing(tmp_path):
    target = tmp_path / "hooks.json"
    add_to_config.install(PLUGINS / "cursor" / "hooks.json", target, "/k/scripts/run.sh")
    first = target.read_text()
    add_to_config.install(PLUGINS / "cursor" / "hooks.json", target, "/k/scripts/run.sh")
    assert target.read_text() == first


def test_a_broken_config_is_left_alone(tmp_path):
    target = tmp_path / "hooks.json"
    target.write_text("{ // comment")
    runner = [sys.executable, str(ROOT / "scripts" / "add_to_config.py"), str(PLUGINS / "cursor" / "hooks.json"), str(target), "/k/scripts/run.sh"]
    done = subprocess.run(runner, capture_output=True, text=True)
    assert done.returncode == 1 and target.read_text() == "{ // comment"


@pytest.fixture
def hermes(monkeypatch):
    spec = importlib.util.spec_from_file_location("kanerva_hermes", PLUGINS / "hermes" / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def call(event, session_id, **fields):
        calls.append((event, session_id, fields))
        return {"session_start": "project note", "prompt": "prompt note"}.get(event, "")

    monkeypatch.setattr(module, "call", call)
    module.calls = calls
    return module


def test_hermes_first_turn_gets_session_and_prompt_notes(hermes):
    assert hermes.pre_llm_call(session_id="s", user_message="hello there", is_first_turn=True) == {"context": "project note\nprompt note"}
    assert hermes.pre_llm_call(session_id="s", user_message="and again", is_first_turn=False) == {"context": "prompt note"}


def test_hermes_records_tools_in_the_background(hermes):
    hermes.post_tool_call(tool_name="terminal", args={"command": "pytest"}, session_id="s", status="error", error_message="boom")
    hermes.on_session_end(session_id="s")
    hermes.writes.shutdown(wait=True)
    assert [c[0] for c in hermes.calls] == ["tool", "stop"]
    assert hermes.calls[0][2]["tool_response"] == {"exit_code": 1, "stderr": "boom"}


def test_hermes_registers_its_hooks(hermes):
    names = []

    class Context:
        def register_hook(self, name, _callback):
            names.append(name)

    hermes.register(Context())
    assert sorted(names) == ["on_session_end", "post_tool_call", "pre_llm_call"]
