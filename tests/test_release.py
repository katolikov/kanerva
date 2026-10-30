import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from kanerva import Memory
from kanerva import hooks as hook_module
from kanerva.mcp_server import Server
from kanerva.secrets_filter import redact
from kanerva.text import approx_tokens, content_terms, tokenize

ROOT = Path(__file__).resolve().parent.parent
TOKEN = "ghp_" + "a1B2c3D4e5" * 3 + "a1B2c3"


def test_mcp_search_leaves_no_open_transaction(tmp_path):
    path = tmp_path / "memory.sqlite"
    server = Server(Memory(path), scope="project:x")
    server.call("memory_remember", {"text": "the staging port is 8443"})
    assert server.call("memory_search", {"query": "staging port"})["memories"]
    other = sqlite3.connect(path, timeout=0.2)
    other.execute("INSERT INTO meta VALUES ('probe', '1')")
    other.commit()
    other.close()


@pytest.mark.parametrize(
    "line",
    [
        "DB_PASSWORD=hunter2hunter2",
        "export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCYzEXAMPLEKEY",
        "GITHUB_TOKEN=abcd1234efgh5678",
        "Authorization: Bearer abcdefghijklmnop1234567890",
        'curl -H "Authorization: Bearer abcdefghijklmnop1234567890" https://api.example.com',
    ],
)
def test_environment_style_secrets_are_redacted(line):
    text, kinds = redact(line)
    assert kinds and "REDACTED" in text
    assert "hunter2" not in text and "abcdefghijklmnop" not in text and "wJalr" not in text and "abcd1234" not in text


@pytest.mark.parametrize("safe", ["max_tokens = 1024", "token_count: 12345678", "password reset emails go out at noon"])
def test_ordinary_settings_are_kept(safe):
    assert redact(safe)[1] == []


def test_secrets_do_not_reach_any_table(tmp_path, monkeypatch):
    monkeypatch.setenv("KANERVA_HOME", str(tmp_path))
    payload = {"session_id": "s", "cwd": str(tmp_path)}
    hook_module.handle("prompt", {**payload, "prompt": f"please rotate {TOKEN} before the release"})
    hook_module.handle("tool", {**payload, "tool_name": "Bash", "tool_input": {"command": f"curl -H 'x: {TOKEN}' example.com"}})
    hook_module.handle("tool", {**payload, "tool_name": "Edit", "tool_input": {"file_path": "a.py", "new_string": TOKEN}})
    mem = Memory(tmp_path / "memory.sqlite")
    mem.write("the staging port is 8443", at=1.0)
    mem.retrieve(f"which port, and is {TOKEN} still valid", 100, now=2.0)
    mem.close()
    db = sqlite3.connect(tmp_path / "memory.sqlite")
    tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for table in tables:
        for row in db.execute(f'SELECT * FROM "{table}"'):
            assert TOKEN not in " ".join(str(v) for v in row)
    db.close()


@pytest.mark.parametrize(
    "note,query",
    [
        ("Сервер стейджинга слушает порт 8443", "какой порт слушает сервер стейджинга"),
        ("测试服务器监听 8443 端口", "服务器监听哪个端口"),
        ("ステージングサーバーはポート 8443 を使います", "ステージングサーバーのポート"),
    ],
)
def test_notes_in_other_scripts_are_found(note, query):
    mem = Memory()
    mem.write(note)
    mem.write("lunch was good")
    assert any("8443" in item.text for item in mem.retrieve(query, 200, record=False).items)


def test_tokens_keep_their_letters():
    assert "сервер" in tokenize("Сервер")
    assert "naïve" in content_terms("a naïve plan")


@pytest.mark.parametrize(
    "text,real",
    [
        ("Сервер стейджинга слушает порт 8443 за nginx, а тесты запускаются командой tox перед релизом.", 45),
        ("测试服务器在 nginx 后面监听 8443 端口，每次发布之前都要运行集成测试。", 28),
        (" ".join(["3f2a9c1e7b4d8065a1c3e5f7092b4d6e8f1a3c5e"] * 10), 200),
    ],
)
def test_token_estimate_does_not_undercount_badly(text, real):
    assert approx_tokens(text) >= 0.9 * real


def run_script(cwd, *args, stdin=""):
    env = {**os.environ, "KANERVA_HOME": str(cwd / "home")}
    env.pop("KANERVA_PROJECT_DIR", None)
    script = ROOT / "plugins" / "claude-code" / "scripts" / "run.sh"
    return subprocess.run(["sh", str(script), *args], cwd=cwd, env=env, input=stdin, capture_output=True, text=True, timeout=120)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
def test_hook_ignores_modules_and_python_pins_in_the_project(tmp_path):
    (tmp_path / "json.py").write_text("raise SystemExit('project code ran')\n")
    (tmp_path / "numpy.py").write_text("raise SystemExit('project code ran')\n")
    (tmp_path / ".python-version").write_text("2.7\n")
    event = {"hook_event_name": "UserPromptSubmit", "prompt": "how do I run the tests here", "session_id": "s", "cwd": str(tmp_path)}
    result = run_script(tmp_path, "hook", stdin=json.dumps(event))
    assert result.returncode == 0
    assert "project code ran" not in result.stdout + result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
def test_cli_keeps_the_project_scope_and_export_path(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert run_script(tmp_path, "add", "we deploy on fridays").returncode == 0
    assert "fridays" in run_script(tmp_path, "search", "when do we deploy").stdout
    assert run_script(tmp_path, "export", "memory.md").returncode == 0
    assert "fridays" in (tmp_path / "memory.md").read_text()
