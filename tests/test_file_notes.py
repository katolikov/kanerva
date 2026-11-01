"""Notes that surface when a session first touches a file."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kanerva import hooks, hosts
from kanerva.memory import Memory
from kanerva.paths import db_path, project_scope

ROOT = Path(__file__).resolve().parent.parent
CWD = "/tmp/project-x"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("KANERVA_HOME", str(tmp_path / "kanerva"))
    return tmp_path


def remember(text, kind="semantic", at=None):
    memory = Memory(db_path())
    memory.write(text, scope=project_scope(CWD), kind=kind, at=at)
    memory.commit()
    memory.close()


def touch(path, session="s1", tool="Read"):
    return hooks.handle("tool", {"session_id": session, "cwd": CWD, "tool_name": tool, "tool_input": {"file_path": path}, "tool_response": {}})


def test_a_note_about_the_file_shows_up_once(store):
    remember("The retry loop in src/net/client.py hides timeouts, so log before retrying.")
    assert "retry loop" in touch("src/net/client.py")
    assert touch("src/net/client.py") == ""


def test_an_absolute_path_is_matched_inside_the_project(store):
    remember("The retry loop in src/net/client.py hides timeouts.")
    context = touch(f"{CWD}/src/net/client.py")
    assert "retry loop" in context and CWD not in context


def test_another_session_gets_it_again(store):
    remember("client.py hides timeouts.")
    assert touch("src/net/client.py", "a") and touch("src/net/client.py", "b")


def test_a_longer_name_does_not_match(store):
    remember("test_client.py is flaky on macOS.")
    assert touch("src/net/client.py") == ""


def test_a_common_name_needs_its_folder(store):
    remember("Exports go through index.ts at the package root.")
    remember("web/index.ts registers the routes.")
    context = touch("web/index.ts")
    assert "registers the routes" in context and "Exports go" not in context


def test_the_edit_log_is_not_a_note_about_the_file(store):
    touch("src/net/client.py", tool="Edit")
    assert touch("src/net/client.py", "other") == ""


def test_superseded_notes_stay_out(store):
    memory = Memory(db_path())
    scope = project_scope(CWD)
    memory.write("client.py retries 3 times", scope=scope, kind="semantic", key="retries", at=1.0)
    memory.write("client.py retries 5 times", scope=scope, kind="semantic", key="retries", at=2.0)
    memory.commit()
    memory.close()
    context = touch("src/net/client.py")
    assert "5 times" in context and "3 times" not in context


def test_a_session_has_a_ceiling(store, monkeypatch):
    monkeypatch.setattr(hooks, "FILE_SESSION_BUDGET", hooks.FILE_BUDGET)
    remember("alpha.py: " + "word " * 80)
    remember("beta.py: " + "word " * 80 + "more")
    assert touch("alpha.py") and touch("beta.py") == ""


def test_the_switch_turns_it_off(store, monkeypatch):
    monkeypatch.setattr(hooks, "FILE_BUDGET", 0)
    remember("client.py hides timeouts.")
    assert touch("src/net/client.py") == ""


def test_a_patch_counts_each_file(store):
    remember("a.py needs a lock around the cache.")
    patch = "*** Begin Patch\n*** Update File: a.py\n*** End Patch"
    context = hooks.handle("tool", {"session_id": "s1", "cwd": CWD, "tool_name": "apply_patch", "tool_input": {"command": patch}, "tool_response": {}})
    assert "needs a lock" in context


def test_the_hook_prints_context_for_a_tool_call(store):
    remember("client.py hides timeouts.")
    payload = {"session_id": "s1", "cwd": CWD, "hook_event_name": "PostToolUse", "tool_name": "Read",
               "tool_input": {"file_path": "client.py"}, "tool_response": {}}
    env = {"KANERVA_HOME": str(store / "kanerva"), "PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, "-m", "kanerva", "hook"], input=json.dumps(payload), capture_output=True, text=True, env=env, cwd=store)
    answer = json.loads(out.stdout)["hookSpecificOutput"]
    assert answer["hookEventName"] == "PostToolUse" and "hides timeouts" in answer["additionalContext"]


def test_cursor_gets_held_context_and_file_notes_together(store):
    remember("client.py hides timeouts.")
    hosts.run("cursor", "beforeSubmitPrompt", {"conversation_id": "c1", "workspace_roots": [CWD], "prompt": "why do timeouts vanish in the client?"})
    answer = json.loads(hosts.run("cursor", "postToolUse", {"conversation_id": "c1", "workspace_roots": [CWD], "tool_name": "Read", "tool_input": {"file_path": "client.py"}}))
    assert "hides timeouts" in answer["additional_context"]


def test_a_note_the_prompt_already_brought_is_not_repeated(store):
    remember("The retry loop in src/net/client.py hides timeouts, so log before retrying.")
    prompt = hooks.handle("prompt", {"session_id": "s1", "cwd": CWD, "prompt": "why does the retry loop in src/net/client.py hide timeouts"})
    assert "retry loop" in prompt
    assert touch("src/net/client.py") == ""


def test_a_file_outside_the_project_gets_no_notes(store):
    remember("Keep MEMORY.md short.")
    assert touch("/Users/me/.claude/projects/x/memory/MEMORY.md") == ""


def test_an_edit_outside_the_project_is_not_work(store):
    touch("/Users/me/.claude/projects/x/memory/note.md", tool="Write")
    memory = Memory(db_path())
    assert memory.db.execute("SELECT count(*) FROM items").fetchone()[0] == 0
    assert memory.db.execute("SELECT count(*) FROM session_work").fetchone()[0] == 0
    memory.close()


def test_an_edit_inside_the_project_is_stored_relative(store):
    touch(f"{CWD}/src/net/client.py", tool="Edit")
    memory = Memory(db_path())
    assert [r[0] for r in memory.db.execute("SELECT text FROM items")] == ["Edited src/net/client.py"]
    memory.close()


def test_a_test_command_that_never_ran_is_not_kept(store):
    def shell(response):
        hooks.handle("tool", {"session_id": "s1", "cwd": CWD, "tool_name": "Bash", "tool_input": {"command": "python -m pytest"}, "tool_response": response})

    shell({"stdout": "(eval):1: command not found: python", "stderr": ""})
    memory = Memory(db_path())
    assert memory.db.execute("SELECT count(*) FROM items").fetchone()[0] == 0
    memory.close()
    shell({"stdout": "3 passed", "stderr": ""})
    memory = Memory(db_path())
    assert [r[0] for r in memory.db.execute("SELECT text FROM items")] == ["Test command: python -m pytest"]
    memory.close()


def test_a_session_does_not_open_with_edit_lines(store):
    remember("Edited src/net/client.py", kind="episodic")
    remember("Architecture decision: the client keeps one connection per host.")
    context = hooks.handle("session_start", {"session_id": "s1", "cwd": CWD})
    assert "Architecture decision" in context and "Edited" not in context


def test_cpp_build_systems_count_as_test_runs():
    for command in ("ctest --output-on-failure", "make check", "ninja test", "cmake --build build --target test", "meson test -C build", "bazel test //..."):
        assert hooks._TEST_COMMAND.search(command), command
    assert not hooks._TEST_COMMAND.search("make all")


def test_a_cpp_name_is_distinctive():
    assert "Client::connect" in hooks.distinctive_tokens("Client::connect hides timeouts")
    assert hooks.item_used("Client::connect hides timeouts", "git grep Client::connect")


def test_a_cpp_file_finds_its_notes(store):
    remember("Status::OK in src/db/impl.cc must be checked before Client::connect retries.")
    assert "Client::connect" in touch("src/db/impl.cc")
    assert touch("src/db/other.cc") == ""


def shell(command, stdout="ok"):
    hooks.handle("tool", {"session_id": "s1", "cwd": CWD, "tool_name": "Bash", "tool_input": {"command": command}, "tool_response": {"stdout": stdout, "stderr": ""}})


def stored():
    memory = Memory(db_path())
    texts = [r[0] for r in memory.db.execute("SELECT text FROM items WHERE valid_to IS NULL")]
    memory.close()
    return texts


def test_a_build_that_worked_is_kept_and_the_latest_wins(store):
    shell("cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug")
    assert stored() == ["Build command: cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug"]
    shell("cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G Ninja")
    assert stored() == ["Build command: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G Ninja"]


def test_a_build_that_failed_is_not_kept(store):
    shell("make -j8", "src/cache.cpp:12:3: error: use of undeclared identifier 'x'")
    shell("clang++ -c a.cpp", "(eval):1: command not found: clang++")
    assert stored() == []


def test_a_word_that_only_looks_like_a_build_tool_is_ignored(store):
    shell("echo make it so")
    shell("git commit -m 'makefile'")
    assert stored() == []


def test_a_build_after_other_commands_counts(store):
    shell("cd build && CC=clang make -j8")
    assert stored() == ["Build command: cd build && CC=clang make -j8"]


def test_a_file_most_notes_name_says_nothing(store):
    for index in range(30):
        remember(f"Change {index} touched src/core.cc and nothing else of note.")
    remember("src/rare.cc hides a retry loop.")
    assert touch("src/core.cc") == ""
    assert "retry loop" in touch("src/rare.cc")
