"""Wall time of real hook calls: the plugin launcher, a new process per call, and a store of real notes.

Each mode also gets one first call with an empty uv cache, which is what a new user meets: uv downloads the packages
during that call. Writes results/hook_latency.json.

Usage: python -m experiments.hook_latency <folder with the MiniLM model>
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from experiments.embed_cache import TextCache
from experiments.harness import RESULTS, git_commit
from experiments.registry import load
from kanerva import Memory
from kanerva.paths import project_scope

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "plugins" / "claude-code" / "scripts" / "run.sh"
SIZES = (1000, 10000)
RUNS = 15
MODES = {"lexical": (None, None), "static": ("static", "lme-m2v"), "minilm": ("minilm", "lme")}
EVENTS = ("prompt", "tool", "stop")


def hook(env: dict, payload: dict, cwd: Path) -> tuple[float, str]:
    start = time.perf_counter()
    done = subprocess.run(["sh", str(LAUNCHER), "hook"], input=json.dumps(payload), capture_output=True, text=True, env=env, cwd=cwd)
    return (time.perf_counter() - start) * 1000, done.stdout


def fill(store: Path, project: Path, texts: list[str], cache_name: str | None) -> None:
    cache = TextCache(cache_name) if cache_name else None
    mem = Memory(store / "memory.sqlite")
    scope = project_scope(str(project))
    for i, text in enumerate(texts):
        mem.write(text, scope=scope, kind="semantic", at=1.6e9 + i * 60, session=str(i // 40),
                  embedding=None if cache is None else cache.vector(text))
        if i % 1000 == 0:
            mem.commit()
    mem.close()


def summary(values: list[float]) -> dict:
    return {"p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)), "max_ms": float(max(values)), "n": len(values)}


def payload(event: str, i: int, project: Path, questions: list[str]) -> dict:
    base = {"session_id": f"s{i}", "cwd": str(project)}
    if event == "prompt":
        return base | {"hook_event_name": "UserPromptSubmit", "prompt": questions[i % len(questions)]}
    if event == "tool":
        return base | {"hook_event_name": "PostToolUse", "tool_name": "Edit", "tool_input": {"file_path": f"src/m{i}.py"}, "tool_response": {}}
    return base | {"hook_event_name": "Stop", "last_assistant_message": "Done."}


def measure(mode: str, size: int, texts: list[str], questions: list[str], model: Path) -> dict:
    name, cache_name = MODES[mode]
    folder = Path(tempfile.mkdtemp(prefix="kanerva-latency-"))
    store, project = folder / "store", folder / "project"
    project.mkdir()
    store.mkdir()
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    fill(store, project, texts[:size], cache_name)
    if mode == "minilm":
        shutil.copytree(model, store / "models" / "all-MiniLM-L6-v2")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KANERVA", "CLAUDE"))}
    env |= {"KANERVA_HOME": str(store), "KANERVA_PROJECT_DIR": str(project), "HF_HUB_OFFLINE": "1"}
    if name:
        env["KANERVA_EMBEDDER"] = name
    out: dict = {}
    if size == SIZES[0]:
        out["first_call_ms"], answer = hook(env | {"UV_CACHE_DIR": str(folder / "uv")}, payload("prompt", 0, project, questions), project)
        out["first_call_answered"] = '"additionalContext"' in answer
    hook(env, payload("prompt", 0, project, questions), project)  # uv's environment for this mode now exists
    for event in EVENTS:
        out[event] = summary([hook(env, payload(event, i, project, questions), project)[0] for i in range(RUNS)])
    answers = [hook(env, payload("prompt", i, project, questions), project)[1] for i in range(RUNS)]
    out["prompts_with_notes"] = sum('"additionalContext"' in answer for answer in answers)
    shutil.rmtree(folder, ignore_errors=True)
    print(mode, size, json.dumps(out), flush=True)
    return out


def main() -> None:
    model = Path(sys.argv[1])
    streams = load("lme")
    texts = list(dict.fromkeys(t.text for s in streams for t in s.writes))
    np.random.default_rng(0).shuffle(texts)
    questions = [q.question for s in streams for q in s.queries][:RUNS]
    results = {f"{mode}@{size}": measure(mode, size, texts, questions, model) for mode in MODES for size in SIZES}
    meta = {"experiment": "hook wall time", "commit": git_commit(), "runs": RUNS, "machine": platform.processor()}
    (RESULTS / "hook_latency.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1))


if __name__ == "__main__":
    main()
