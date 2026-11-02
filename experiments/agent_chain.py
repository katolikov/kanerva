"""Does learning from verdicts change what a real agent gets? Chains of sessions on one store, with learning on and off.

Each chain is a repository the method has never seen, a store of the size real projects reach, and a run of documentation
chores. A chore needs one fact, and each need repeats an earlier one with the share measured on real sessions, where 35% of
turns that touch files return to a file from an earlier session. A prompt that names a symbol leaves nothing to learn, since search
puts its note first, and real prompts are vaguer, so a cheap model rewrites each need in plain words, with no identifier or file
name and differently for each repeat. Questions are kept in experiments/chain_questions.json. Both arms get the same store and the
same chores in the same order, and the agent gives its verdict at each stop in both. The first half of a chain is for learning,
and the second is scored.
The primary measures and the bar in ``PASS`` were set on the development repositories, before the first run on ``REPOS``. The store also carries the shadow check's record,
which says whether learning would have switched itself on.

Needs the claude CLI signed in, and spends real tokens. Sessions load no user settings and write no transcripts, and the store
and the project live in a temporary folder. Writes results/agent_chain.json.

Usage: python -m experiments.agent_chain [sessions per chain] [repositories]
"""

from __future__ import annotations

import itertools
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from experiments.agent_signal import MODEL, MODELS_HOME, run_agent
from experiments.agent_write_path import SESSION_VARIABLES
from experiments.harness import RESULTS, git_commit
from experiments.repomem import (
    c_snapshot_facts,
    ensure_clone,
    extract_c_facts,
    extract_cpp_facts,
    git,
    render_fact,
    snapshot_facts,
)
from kanerva import Memory
from kanerva.embed import OnnxEmbedder, fetch_minilm, minilm_folder
from kanerva.paths import project_scope

SESSIONS = 24
RECURRENCE = 0.353  # results/real_gate.json
STORE_NOTES = 160
EDIT_NOTES = 30
WORKERS = 6
BUDGETS = {"default": "600"}
QUESTIONS = Path(__file__).resolve().parent / "chain_questions.json"
REWRITER = "claude-haiku-5-5"
REWRITES = 3
REWRITE = (
    "Facts from a codebase follow, one per line, each after its number. For each fact write {n} different questions a developer might ask "
    "when they need it. Use plain words, and describe the thing by what it does or is for. Use no identifier, file name or value from the fact. "
    'Reply with JSON only, mapping each number to its list of questions: {{"0": ["...", "..."]}}.\n\n{facts}'
)
PROMPT = "{question} Put the answer in docs/round{n}/answer.md, who the project is for in docs/round{n}/audience.md, and one line per file in docs/round{n}/index.md. Then stop."
ARMS = {"fixed": "shadow", "learned": "auto"}  # verdicts in both, and in the second what ships by default
PASS = {"loss": 0.05}  # learning passes if sessions a note helped rise, or lookups or injected tokens fall, with an interval clear of zero, and neither those sessions nor right answers fall by more than this
# Repositories no earlier test has read: (source root, reader of its facts).
REPOS = {
    "dateutil/dateutil": ("src/dateutil", "python"),
    "PyCQA/isort": ("isort", "python"),
    "Pylons/pyramid": ("src/pyramid", "python"),
    "jemalloc/jemalloc": ("src", "c"),
    "ToruNiina/toml11": ("include", "cpp"),
    "fastfloat/fast_float": ("include", "cpp"),
}
SPENT_REPOS = {  # read by a run on an earlier build, whose results are in results/agent_chain_first.json
    "yaml/pyyaml": ("lib/yaml", "python"),
    "lepture/mistune": ("src/mistune", "python"),
    "hynek/structlog": ("src/structlog", "python"),
    "Cyan4973/xxHash": ("", "c"),
    "libssh2/libssh2": ("src", "c"),
    "gulrak/filesystem": ("include", "cpp"),
}
DEV_REPOS = {"python-poetry/tomlkit": ("tomlkit", "python"), "marzer/tomlplusplus": ("include", "cpp")}  # settings were chosen on these


def facts_at_head(repo: str) -> tuple[Path, str, dict]:
    root, reader = (REPOS | SPENT_REPOS | DEV_REPOS)[repo]
    clone = ensure_clone(repo)
    head = git(clone, "rev-parse", "HEAD").strip()
    if reader == "python":
        return clone, head, snapshot_facts(clone, head, root)
    return clone, head, c_snapshot_facts(clone, head, root, extract_c_facts if reader == "c" else extract_cpp_facts)


def ask_for_questions(facts: dict[str, str]) -> dict[str, list[str]]:
    listing = "\n".join(f"{i}: {text}" for i, text in enumerate(facts.values()))
    env = {k: v for k, v in os.environ.items() if k not in SESSION_VARIABLES}
    command = ["claude", "-p", REWRITE.format(n=REWRITES, facts=listing), "--model", REWRITER, "--setting-sources", "project", "--no-session-persistence", "--tools", "", "--output-format", "json"]
    done = subprocess.run(command, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600, cwd=tempfile.gettempdir())
    reply = json.loads(done.stdout)["result"]
    written = json.loads(reply[reply.index("{"): reply.rindex("}") + 1])
    return {key: written[str(i)] for i, key in enumerate(facts) if str(i) in written}


def vague_questions(repo: str, keys: list[str], facts: dict) -> dict[str, list[str]]:
    kept = json.loads(QUESTIONS.read_text()) if QUESTIONS.exists() else {}
    have = kept.setdefault(repo, {})
    missing = [k for k in keys if k not in have]
    for start in range(0, len(missing), 40):
        have |= ask_for_questions({k: render_fact(*facts[k][:3], facts[k][3]) for k in missing[start:start + 40]})
    QUESTIONS.write_text(json.dumps(kept, indent=1, sort_keys=True))
    return {k: have[k] for k in keys if k in have}


def make_chain(repo: str, rng: random.Random, sessions: int) -> dict:
    clone, head, facts = facts_at_head(repo)
    facts = {k: f for k, f in facts.items() if f[0] != "dependency" or any(c.isdigit() for c in f[2])}  # console-script entries read as constraints
    answerable = sorted(k for k, f in facts.items() if f[0] in ("constant", "signature"))
    pool = rng.sample(answerable, min(sessions + 8, len(answerable)))
    questions = vague_questions(repo, pool, facts)
    needs: list[str] = []
    tasks = []
    for session in range(sessions):
        fresh = [k for k in pool if k not in needs and k in questions]
        key = rng.choice(needs) if needs and rng.random() < RECURRENCE else rng.choice(fresh)
        turn = needs.count(key) % len(questions[key])
        needs.append(key)
        tasks.append({"keys": [key], "prompt": PROMPT.format(question=questions[key][turn], n=session), "values": [facts[key][2]], "repeat": turn > 0})
    filler = rng.sample(sorted(set(facts) - set(needs)), min(STORE_NOTES - EDIT_NOTES - len(set(needs)), len(facts) - len(set(needs))))
    paths = sorted({facts[k][3] for k in facts})
    start = time.time() - 60 * 86400
    notes = [{"key": k, "text": render_fact(*facts[k][:3], facts[k][3]), "kind": "semantic", "memory_key": None} for k in sorted(set(needs)) + filler]
    notes += [{"key": None, "text": f"Edited {rng.choice(paths)}", "kind": "episodic", "memory_key": None} for _ in range(EDIT_NOTES)]
    rng.shuffle(notes)
    for index, note in enumerate(notes):
        note["at"] = start + index * 3000.0
    return {"repo": repo, "clone": str(clone), "head": head, "notes": notes, "tasks": tasks}


def seed_store(chain: dict, folder: Path, embedder: OnnxEmbedder) -> tuple[Path, Path, dict[str, int]]:
    home, project = folder / "kanerva", folder / "project"
    home.mkdir(parents=True)
    project.mkdir()
    archive = subprocess.run(["git", "-C", chain["clone"], "archive", chain["head"]], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(project)], input=archive, check=True)
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    (home / "models").symlink_to(MODELS_HOME / "models")
    mem = Memory(home / "memory.sqlite", embedder=embedder)
    scope = project_scope(str(project))
    ids = {}
    for note in chain["notes"]:
        item = mem.write(note["text"], at=note["at"], scope=scope, kind=note["kind"], key=note["memory_key"])
        if note["key"]:
            ids[note["key"]] = item
    mem.commit()
    mem.close()
    return home, project, ids


def session_row(index: int, task: dict, run: dict, home: Path, project: Path, ids: dict[str, int], seen: tuple[int, float]) -> tuple[dict, tuple[int, float]]:
    last_retrieval, last_verdict = seen
    with sqlite3.connect(home / "memory.sqlite") as db:
        rows = db.execute("SELECT id, item_ids FROM retrievals WHERE id > ?", (last_retrieval,)).fetchall()
        shown = {i for _, item_ids in rows for i in json.loads(item_ids)}
        tokens = sum(t for i in shown for (t,) in db.execute("SELECT tokens + 14 FROM items WHERE id = ?", (i,)))
        marks = db.execute("SELECT item_id, helped, at FROM verdicts WHERE item_id IS NOT NULL AND at > ?", (last_verdict,)).fetchall()
        wrong = [text[:90] for (text,) in db.execute(f"SELECT text FROM items WHERE id IN ({','.join(str(i) for i, h, _ in marks if not h) or 0})")]
        retired = db.execute("SELECT count(*) FROM items WHERE valid_to IS NOT NULL AND id <= ?", (STORE_NOTES,)).fetchone()[0]
        settings = {k: json.loads(v) for k, v in db.execute("SELECT key, value FROM meta WHERE key = 'shadow:*'")}
        probed = db.execute("SELECT count(*) FROM gate_log WHERE retrieval_id > ? AND note_weight > 1", (last_retrieval,)).fetchone()[0]
        resolved = db.execute("SELECT count(*) FROM retrievals WHERE id > ? AND resolved = 1", (last_retrieval,)).fetchone()[0]
    helped, misled = {i for i, h, _ in marks if h}, {i for i, h, _ in marks if not h}
    seeded = set(ids.values())
    wanted = {ids[k] for k in task["keys"]}
    calls = run["calls"]
    before = calls[:calls.index("Write")] if "Write" in calls else calls
    answers = [project / "docs" / f"round{index}" / "answer.md"]
    right = [path.exists() and value.replace(" ", "") in path.read_text(errors="replace").replace(" ", "").replace("\n", "") for path, value in zip(answers, task["values"])]
    row = {"session": index, "repeat": task["repeat"], "judged": "memory_feedback" in calls, "shown": len(shown), "needed_shown": len(shown & wanted), "helped": len(helped & shown), "helped_needed": len(helped & wanted),
           "misled": len(misled), "tokens": tokens, "lookups": sum(c in ("Grep", "Read", "Glob", "Bash") for c in before), "right": sum(right), "cost": run["cost"], "retrievals": len(rows), "resolved": resolved, "helped_seeded": len(helped & shown & seeded), "probed": probed, "wrong": wrong, "retired": retired, "shadow_on": bool(settings.get("shadow:*", {}).get("on"))}
    return row, (max([r for r, _ in rows], default=last_retrieval), max([at for _, _, at in marks], default=last_verdict))


def run_chain(chain: dict, budget: str, arm: str, folder: Path, embedder: OnnxEmbedder) -> dict:
    home, project, ids = seed_store(chain, folder, embedder)
    extra = {"KANERVA_LEARN": ARMS[arm], "KANERVA_PROMPT_BUDGET": BUDGETS[budget]}
    seen, rows = (0, 0.0), []
    for index, task in enumerate(chain["tasks"]):
        try:
            run = run_agent(task["prompt"], project, home, extra)
        except subprocess.TimeoutExpired:
            run = {"calls": [], "cost": 0.0}
        row, seen = session_row(index, task, run, home, project, ids, seen)
        rows.append(row)
        print(json.dumps({"repo": chain["repo"], "budget": budget, "arm": arm} | row), flush=True)
    with sqlite3.connect(home / "memory.sqlite") as db:
        shadow = {key: json.loads(value) for key, value in db.execute("SELECT key, value FROM meta WHERE key LIKE 'shadow:%'")}
        rewarded = db.execute("SELECT count(*) FROM items WHERE used > 1").fetchone()[0]
    return {"repo": chain["repo"], "budget": budget, "arm": arm, "sessions": rows, "shadow": {"*" if k.endswith("*") else "project": v for k, v in shadow.items()}, "notes_with_record": rewarded}


def half_means(chain: dict) -> dict:
    rows = chain["sessions"][len(chain["sessions"]) // 2:]
    shown = sum(r["shown"] for r in rows)
    wanted = len(rows)
    repeats = [r for r in rows if r["repeat"]]
    return {"helped_count": sum(r["helped"] > 0 for r in rows), "right_count": sum(r["right"] for r in rows), "probed": sum(r["probed"] for r in chain["sessions"]) / len(chain["sessions"]), "seeded_share": sum(r["helped_seeded"] for r in rows) / max(sum(r["helped"] for r in rows), 1), "helped_sessions": sum(r["helped"] > 0 for r in rows) / wanted, "repeat_helped": sum(r["helped"] > 0 for r in repeats) / max(len(repeats), 1), "repeat_needed_shown": sum(r["needed_shown"] for r in repeats) / max(len(repeats), 1), "helped_share": sum(r["helped"] for r in rows) / max(shown, 1), "needed_shown": sum(r["needed_shown"] for r in rows) / wanted, "tokens": float(np.mean([r["tokens"] for r in rows])),
            "lookups": float(np.mean([r["lookups"] for r in rows])), "right": sum(r["right"] for r in rows) / wanted, "cost": float(np.mean([r["cost"] for r in rows])), "shown": float(np.mean([r["shown"] for r in rows]))}


def bootstrap_ci(delta: np.ndarray) -> np.ndarray:
    """Percentile interval of the mean over repositories."""
    rng = np.random.default_rng(0)
    return np.percentile(rng.choice(delta, (10000, len(delta))).mean(axis=1), [2.5, 97.5])


def compare_arms(chains: list[dict]) -> dict:
    out: dict = {}
    for budget in BUDGETS:
        by = {(c["repo"], c["arm"]): half_means(c) for c in chains if c["budget"] == budget}
        repos = sorted({repo for repo, _ in by})
        if not repos:
            continue
        table = {}
        for measure in ("helped_count", "right_count", "probed", "seeded_share", "helped_sessions", "repeat_helped", "helped_share", "needed_shown", "repeat_needed_shown", "tokens", "lookups", "right", "cost", "shown"):
            fixed = np.array([by[(r, "fixed")][measure] for r in repos])
            learned = np.array([by[(r, "learned")][measure] for r in repos])
            delta = learned - fixed
            flips = np.array(list(itertools.product((1, -1), repeat=len(delta))))
            p = float((np.abs((flips * delta).mean(axis=1)) >= abs(delta.mean()) - 1e-12).mean())
            table[measure] = {"p": round(p, 3), "fixed": round(float(fixed.mean()), 4), "learned": round(float(learned.mean()), 4), "delta": round(float(delta.mean()), 4),
                              "interval": [round(float(x), 4) for x in bootstrap_ci(delta)], "up": int((delta > 0).sum()), "down": int((delta < 0).sum())}
        gain = table["helped_sessions"]["interval"][0] > 0 or table["lookups"]["interval"][1] < 0 or table["tokens"]["interval"][1] < 0
        table["passed"] = bool(gain and table["helped_sessions"]["delta"] >= -PASS["loss"] and table["right"]["delta"] >= -PASS["loss"])
        out[budget] = table
    return out


def lift_active(chains: list[dict]) -> dict:
    """In how many acting chains the shadow check switched lift on, and in how many scored sessions it was on."""
    acting = [c["sessions"] for c in chains if c["arm"] == "learned"]
    return {"chains": sum(any(s["shadow_on"] for s in rows) for rows in acting),
            "scored_sessions": sum(s["shadow_on"] for rows in acting for s in rows[len(rows) // 2:])}


def main(sessions: int = SESSIONS, repos: tuple[str, ...] = tuple(REPOS)) -> None:
    rng = random.Random("agent-chain-1")
    version = subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
    os.environ["KANERVA_HOME"] = str(MODELS_HOME)
    fetch_minilm()
    embedder = OnnxEmbedder(minilm_folder())
    chains = [make_chain(repo, rng, sessions) for repo in repos]
    folder = Path(tempfile.mkdtemp(prefix="kanerva-chain-")).resolve()
    start = time.time()
    jobs = [(chain, budget, arm) for chain in chains for budget in BUDGETS for arm in ARMS]
    with ThreadPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(lambda pair: run_chain(*pair[1], folder / f"chain-{pair[0]}", embedder), enumerate(jobs)))
    shutil.rmtree(folder, ignore_errors=True)
    for path in (Path.home() / ".claude" / "projects").glob(f"*{folder.name.replace('_', '-')}*"):
        shutil.rmtree(path, ignore_errors=True)
    (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
    (RESULTS / "raw" / "agent_chain.json").write_text(json.dumps(rows))
    report = {"meta": {"experiment": "learning from verdicts on chains of real sessions", "commit": git_commit(), "agent": version, "model": MODEL, "sessions": sessions, "recurrence": RECURRENCE,
                       "store_notes": STORE_NOTES, "repos": list(repos), "pass": PASS, "cost_usd": round(sum(s["cost"] for c in rows for s in c["sessions"]), 2), "seconds": round(time.time() - start)},
              "budgets": compare_arms(rows), "lift": lift_active(rows), "shadow": [{"repo": c["repo"], "budget": c["budget"], "arm": c["arm"], "records": c["shadow"]} for c in rows]}
    (RESULTS / "agent_chain.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else SESSIONS, tuple(sys.argv[2:]) or tuple(REPOS)))
