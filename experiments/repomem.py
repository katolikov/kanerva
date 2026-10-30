"""RepoMem: a multi-session coding-agent memory benchmark built from real git histories.

Each repository becomes one memory stream. Evenly spaced commits are snapshots; the interval between
two snapshots is a work session. During a session the simulated agent records notes about the commits
in the window and about the code facts it observed in the files that changed. Facts are extracted
deterministically with ``ast`` from the code at each snapshot, so every label comes from the repository
itself: a fact is stale when the same key held a different value at an earlier snapshot.

Queries are asked at later sessions. Track ``fact`` asks for the current value of a constant, function
signature, symbol location or dependency, and track ``task`` uses a real later commit subject as the
query, with earlier commits that touched the same files as gold.
"""

from __future__ import annotations

import ast
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from experiments.datasets import DATA, ROOT, Query, Stream, Turn

REPOS = {
    # repository: (package source root, split)
    "pallets/click": ("src/click", "dev"),
    "python-attrs/attrs": ("src/attr", "dev"),
    "pallets/flask": ("src/flask", "test"),
    "psf/requests": ("src/requests", "test"),
    "encode/httpx": ("httpx", "test"),
    "Textualize/rich": ("rich", "test"),
    "pallets/werkzeug": ("src/werkzeug", "test"),
    "pallets/jinja": ("src/jinja2", "test"),
    # Added after the first test run, for checking one post-hoc change on untouched repositories.
    "encode/starlette": ("starlette", "holdout"),
    "fastapi/typer": ("typer", "holdout"),
    "psf/black": ("src/black", "holdout"),
    "aio-libs/aiohttp": ("aiohttp", "holdout"),
    # Round-2 fresh test, named in docs/research-2/protocol.md before the histories were fetched.
    "marshmallow-code/marshmallow": ("src/marshmallow", "fresh"),
    "pytest-dev/pluggy": ("src/pluggy", "fresh"),
    "pypa/packaging": ("src/packaging", "fresh"),
    "Delgan/loguru": ("loguru", "fresh"),
    "tqdm/tqdm": ("tqdm", "fresh"),
}
LEGACY_ROOTS = {"src/click": "click", "src/attr": "src/attr", "src/flask": "flask", "src/requests": "requests",
                "src/werkzeug": "werkzeug", "src/jinja2": "jinja2", "src/black": "black",
                "src/marshmallow": "marshmallow", "src/pluggy": "pluggy", "src/packaging": "packaging"}

SNAPSHOTS = 14
YEARS = 6
SEEDS = (0, 1, 2)
MANIFEST = ROOT / "experiments" / "repomem_manifest.json"
BUILD = DATA / "repomem"

MAX_FACTS_PER_FILE = 12
MAX_COMMITS_PER_SESSION = 40
FACT_QUERIES_PER_SESSION = 18
TASK_QUERIES_PER_SESSION = 6


def git(repo_dir: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args], check=True, capture_output=True, text=True, errors="replace"
    ).stdout


def ensure_clone(repo: str) -> Path:
    target = BUILD / "repos" / (repo.replace("/", "__") + ".git")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--bare", f"https://github.com/{repo}.git", str(target)], check=True)
    return target


def _literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool)):
        text = repr(node.value)
        return text if len(text) <= 60 else None
    if isinstance(node, (ast.Tuple, ast.List)) and 0 < len(node.elts) <= 6:
        parts = [_literal(e) for e in node.elts]
        return None if None in parts else ("[" + ", ".join(parts) + "]")
    return None


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    parts: list[str] = []
    positional = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(positional, defaults):
        parts.append(arg.arg if default is None else f"{arg.arg}={ast.unparse(default)}")
    if args.vararg:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(arg.arg if default is None else f"{arg.arg}={ast.unparse(default)}")
    if args.kwarg:
        parts.append("**" + args.kwarg.arg)
    return ", ".join(p for p in parts if p not in ("self", "cls"))


def extract_facts(path: str, source: str) -> dict[str, tuple[str, str, str]]:
    """Facts in one file as key -> (kind, name, value)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    facts: dict[str, tuple[str, str, str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            value = _literal(node.value)
            if value is not None and not name.startswith("__"):
                facts[f"const:{path}:{name}"] = ("constant", name, value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            facts[f"sig:{node.name}"] = ("signature", node.name, _signature(node))
            facts[f"loc:{node.name}"] = ("location", node.name, path)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            facts[f"loc:{node.name}"] = ("location", node.name, path)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    not child.name.startswith("_") or child.name == "__init__"
                ):
                    qual = f"{node.name}.{child.name}"
                    facts[f"sig:{qual}"] = ("signature", qual, _signature(child))
    return facts


def render_fact(kind: str, name: str, value: str, path: str) -> str:
    if kind == "constant":
        return f"{name} in {path} is set to {value}"
    if kind == "signature":
        return f"{name}({value}) is the signature, defined in {path}"
    if kind == "location":
        return f"{name} is defined in {value}"
    return f"The project requires {name} {value}"


QUERY_TEMPLATES = {
    "constant": ["What is the value of {name}?", "What is {name} currently set to in {path}?", "Which value does the code use for {name}?"],
    "signature": ["What parameters does {name} take?", "What is the signature of {name}?", "How do I call {name}?"],
    "location": ["Where is {name} defined?", "Which file contains {name}?", "Which module should I open to edit {name}?"],
    "dependency": ["Which version of {name} does the project require?", "What is the {name} version constraint?"],
}

# Operator characters are excluded inside a clause, so the repetition cannot backtrack exponentially.
_DEP = re.compile(r"""["']\s*([A-Za-z0-9_.\-]+)\s*((?:[<>=!~]=?[^"';,<>=!~]*,?\s*)+)["']""")


def dependency_facts(source: str) -> dict[str, tuple[str, str, str]]:
    facts = {}
    for m in _DEP.finditer(source):
        name, spec = m.group(1), m.group(2).strip().rstrip(",")
        facts[f"dep:{name.lower()}"] = ("dependency", name, spec)
    return facts


def snapshot_facts(repo_dir: Path, sha: str, root: str) -> dict[str, tuple[str, str, str, str]]:
    """All facts at a commit as key -> (kind, name, value, path)."""
    listing = git(repo_dir, "ls-tree", "-r", "--name-only", sha).splitlines()
    roots = {root, LEGACY_ROOTS.get(root, root)}
    files = [f for f in listing if f.endswith(".py") and any(f.startswith(r + "/") for r in roots)]
    facts: dict[str, tuple[str, str, str, str]] = {}
    ambiguous: set[str] = set()
    for path in files:
        source = git(repo_dir, "show", f"{sha}:{path}")
        for key, (kind, name, value) in extract_facts(path, source).items():
            if key in facts:
                ambiguous.add(key)
            facts[key] = (kind, name, value, path)
    for key in ambiguous:
        del facts[key]
    for dep_file in ("pyproject.toml", "setup.py"):
        if dep_file in listing:
            for key, (kind, name, value) in dependency_facts(git(repo_dir, "show", f"{sha}:{dep_file}")).items():
                facts[key] = (kind, name, value, dep_file)
    return facts


@dataclass
class Commit:
    sha: str
    at: float
    subject: str
    files: list[str]


def _window_commits(repo_dir: Path, a: str, b: str) -> list[Commit]:
    log = git(repo_dir, "log", "--no-merges", "--first-parent", "--reverse", "--format=%x1e%H%x1f%ct%x1f%s",
              "--name-only", f"{a}..{b}")
    commits = []
    for chunk in log.split("\x1e"):
        if not chunk.strip():
            continue
        head, _, rest = chunk.partition("\n")
        sha, at, subject = head.split("\x1f", 2)
        files = [f for f in rest.splitlines() if f.strip()]
        commits.append(Commit(sha, float(at), subject, files))
    return commits


def build_repo(repo: str) -> dict:
    """Extract and cache snapshots, commit windows and facts for one repository."""
    root, split = REPOS[repo]
    repo_dir = ensure_clone(repo)
    head = git(repo_dir, "rev-parse", "HEAD").strip()
    lines = git(repo_dir, "log", "--first-parent", "--format=%H %ct", head).splitlines()
    history = [(ln.split()[0], float(ln.split()[1])) for ln in lines][::-1]
    latest = history[-1][1]
    span = [h for h in history if h[1] >= latest - YEARS * 365.25 * 86400]
    step = (len(span) - 1) / (SNAPSHOTS - 1)
    picks = [span[round(i * step)] for i in range(SNAPSHOTS)]
    snapshots = []
    for sha, at in picks:
        snapshots.append({"sha": sha, "at": at, "facts": snapshot_facts(repo_dir, sha, root)})
    windows = []
    for a, b in zip(picks, picks[1:]):
        windows.append([c.__dict__ for c in _window_commits(repo_dir, a[0], b[0])])
    return {"repo": repo, "head": head, "split": split, "snapshots": snapshots, "windows": windows}


def _commit_note(commit: dict) -> str:
    files = ", ".join(commit["files"][:6])
    return f"commit {commit['sha'][:8]}: {commit['subject']} (files: {files})"


def stream_for(data: dict, seed: int) -> Stream:
    rng = random.Random(f"{data['repo']}-{seed}")
    writes: list[Turn] = []
    queries: list[Query] = []
    versions: dict[str, list[tuple[str, str, str, str, str]]] = {}  # key -> [(text, kind, name, value, path)]
    snaps = data["snapshots"]
    past_commits: list[dict] = []
    repo_name = data["repo"].split("/")[1]
    current = ["0"]

    def observe(key: str, fact: list, at: float) -> None:
        kind, name, value, path = fact
        text = render_fact(kind, name, value, path)
        chain = versions.setdefault(key, [])
        if chain and chain[-1][0] == text:
            writes.append(Turn(text, at, key=key, kind="semantic", session=current[0]))  # reconfirmation
            return
        chain.append((text, kind, name, value, path))
        writes.append(Turn(text, at, key=key, kind="semantic", session=current[0]))

    first = snaps[0]
    initial = sorted(first["facts"])
    rng.shuffle(initial)
    for key in initial[:300]:
        observe(key, first["facts"][key], first["at"])

    for session, window in enumerate(data["windows"]):
        current[0] = str(session + 1)
        before, after = snaps[session], snaps[session + 1]
        span_days = max((after["at"] - before["at"]) / 86400, 1.0)
        commits = window if len(window) <= MAX_COMMITS_PER_SESSION else rng.sample(window, MAX_COMMITS_PER_SESSION)
        commits = sorted(commits, key=lambda c: c["at"])
        for commit in commits:
            writes.append(Turn(_commit_note(commit), commit["at"], session=current[0]))
        changed = [k for k, f in after["facts"].items() if k in before["facts"] and before["facts"][k][2:] != f[2:]]
        changed_paths = {after["facts"][k][3] for k in changed}
        touched = {f for c in window for f in c["files"]} | changed_paths
        for key in changed:
            if key in versions:
                observe(key, after["facts"][key], after["at"])
        stable = [k for k, f in after["facts"].items() if k not in changed and f[3] in touched]
        rng.shuffle(stable)
        for key in stable[: MAX_FACTS_PER_FILE * max(len(touched & changed_paths), 2)]:
            observe(key, after["facts"][key], after["at"])

        if session < 2:
            past_commits.extend(commits)
            continue
        now = after["at"] + span_days * 86400 * 0.01
        known = [k for k, chain in versions.items() if k in after["facts"] and chain[-1][0] == render_fact(*after["facts"][k])]
        moved = [k for k in known if len(versions[k]) > 1]
        still = [k for k in known if len(versions[k]) == 1]
        rng.shuffle(moved)
        rng.shuffle(still)
        n_moved = min(len(moved), FACT_QUERIES_PER_SESSION * 2 // 3)
        chosen = moved[:n_moved] + still[: FACT_QUERIES_PER_SESSION - n_moved]
        for index, key in enumerate(chosen):
            chain = versions[key]
            text, kind, name, value, path = chain[-1]
            template = rng.choice(QUERY_TEMPLATES[kind])
            question = template.format(name=name, path=path)
            category = f"fact-{kind}-{'changed' if len(chain) > 1 else 'stable'}"
            queries.append(
                Query(len(writes), f"{repo_name}-s{session}-f{index}", question, now, category,
                      gold={text}, stale={c[0] for c in chain[:-1] if c[0] != text}, answer=value)
            )
        tasks = [c for c in commits if 1 <= len(c["files"]) <= 6 and len(c["subject"]) > 15]
        rng.shuffle(tasks)
        emitted = 0
        for commit in tasks:
            files = set(commit["files"])
            gold = {
                _commit_note(p) for p in past_commits
                if len(files & set(p["files"])) / len(files | set(p["files"])) >= 0.5 and p["subject"] != commit["subject"]
            }
            if gold:
                queries.append(Query(len(writes), f"{repo_name}-s{session}-t{emitted}", commit["subject"], commit["at"], "task", gold=gold))
                emitted += 1
            if emitted >= TASK_QUERIES_PER_SESSION:
                break
        past_commits.extend(commits)
    order = sorted(range(len(writes)), key=lambda i: writes[i].at)
    ordered = [writes[i] for i in order]
    for query in queries:
        query.pos = sum(1 for t in ordered if t.at < query.now)
    return Stream(f"{repo_name}-{seed}", data["split"], ordered, sorted(queries, key=lambda q: q.pos))


def load_repomem(holdout: bool = False, split: str | None = None) -> list[Stream]:
    """Round-1 dev and test repositories by default; ``holdout=True`` or ``split`` picks another group."""
    wanted = split or ("holdout" if holdout else None)
    streams = []
    for repo, (_, repo_split) in REPOS.items():
        if wanted is None and repo_split not in ("dev", "test") or wanted is not None and repo_split != wanted:
            continue
        data = json.loads((BUILD / (repo.replace("/", "__") + ".json")).read_text())
        for seed in SEEDS:
            streams.append(stream_for(data, seed))
    return streams


def main() -> None:
    manifest = {}
    for repo in REPOS:
        path = BUILD / (repo.replace("/", "__") + ".json")
        if path.exists():
            data = json.loads(path.read_text())
        else:
            data = build_repo(repo)
            BUILD.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        manifest[repo] = {
            "head": data["head"],
            "snapshots": [s["sha"] for s in data["snapshots"]],
            "split": data["split"],
            "n_facts_last": len(data["snapshots"][-1]["facts"]),
        }
        print(repo, data["head"][:10], manifest[repo]["n_facts_last"], [len(w) for w in data["windows"]], flush=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    sys.exit(main())
