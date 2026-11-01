"""RepoMem: a multi-session coding-agent memory benchmark built from real git histories.

Each repository becomes one memory stream. Evenly spaced commits are snapshots; the interval between
two snapshots is a work session. During a session the simulated agent records notes about the commits
in the window and about the code facts it observed in the files that changed. Facts are extracted
deterministically from the code at each snapshot, with ``ast`` for Python and regular expressions for C and
C++, so every label comes from the repository itself: a fact is stale when the same key held a different value
at an earlier snapshot.

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
    # Extra repositories that no parameter was fitted on.
    "encode/starlette": ("starlette", "holdout"),
    "fastapi/typer": ("typer", "holdout"),
    "psf/black": ("src/black", "holdout"),
    "aio-libs/aiohttp": ("aiohttp", "holdout"),
    # The held-out test.
    "marshmallow-code/marshmallow": ("src/marshmallow", "fresh"),
    "pytest-dev/pluggy": ("src/pluggy", "fresh"),
    "pypa/packaging": ("src/packaging", "fresh"),
    "Delgan/loguru": ("loguru", "fresh"),
    "tqdm/tqdm": ("tqdm", "fresh"),
    # A second held-out test, for changes made after the first one.
    "encode/uvicorn": ("uvicorn", "new"),
    "agronholm/anyio": ("src/anyio", "new"),
    "python-trio/trio": ("src/trio", "new"),
    "tox-dev/tox": ("src/tox", "new"),
    "Pylons/waitress": ("src/waitress", "new"),
    "benoitc/gunicorn": ("gunicorn", "new"),
    "sqlalchemy/alembic": ("alembic", "new"),
    "python-jsonschema/jsonschema": ("jsonschema", "new"),
    # A check beyond Python. The task track needs only git log, so these have no source root and no facts.
    "vitejs/vite": (None, "other"),
    "honojs/hono": (None, "other"),
    "colinhacks/zod": (None, "other"),
    "gin-gonic/gin": (None, "other"),
    "caddyserver/caddy": (None, "other"),
    "junegunn/fzf": (None, "other"),
    "BurntSushi/ripgrep": (None, "other"),
    "tokio-rs/axum": (None, "other"),
    "clap-rs/clap": (None, "other"),
    # The third held-out test, for the method fitted on every repository above.
    "scrapy/scrapy": ("scrapy", "final"),
    "celery/celery": ("celery", "final"),
    "httpie/cli": ("httpie", "final"),
    "sanic-org/sanic": ("sanic", "final"),
    "boto/boto3": ("boto3", "final"),
    "django/channels": ("channels", "final"),
    "vuejs/core": (None, "final"),
    "TanStack/query": (None, "final"),
    "immerjs/immer": (None, "final"),
    "spf13/cobra": (None, "final"),
    "labstack/echo": (None, "final"),
    "charmbracelet/bubbletea": (None, "final"),
    "sharkdp/bat": (None, "final"),
    "hyperium/hyper": (None, "final"),
    "sharkdp/fd": (None, "final"),
    # C and C++. The first group is for fixing what these languages break, and the second is a test nothing was tuned on.
    "tmux/tmux": ("", "c-dev"),
    "libuv/libuv": ("src", "c-dev"),
    "nlohmann/json": ("include/nlohmann", "c-dev"),
    "google/googletest": ("googletest", "c-dev"),
    "redis/redis": ("src", "c-test"),
    "curl/curl": ("lib", "c-test"),
    "jqlang/jq": ("src", "c-test"),
    "fmtlib/fmt": ("include/fmt", "c-test"),
    "gabime/spdlog": ("include/spdlog", "c-test"),
    "google/leveldb": ("db", "c-test"),
    # After the test above, its six repositories joined the fit, so a change for these languages needs repositories of its own.
    "nginx/nginx": ("src", "c-final"),
    "libgit2/libgit2": ("src", "c-final"),
    "libevent/libevent": ("", "c-final"),
    "catchorg/Catch2": ("src/catch2", "c-final"),
    "google/benchmark": ("src", "c-final"),
    "ocornut/imgui": ("", "c-final"),
    # After that test: these also read templates, overloads and class methods, which the groups above leave out.
    "lz4/lz4": ("lib", "c-last"),
    "facebook/zstd": ("lib", "c-last"),
    "libexpat/libexpat": ("expat/lib", "c-last"),
    "yhirose/cpp-httplib": ("", "c-last"),
    "CLIUtils/CLI11": ("include/CLI", "c-last"),
    "jbeder/yaml-cpp": ("include/yaml-cpp", "c-last"),
}
C_SPLITS = ("c-dev", "c-test", "c-final", "c-last")
LANGUAGES = {
    "TypeScript": ("vite", "hono", "zod"),
    "Go": ("gin", "caddy", "fzf"),
    "Rust": ("ripgrep", "axum", "clap"),
}
C_LANGUAGES = {"C": ("redis", "curl", "jq"), "C++": ("fmt", "spdlog", "leveldb")}
C_FINAL_LANGUAGES = {"C": ("nginx", "libgit2", "libevent"), "C++": ("Catch2", "benchmark", "imgui")}
C_LAST_LANGUAGES = {"C": ("lz4", "zstd", "libexpat"), "C++": ("cpp-httplib", "CLI11", "yaml-cpp")}
FINAL_LANGUAGES = {
    "Python": ("scrapy", "celery", "cli", "sanic", "boto3", "channels"),
    "TypeScript": ("core", "query", "immer"),
    "Go": ("cobra", "echo", "bubbletea"),
    "Rust": ("bat", "hyper", "fd"),
}
LEGACY_ROOTS = {"src/click": "click", "src/attr": "src/attr", "src/flask": "flask", "src/requests": "requests",
                "src/werkzeug": "werkzeug", "src/jinja2": "jinja2", "src/black": "black",
                "src/marshmallow": "marshmallow", "src/pluggy": "pluggy", "src/packaging": "packaging",
                "src/anyio": "anyio", "src/trio": "trio", "src/tox": "tox", "src/waitress": "waitress"}

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


C_HEADERS = (".h", ".hh", ".hpp", ".hxx")
C_SOURCES = (".c", ".cc", ".cpp", ".cxx")
C_KEYWORDS = frozenset("""if for while switch return sizeof catch defined alignof decltype static_assert operator""".split())
_C_DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w{2,})[ \t]+(.+?)[ \t]*(?:/[/*].*)?$", re.M)
_C_CONST = re.compile(r"^[ \t]*(?:static[ \t]+)?(?:inline[ \t]+)?(?:constexpr|const)[ \t]+[\w:<>, ]*?[ \t&*]([A-Za-z_]\w{2,})[ \t]*=[ \t]*([^;{]{1,40});", re.M)
_C_LITERAL = re.compile(r"""^\(?-?(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?)[uUlLfF]*\)?$|^"[^"\n]{0,40}"$""")
_C_FUNCTION = re.compile(r"^(?P<lead>[ \t]*)(?P<head>(?:[A-Za-z_][\w:<>,*&~\[\]]*[ \t\n*&]+)+)(?P<name>[A-Za-z_]\w*)[ \t]*\((?P<args>[^;{}()]*)\)"
                         r"[ \t\n]*(?:const[ \t\n]*)?(?:noexcept[ \t\n]*)?(?:override[ \t\n]*)?(?:=[ \t]*(?:0|default|delete)[ \t\n]*)?(?P<end>[;{])", re.M)
_C_TYPE = re.compile(r"^[ \t]*(?:typedef[ \t]+)?(?:template[ \t]*<[^;{]*>[ \t\n]*)?(?:struct|class|union|enum(?:[ \t]+class)?)[ \t]+(?:[A-Z_]+_EXPORT[ \t]+)?([A-Za-z_]\w+)[^;{()]*\{", re.M)
_C_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)


def extract_c_facts(path: str, source: str) -> dict[str, tuple[str, str, str]]:
    """Facts in one C or C++ file: literal macros and constants, public functions, and types."""
    facts: dict[str, tuple[str, str, str]] = {}
    for pattern in (_C_DEFINE, _C_CONST):
        for name, value in pattern.findall(source):
            if _C_LITERAL.match(value.strip()):
                facts[f"const:{path}:{name}"] = ("constant", name, value.strip())
    code = _C_COMMENT.sub("", source)
    found: dict[str, list[str]] = {}
    for m in _C_FUNCTION.finditer(code):
        name, head = m["name"], m["head"]
        if name in C_KEYWORDS or name.startswith("_") or name.isupper() or re.search(r"\b(?:static|return|else|new|delete|typedef|goto|case)\b", head):
            continue
        if path.endswith(C_SOURCES) and (m["end"] != "{" or m["lead"]):  # in a source file only a definition at the left margin counts
            continue
        found.setdefault(name, []).append(" ".join(m["args"].split()))
    for name, values in found.items():
        if len(set(values)) == 1 and len(values[0]) <= 100:  # an overloaded name has no single signature
            facts[f"sig:{name}"] = ("signature", name, values[0])
            facts[f"loc:{name}"] = ("location", name, path)
    for name in _C_TYPE.findall(code):
        if not name.startswith("_"):
            facts[f"loc:{name}"] = ("location", name, path)
    return facts


_C_TEMPLATE = re.compile(r"template[ \t\n]*<[^;{}()]*>[ \t\n]*")
_C_CLASS = re.compile(r"^[ \t]*(?:class|struct)[ \t]+(?:[A-Z_]+_(?:EXPORT|API)[ \t]+)?([A-Za-z_]\w+)[^;{()]*\{", re.M)
MAX_OVERLOADS = 3


def class_spans(code: str) -> list[tuple[int, int, str]]:
    """Where each class body lies, innermost last."""
    spans = []
    for m in _C_CLASS.finditer(code):
        depth, at = 1, m.end()
        while depth and at < len(code):
            depth += {"{": 1, "}": -1}.get(code[at], 0)
            at += 1
        spans.append((m.end(), at, m.group(1)))
    return spans


def extract_cpp_facts(path: str, source: str) -> dict[str, tuple[str, str, str]]:
    """As ``extract_c_facts``, and templates count, methods carry their class, and an overloaded name lists its signatures."""
    facts = {key: fact for key, fact in extract_c_facts(path, source).items() if not key.startswith("sig:")}
    code = _C_TEMPLATE.sub("", _C_COMMENT.sub("", source))
    spans = class_spans(code)
    found: dict[str, list[str]] = {}
    for m in _C_FUNCTION.finditer(code):
        name, head = m["name"], m["head"]
        if name in C_KEYWORDS or name.startswith("_") or name.isupper() or re.search(r"\b(?:static|return|else|new|delete|typedef|goto|case)\b", head):
            continue
        if path.endswith(C_SOURCES) and (m["end"] != "{" or m["lead"]):
            continue
        owners = [owner for start, end, owner in spans if start <= m.start() < end]
        qualified = f"{owners[-1]}::{name}" if owners else name
        found.setdefault(qualified, [])
        args = " ".join(m["args"].split())
        if args not in found[qualified]:
            found[qualified].append(args)
    for name, values in found.items():
        value = " | ".join(sorted(values))
        if len(values) <= MAX_OVERLOADS and len(value) <= 120:
            facts[f"sig:{name}"] = ("signature", name, value)
            facts[f"loc:{name}"] = ("location", name, path)
    return facts


def c_snapshot_facts(repo_dir: Path, sha: str, root: str, read=extract_c_facts) -> dict[str, tuple[str, str, str, str]]:
    """Headers first: a function declared in one and defined in a source file is one fact, and the header holds it."""
    listing = git(repo_dir, "ls-tree", "-r", "--name-only", sha).splitlines()
    inside = [f for f in listing if not root or f.startswith(root + "/")]
    if not root:
        inside = [f for f in inside if "/" not in f]
    headers = [f for f in inside if f.endswith(C_HEADERS)]
    sources = [f for f in inside if f.endswith(C_SOURCES) and "test" not in f.lower()]
    facts: dict[str, tuple[str, str, str, str]] = {}
    ambiguous: set[str] = set()
    for group in (headers, sources):
        seen: set[str] = set()
        for path in group:
            for key, (kind, name, value) in read(path, git(repo_dir, "show", f"{sha}:{path}")).items():
                if key in facts and group is sources and key not in seen:
                    continue
                if key in seen:
                    ambiguous.add(key)
                seen.add(key)
                facts[key] = (kind, name, value, path)
    for key in ambiguous:
        del facts[key]
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
    pinned = json.loads(MANIFEST.read_text()).get(repo, {}) if MANIFEST.exists() else {}
    head = pinned.get("head") or git(repo_dir, "rev-parse", "HEAD").strip()
    lines = git(repo_dir, "log", "--first-parent", "--format=%H %ct", head).splitlines()
    history = [(ln.split()[0], float(ln.split()[1])) for ln in lines][::-1]
    latest = history[-1][1]
    span = [h for h in history if h[1] >= latest - YEARS * 365.25 * 86400]
    step = (len(span) - 1) / (SNAPSHOTS - 1)
    picks = [span[round(i * step)] for i in range(SNAPSHOTS)]
    snapshots = []
    for sha, at in picks:
        if root is None:
            facts = {}
        elif split == "c-last":
            facts = c_snapshot_facts(repo_dir, sha, root, extract_cpp_facts)
        elif split in C_SPLITS:
            facts = c_snapshot_facts(repo_dir, sha, root)
        else:
            facts = snapshot_facts(repo_dir, sha, root)
        snapshots.append({"sha": sha, "at": at, "facts": facts})
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
        before_commit = {c["sha"]: commits[i - 1]["files"] if i else [] for i, c in enumerate(commits)}
        emitted = 0
        for commit in tasks:
            files = set(commit["files"])
            gold = {
                _commit_note(p) for p in past_commits
                if len(files & set(p["files"])) / len(files | set(p["files"])) >= 0.5 and p["subject"] != commit["subject"]
            }
            if gold:
                queries.append(Query(len(writes), f"{repo_name}-s{session}-t{emitted}", commit["subject"], commit["at"], "task", gold=gold,
                                     files=tuple(commit["files"]), previous_files=tuple(before_commit[commit["sha"]])))
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
    """The dev and test repositories by default; ``holdout=True`` or ``split`` picks another group."""
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
