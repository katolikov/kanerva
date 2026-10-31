"""Notes about the files an agent is working on, on RepoMem's task track.

The prompt alone says little about a change, while the agent's open files say a lot. Two simulated sessions: ``one`` has
just opened one file of the change, ``previous`` has just worked on the files of the commit before it. The first is the
generous case, since task labels come from file overlap. Compared at equal tokens against spending the same budget on the
prompt, and against adding file names to the query. Writes results/file_recall.json.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.embed_cache import TextCache
from experiments.harness import RESULTS, apply_write, git_commit, new_memory
from experiments.methods import LEXICAL
from experiments.registry import load
from kanerva import RetrievalConfig

PROMPT_BUDGET = 512
FILE_BUDGET = 150
MAX_FILES = 3
SETS = ("repomem", "repomem-holdout", "repomem-fresh", "repomem-new", "repomem-other")
HELD_OUT = ("repomem-final",)
C_FAMILY = ("repomem-c-dev", "repomem-c-test")


def configs() -> dict[str, RetrievalConfig]:
    base = RetrievalConfig().with_(use_bandit=False)
    return {"full": base, "lexical": base.with_(**LEXICAL)}


def picked(query, session: str, rng: random.Random) -> list[str]:
    if session == "one":
        return [rng.choice(sorted(query.files))]
    return sorted(query.previous_files)[:MAX_FILES]


def run_stream(args: tuple) -> list[dict]:
    cache_name, stream = args
    cache = TextCache(cache_name)
    cfgs = configs()
    mem = new_memory(cache, cfgs["full"])
    id_of: dict[str, int] = {}
    rng = random.Random(stream.name)
    rows, cursor = [], 0
    for query in stream.queries:
        while cursor < query.pos:
            apply_write(mem, cache, stream.writes[cursor], id_of, False)
            cursor += 1
        gold = {id_of[t] for t in query.gold if t in id_of}
        if query.category != "task" or not gold:
            continue
        qvec = cache.vector(query.question)
        repo = stream.name.rsplit("-", 1)[0]
        for session in ("one", "previous"):
            files = picked(query, session, rng)
            if not files:
                continue
            found: dict[int, int] = {}
            for path in files:
                for item in mem.about_file(path, FILE_BUDGET, None, found):
                    found[item.id] = item.tokens + mem.config.item_overhead
            spent = sum(found.values())
            for name, cfg in cfgs.items():
                prepared = mem.prepare(query.question, query.now, None, cfg, qvec)
                base = {item.id for item in mem.select(prepared, PROMPT_BUDGET, record=False).items}
                wider = {item.id for item in mem.select(prepared, PROMPT_BUDGET + spent, record=False).items}
                grown: set[int] = set()
                if name == "lexical":  # the embedding cache has no vectors for expanded queries
                    expanded = mem.prepare(query.question + " " + " ".join(p.rsplit("/", 1)[-1] for p in files), query.now, None, cfg)
                    grown = {item.id for item in mem.select(expanded, PROMPT_BUDGET, record=False).items}
                rows.append({
                    "repo": repo, "session": session, "config": name, "file_tokens": spent, "file_notes": len(found),
                    "file_precision": len(set(found) & gold) / len(found) if found else None,
                    "prompt": len(gold & base) / len(gold), "wider": len(gold & wider) / len(gold),
                    "with_files": len(gold & (base | set(found))) / len(gold), "expanded": len(gold & grown) / len(gold),
                })
    mem.close()
    return rows


def run_set(name: str) -> list[dict]:
    jobs = [(name, s) for s in load(name)]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for part in pool.map(run_stream, jobs):
            rows.extend(part)
    return rows


def summary(rows: list[dict]) -> dict:
    out: dict = {"queries": len(rows)}
    for key in ("prompt", "wider", "with_files", "expanded", "file_tokens", "file_notes"):
        out[key] = round(float(np.mean([r[key] for r in rows])), 3)
    precision = [r["file_precision"] for r in rows if r["file_precision"] is not None]
    out["file_precision"] = round(float(np.mean(precision)), 3) if precision else None
    by_repo = defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r["with_files"] - r["wider"])
    diffs = [float(np.mean(v)) for v in by_repo.values()]
    out["with_files_vs_wider"] = {"mean": round(float(np.mean(diffs)), 4), "up": sum(d > 0 for d in diffs), "down": sum(d < 0 for d in diffs),
                                  "repositories": len(diffs)}
    return out


def main(names: tuple[str, ...]) -> None:
    report: dict = {"meta": {"experiment": "file recall", "commit": git_commit(), "prompt_budget": PROMPT_BUDGET, "file_budget": FILE_BUDGET}}
    groups = {name: run_set(name) for name in names}
    for name, rows in groups.items():
        report[name] = {}
        for session in ("one", "previous"):
            for config in ("lexical", "full"):
                now = [r for r in rows if r["session"] == session and r["config"] == config]
                if now:
                    report[name][f"{session}/{config}"] = summary(now)
                    print(name, session, config, report[name][f"{session}/{config}"], flush=True)
    out = "file_recall_final.json" if names == HELD_OUT else "file_recall_c.json" if set(names) <= set(C_FAMILY) else "file_recall.json"
    (RESULTS / out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main(tuple(sys.argv[1:]) or SETS))
