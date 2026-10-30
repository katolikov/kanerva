"""Prequential online learning on real streams, with feedback simulated from evidence labels.

Each seed shuffles the order of streams and of queries inside a stream and shares one policy across the
whole run. After each retrieval at budget B, every injected item gets reward 1 if it is gold and 0 otherwise,
flipped with probability ``noise``. Superseded items are skipped, as in the model of Theorem 2. The debiased
setting applies the unbiased correction for a known flip rate, (y - p) / (1 - 2p).
"""

from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.embed_cache import TextCache
from experiments.harness import RESULTS, apply_write, git_commit, new_memory
from experiments.registry import load
from kanerva import RetrievalConfig
from kanerva.bandit import LinearBandit
from kanerva.policy import BANDIT_THETA

SEEDS = (0, 1, 2, 3, 4)
BUDGET = 512
SETTINGS = {
    "static": {"learn": False, "beta": 0.0, "noise": 0.0},
    "online": {"learn": True, "beta": 0.0, "noise": 0.0},
    "online-ucb": {"learn": True, "beta": 0.05, "noise": 0.0},
    "online-noisy": {"learn": True, "beta": 0.0, "noise": 0.2},
    "online-debiased": {"learn": True, "beta": 0.0, "noise": 0.2, "debias": True},
}


def replay(args: tuple) -> dict:
    dataset, keyed, setting, seed, ridge = args
    spec = SETTINGS[setting]
    rng = random.Random(seed)
    streams = [s for s in load(dataset) if s.split == "test"]
    if dataset == "repomem":
        streams = [s for s in streams if s.name.endswith(f"-{seed % 3}")]
    rng.shuffle(streams)
    cfg = RetrievalConfig(bandit_ridge=ridge)
    bandit = LinearBandit(np.asarray(BANDIT_THETA, dtype=float), ridge, spec["beta"])
    cache = TextCache(dataset)
    hits: list[float] = []
    tokens: list[float] = []
    for stream in streams:
        mem = new_memory(cache, cfg)
        mem.bandit = bandit
        id_of: dict[str, int] = {}
        cursor = 0
        queries = list(stream.queries)
        if dataset == "locomo":
            rng.shuffle(queries)
        for query in queries:
            while cursor < query.pos:
                apply_write(mem, cache, stream.writes[cursor], id_of, keyed)
                cursor += 1
            gold = {id_of[t] for t in query.gold if t in id_of}
            if not gold:
                continue
            result = mem.retrieve(query.question, BUDGET, now=query.now, query_vector=cache.vector(query.question))
            chosen = {i.id for i in result.items}
            hits.append(len(chosen & gold) / len(gold))
            tokens.append(result.tokens)
            if spec["learn"] and result.retrieval_id is not None:
                rewards = {}
                for item in chosen:
                    good = item in gold
                    if rng.random() < spec["noise"]:
                        good = not good
                    reward = 1.0 if good else 0.0
                    if spec.get("debias"):
                        reward = (reward - spec["noise"]) / (1.0 - 2.0 * spec["noise"])
                    rewards[item] = reward
                mem.feedback(result.retrieval_id, rewards)
        mem.close()
    return {"dataset": dataset, "setting": setting, "seed": seed, "recall": hits, "tokens": tokens, "theta": bandit.theta.tolist()}


def main() -> None:
    start = time.time()
    runs = [("lme", False), ("locomo", False), ("repomem", True)]
    only = set(sys.argv[1:])
    jobs = [(d, k, s, seed, 4.0) for d, k in runs if not only or d in only for s in SETTINGS for seed in SEEDS]
    with ProcessPoolExecutor(8) as pool:
        results = list(pool.map(replay, jobs))
    (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
    (RESULTS / "raw" / "online.json").write_text(json.dumps(results))
    summary: dict = {}
    for r in results:
        n = len(r["recall"])
        tail = r["recall"][n // 2:]
        entry = summary.setdefault(r["dataset"], {}).setdefault(r["setting"], {"all": [], "second_half": []})
        entry["all"].append(float(np.mean(r["recall"])))
        entry["second_half"].append(float(np.mean(tail)))
    out = {"meta": {"experiment": "online", "budget": BUDGET, "seeds": SEEDS, "commit": git_commit(),
                    "wall_seconds": round(time.time() - start, 1)}, "summary": summary}
    (RESULTS / "online.json").write_text(json.dumps(out, indent=1))
    for d, settings in summary.items():
        for s, e in settings.items():
            print(d, s, f"all={np.mean(e['all']):.4f}", f"second_half={np.mean(e['second_half']):.4f}")


if __name__ == "__main__":
    main()
