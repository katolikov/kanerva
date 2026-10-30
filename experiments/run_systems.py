"""E9: Mem0's open-source library as a real-system baseline, in its no-LLM mode (infer=False).

Runs in its own environment: .venv-systems/bin/python -m experiments.run_systems. Mem0 stores turns verbatim,
retrieves with its own scoring over local Qdrant, and its ranking is then filled to the budget with the same
token counts as every other method. The comparison uses the same questions for both.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
import time

os.environ.setdefault("OPENAI_API_KEY", "unused-no-llm-calls")
os.environ["MEM0_TELEMETRY"] = "False"

from experiments.datasets import ROOT, load_longmemeval  # noqa: E402

RESULTS = ROOT / "results"
BUDGETS = (128, 256, 512, 1024, 2048, 4096)
QUESTIONS = int(os.environ.get("MEM0_QUESTIONS", "40"))
OVERHEAD = 8


def main() -> None:
    import tiktoken
    from mem0 import Memory

    enc = tiktoken.get_encoding("cl100k_base")
    count = lambda t: len(enc.encode(t, disallowed_special=()))
    streams = [s for s in load_longmemeval() if s.split == "test"]
    streams = random.Random(0).sample(streams, min(QUESTIONS, len(streams)))
    rows: list[dict] = []
    start = time.time()
    add_seconds = search_seconds = 0.0
    for index, stream in enumerate(streams):
        folder = tempfile.mkdtemp()
        memory = Memory.from_config({
            "vector_store": {"provider": "qdrant", "config": {"path": folder + "/q", "embedding_model_dims": 384, "collection_name": "m", "on_disk": False}},
            "embedder": {"provider": "huggingface", "config": {"model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dims": 384}},
            "history_db_path": folder + "/h.db",
        })
        t0 = time.time()
        for turn in stream.writes:
            memory.add(turn.text[:8000], user_id="u", infer=False)
        add_seconds += time.time() - t0
        query = stream.queries[0]
        t0 = time.time()
        found = memory.search(query.question, filters={"user_id": "u"}, top_k=100)["results"]
        search_seconds += time.time() - t0
        order = [r["memory"] for r in found]
        gold = {t.strip()[:8000] for t in query.gold}
        for budget in BUDGETS:
            spent, chosen = 0, []
            for text in order:
                cost = count(text) + OVERHEAD
                if spent + cost <= budget:
                    chosen.append(text)
                    spent += cost
            rows.append({"qid": query.qid, "budget": budget, "recall": len(set(chosen) & gold) / len(gold), "tokens": spent})
        del memory
        shutil.rmtree(folder, ignore_errors=True)
        print(index + 1, f"{time.time() - start:.0f}s", flush=True)
    out = {
        "meta": {"experiment": "systems", "system": "mem0ai 2.2.1 infer=False, qdrant local, all-MiniLM-L6-v2", "questions": QUESTIONS,
                 "wall_seconds": round(time.time() - start, 1), "add_ms_per_turn": add_seconds * 1000 / sum(len(s.writes) for s in streams),
                 "search_ms": search_seconds * 1000 / len(streams)},
        "rows": rows,
    }
    (RESULTS / "systems_mem0.json").write_text(json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
