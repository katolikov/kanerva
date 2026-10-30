"""Content-addressed cache of embeddings and token counts for all benchmark texts."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments.datasets import ROOT, digest

CACHE = ROOT / "experiments" / "cache"


class TextCache:
    """Maps sha1(text) to a row of a dense matrix and a token count."""

    def __init__(self, name: str):
        self.name = name
        self.vectors = np.load(CACHE / f"{name}.npy", mmap_mode="r")
        meta = json.loads((CACHE / f"{name}.json").read_text())
        self.index: dict[str, int] = meta["index"]
        self.tokens: dict[str, int] = meta["tokens"]
        self._counter = None

    def vector(self, text: str) -> np.ndarray:
        return np.asarray(self.vectors[self.index[digest(text)]])

    def count(self, text: str) -> int:
        """Token count; texts altered on write (stripped, redacted) are counted on the fly."""
        known = self.tokens.get(digest(text))
        if known is not None:
            return known
        if self._counter is None:
            from kanerva.text import tiktoken_counter

            self._counter = tiktoken_counter()
        return self._counter(text)


def build(name: str, texts: list[str], model: str) -> None:
    from kanerva.embed import SentenceTransformerEmbedder
    from kanerva.text import tiktoken_counter

    CACHE.mkdir(parents=True, exist_ok=True)
    start = time.time()
    count = tiktoken_counter()
    tokens = {digest(t): count(t) for t in texts}
    print(f"{name}: {len(texts)} texts, tokens counted in {time.time() - start:.0f}s", flush=True)
    embedder = SentenceTransformerEmbedder(model)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), embedder.dim), dtype=np.float32)
    for start_i in range(0, len(order), 4096):
        chunk = order[start_i:start_i + 4096]
        out[chunk] = embedder.encode([texts[i] for i in chunk])
        print(f"  {start_i + len(chunk)}/{len(texts)} {time.time() - start:.0f}s", flush=True)
    np.save(CACHE / f"{name}.npy", out)
    (CACHE / f"{name}.json").write_text(
        json.dumps({"model": model, "index": {digest(t): i for i, t in enumerate(texts)}, "tokens": tokens})
    )


if __name__ == "__main__":
    from experiments.datasets import all_texts
    from experiments.registry import load

    which = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "sentence-transformers/all-MiniLM-L6-v2"
    build(which, all_texts(load(which)), model)
