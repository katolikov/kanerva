"""Cross-encoder logits: computed on demand or read back from experiments/cache/ce_<name>.json."""

from __future__ import annotations

import json
import os

from experiments.datasets import digest
from experiments.embed_cache import CACHE

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def pair_key(question: str, text: str) -> str:
    return digest(question + "\x1f" + text)


class CachedScores:
    """Stands in for a cross-encoder with stored logits. Unknown pairs score NaN."""

    def __init__(self, dataset: str):
        self.scores: dict[str, float] = json.loads((CACHE / f"ce_{dataset}.json").read_text())

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        return [self.scores.get(pair_key(query, t), float("nan")) for t in texts]


class LiveCrossEncoder:
    """Computes the logits it does not have yet, and saves them to the cache file."""

    def __init__(self, name: str):
        self.path = CACHE / f"ce_{name}.json"
        self.scores: dict[str, float] = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.model = None
        self.dirty = False

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        missing = [t for t in texts if pair_key(query, t) not in self.scores]
        if missing:
            if self.model is None:
                from sentence_transformers import CrossEncoder

                self.model = CrossEncoder(MODEL)
            values = self.model.predict([(query, t[:1500]) for t in missing], batch_size=32, show_progress_bar=False)
            for t, v in zip(missing, values):
                self.scores[pair_key(query, t)] = round(float(v), 4)
            self.dirty = True
        return [self.scores[pair_key(query, t)] for t in texts]

    def save(self) -> None:
        """Merge into the file on disk, since parallel workers may have added other pairs meanwhile."""
        if self.dirty:
            merged = json.loads(self.path.read_text()) if self.path.exists() else {}
            merged.update(self.scores)
            tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(merged))
            tmp.replace(self.path)
