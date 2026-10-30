"""Embedding back-ends. All return L2-normalised float32 matrices."""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Signed feature hashing of unigrams and bigrams; a zero-dependency local fallback."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _index(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            for feature in tokens + [a + " " + b for a, b in zip(tokens, tokens[1:])]:
                i, sign = self._index(feature)
                out[row, i] += sign
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm == 0, 1.0, norm)


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model, device=device)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(
            list(texts), batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)


class Model2VecEmbedder:
    def __init__(self, model: str = "minishlab/potion-base-8M"):
        from model2vec import StaticModel

        self._model = StaticModel.from_pretrained(model)
        self.dim = self._model.dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.asarray(self._model.encode(list(texts)), dtype=np.float32)
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm == 0, 1.0, norm)
