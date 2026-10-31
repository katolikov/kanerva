"""Embedding back-ends. All return L2-normalised float32 matrices."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .paths import home
from .text import tokenize

MINILM_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MINILM_FILES = {
    "onnx/model.onnx": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
}
MINILM_MAX_TOKENS = 256  # the model's max_seq_length; tokenizer.json itself says 128
MULTILINGUAL_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MULTILINGUAL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MULTILINGUAL_MODEL = "onnx/model_qint8_arm64.onnx"  # the same bytes as the repository's avx512 files
MULTILINGUAL_FILES = {
    MULTILINGUAL_MODEL: "783fea82d71a58179b830a4dbd2d58447e640609e98eedf9ffa12622d375a672",
    "tokenizer.json": "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8",
}
MULTILINGUAL_MAX_TOKENS = 128  # the model's max_seq_length
FETCH_STALE_SECONDS = 3600


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


def minilm_folder() -> Path:
    return home() / "models" / "all-MiniLM-L6-v2"


def fetch_marker() -> Path:
    return minilm_folder().parent / "fetching"


def minilm_ready(folder: Path | None = None) -> bool:
    folder = folder or minilm_folder()
    return all((folder / name).is_file() for name in MINILM_FILES)


def multilingual_folder() -> Path:
    return home() / "models" / "paraphrase-multilingual-MiniLM-L12-v2"


def multilingual_ready(folder: Path | None = None) -> bool:
    folder = folder or multilingual_folder()
    return all((folder / name).is_file() for name in MULTILINGUAL_FILES)


def _download(repo: str, revision: str, files: dict[str, str], folder: Path) -> Path:
    """Download ``files`` from the official repository at a pinned revision, and check their hashes."""
    from huggingface_hub import hf_hub_download

    for name, expected in files.items():
        path = Path(hf_hub_download(repo, name, revision=revision, local_dir=folder))
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            path.unlink()
            raise ValueError(f"{name} does not match the expected hash")
    return folder


def fetch_multilingual(folder: Path | None = None) -> Path:
    return _download(MULTILINGUAL_REPO, MULTILINGUAL_REVISION, MULTILINGUAL_FILES, folder or multilingual_folder())


def fetch_minilm(folder: Path | None = None) -> Path:
    """What ``kanerva fetch-model`` runs: this model, or the multilingual one if ``KANERVA_EMBEDDER`` names it."""
    if folder is None and os.environ.get("KANERVA_EMBEDDER") == "multilingual":
        return fetch_multilingual()
    return _download(MINILM_REPO, MINILM_REVISION, MINILM_FILES, folder or minilm_folder())


def _fetch_command() -> list[str] | None:
    """How to run the download. A plugin starts without ONNX Runtime until the model is here, so uv brings it along.

    An installed package without the ``onnx`` extra has no such way, and stays lexical.
    """
    fetch = ["-m", "kanerva", "fetch-model"]
    if importlib.util.find_spec("onnxruntime") is not None:
        return [sys.executable, *fetch]
    uv = shutil.which("uv")
    if uv and Path(__file__).resolve().parents[1].name == "lib":
        return [uv, "run", "--quiet", "--no-project", "--with", "numpy>=1.26", "--with", "onnxruntime>=1.18",
                "--with", "tokenizers>=0.19", "python", *fetch]
    return None


def fetch_in_background() -> None:
    """Start one download that outlives this process, since a hook has seconds to live and a model is 90 MB or more.

    The marker stays behind a failed download, so that an offline machine tries once an hour, not on every prompt.
    """
    marker = fetch_marker()
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists() and time.time() - marker.stat().st_mtime < FETCH_STALE_SECONDS:
        return
    command = _fetch_command()
    if command is None:
        return
    marker.touch()
    subprocess.Popen(command, start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average the token vectors the attention mask marks as real, then L2-normalise."""
    weights = mask[..., None].astype(np.float32)
    pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
    return pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)


class OnnxEmbedder:
    """all-MiniLM-L6-v2 on onnxruntime: the vectors of ``SentenceTransformerEmbedder`` without torch."""

    dim = 384
    model_file = "onnx/model.onnx"
    max_tokens = MINILM_MAX_TOKENS
    padding: dict = {}

    def __init__(self, folder: Path | None = None, batch_size: int = 32):
        import onnxruntime
        from tokenizers import Tokenizer

        folder = self._local(folder)
        self.batch_size = batch_size
        self._tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=self.max_tokens)
        self._tokenizer.enable_padding(**self.padding)
        self._session = onnxruntime.InferenceSession(str(folder / self.model_file), providers=["CPUExecutionProvider"])

    def _local(self, folder: Path | None) -> Path:
        folder = folder or minilm_folder()
        if not minilm_ready(folder):
            fetch_minilm(folder)
        return folder

    def _batch(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self._tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        return mean_pool(self._session.run(None, feed)[0], mask)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        order = np.argsort([len(text) for text in texts], kind="stable")  # similar lengths need less padding
        for start in range(0, len(order), self.batch_size):
            chunk = order[start:start + self.batch_size]
            out[chunk] = self._batch([texts[i] for i in chunk])
        return out


class MultilingualEmbedder(OnnxEmbedder):
    """paraphrase-multilingual-MiniLM-L12-v2 on onnxruntime, so that a query finds notes in another language.

    The model has 384 dimensions, as all-MiniLM-L6-v2 does, and a store tells embedders apart by the length of their
    vectors. One zero column makes these 385 long, so the two kinds never mix and ``reembed`` replaces the other kind.

    The quantised model scales each tensor by its own range, so a text's vector would depend on the texts batched and
    padded with it. Batches of one keep it a function of the text, and need no padding work.
    """

    dim = 385
    model_file = MULTILINGUAL_MODEL
    max_tokens = MULTILINGUAL_MAX_TOKENS
    padding = {"pad_id": 1, "pad_token": "<pad>"}

    def __init__(self, folder: Path | None = None, batch_size: int = 1):
        super().__init__(folder, batch_size)

    def _local(self, folder: Path | None) -> Path:
        folder = folder or multilingual_folder()
        if not multilingual_ready(folder):
            fetch_multilingual(folder)
        return folder

    def _batch(self, texts: Sequence[str]) -> np.ndarray:
        vectors = super()._batch(texts)
        return np.hstack([vectors, np.zeros((len(vectors), 1), dtype=np.float32)])


class Model2VecEmbedder:
    def __init__(self, model: str = "minishlab/potion-base-8M"):
        from model2vec import StaticModel

        self._model = StaticModel.from_pretrained(model)
        self.dim = self._model.dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.asarray(self._model.encode(list(texts)), dtype=np.float32)
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm == 0, 1.0, norm)


class CrossEncoderReranker:
    """Optional local reranker: a callable mapping (query, texts) to relevance logits."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str | None = None):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model, device=device)

    def __call__(self, query: str, texts: Sequence[str]) -> np.ndarray:
        pairs = [(query, text[:1500]) for text in texts]
        return np.asarray(self._model.predict(pairs, batch_size=64, show_progress_bar=False), dtype=float)
