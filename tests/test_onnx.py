"""The MiniLM embedder without torch: pooling, download, fallbacks and stores that change embedders."""

import hashlib
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from kanerva import embed, memory
from kanerva.embed import (
    HashingEmbedder,
    fetch_in_background,
    fetch_marker,
    fetch_minilm,
    mean_pool,
    minilm_folder,
    minilm_ready,
)
from kanerva.memory import Memory, open_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("KANERVA_HOME", str(tmp_path / "kanerva"))
    return tmp_path


class Spawned:
    def __init__(self, monkeypatch):
        self.calls = []
        monkeypatch.setattr(embed.subprocess, "Popen", lambda args, **kwargs: self.calls.append(args))


def test_pooling_ignores_padding_and_normalises():
    hidden = np.array([[[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]]], dtype=np.float32)
    assert mean_pool(hidden, np.array([[1, 1, 0]])).tolist() == [[1.0, 0.0]]


def test_pooling_keeps_a_row_of_zeros_finite():
    assert np.isfinite(mean_pool(np.zeros((1, 2, 4), dtype=np.float32), np.ones((1, 2), dtype=np.int64))).all()


def test_the_setting_is_off_by_default(store, monkeypatch):
    monkeypatch.delenv("KANERVA_EMBEDDER", raising=False)
    spawned = Spawned(monkeypatch)
    mem = open_store()
    assert mem.embedder is None and spawned.calls == []
    mem.close()


def test_a_missing_model_means_lexical_search_and_one_download(store, monkeypatch):
    monkeypatch.setenv("KANERVA_EMBEDDER", "minilm")
    spawned = Spawned(monkeypatch)
    for _ in range(3):
        mem = open_store()
        assert mem.embedder is None and mem.config.use_dense is False
        mem.close()
    assert len(spawned.calls) == 1
    assert spawned.calls[0][1:] == ["-m", "kanerva", "fetch-model"]


def test_a_failed_download_is_tried_again_after_a_while(store, monkeypatch):
    spawned = Spawned(monkeypatch)
    fetch_in_background()
    os.utime(fetch_marker(), (0, 0))
    fetch_in_background()
    assert len(spawned.calls) == 2


def test_a_download_in_progress_is_not_started_twice(store, monkeypatch):
    spawned = Spawned(monkeypatch)
    fetch_in_background()
    os.utime(fetch_marker(), (time.time() - 60, time.time() - 60))
    fetch_in_background()
    assert len(spawned.calls) == 1


def write_model(folder):
    for name in embed.MINILM_FILES:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def test_a_missing_package_means_lexical_search(store, monkeypatch):
    write_model(minilm_folder())
    monkeypatch.setenv("KANERVA_EMBEDDER", "minilm")

    def broken(*args, **kwargs):
        raise ImportError("no onnxruntime")

    monkeypatch.setattr(embed, "OnnxEmbedder", broken)
    mem = open_store()
    assert mem.embedder is None
    mem.close()


def test_a_ready_model_is_used(store, monkeypatch):
    write_model(minilm_folder())
    monkeypatch.setenv("KANERVA_EMBEDDER", "minilm")
    monkeypatch.setattr(embed, "OnnxEmbedder", lambda: HashingEmbedder(384))
    mem = open_store()
    assert mem.embedder.dim == 384 and mem.config.use_dense is True
    mem.close()


def fake_hub(monkeypatch, content):
    calls = []

    def download(repo, name, revision, local_dir):
        calls.append((repo, name, revision))
        path = Path(local_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content[name])
        return str(path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=download))
    return calls


def test_the_download_uses_the_official_repository_at_a_pinned_revision(tmp_path, monkeypatch):
    content = {name: name.encode() for name in embed.MINILM_FILES}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in content.items()}
    monkeypatch.setattr(embed, "MINILM_FILES", hashes)
    calls = fake_hub(monkeypatch, content)
    fetch_minilm(tmp_path)
    assert {c[0] for c in calls} == {"sentence-transformers/all-MiniLM-L6-v2"}
    assert {c[2] for c in calls} == {embed.MINILM_REVISION} and len(embed.MINILM_REVISION) == 40
    assert minilm_ready(tmp_path)


def test_a_file_with_the_wrong_hash_is_rejected_and_removed(tmp_path, monkeypatch):
    fake_hub(monkeypatch, {name: b"tampered" for name in embed.MINILM_FILES})
    with pytest.raises(ValueError):
        fetch_minilm(tmp_path)
    assert not (tmp_path / "onnx" / "model.onnx").exists()


def test_a_store_survives_a_change_of_embedder(tmp_path):
    path = tmp_path / "m.sqlite"
    old = Memory(path, embedder=HashingEmbedder(64))
    old.write("The deploy goes through the staging branch", scope="s", kind="semantic")
    old.write("The cache is flushed by the nightly job", scope="s", kind="semantic")
    old.close()

    new = Memory(path, embedder=HashingEmbedder(128))
    new.write("The deploy needs a signed tag", scope="s", kind="semantic")
    assert "signed tag" in new.retrieve("how does the deploy work?", 500, scope="s").render()
    assert new.reembed() == 2
    sizes = {len(r["embedding"]) for r in new.db.execute("SELECT embedding FROM items")}
    assert sizes == {512}
    assert "staging branch" in new.retrieve("how does the deploy work?", 500, scope="s").render()
    assert new.reembed() == 0
    new.close()


def test_a_store_with_mixed_vectors_works_without_an_embedder(tmp_path):
    path = tmp_path / "m.sqlite"
    for dim, text in ((64, "The deploy goes through staging"), (128, "The deploy needs a signed tag")):
        mem = Memory(path, embedder=HashingEmbedder(dim))
        mem.write(text, scope="s", kind="semantic")
        mem.close()
    plain = Memory(path)
    assert "deploy" in plain.retrieve("how does the deploy work?", 500, scope="s").render()
    plain.close()


def test_reembedding_without_an_embedder_does_nothing(tmp_path):
    mem = Memory(tmp_path / "m.sqlite")
    mem.write("A note", scope="s")
    assert mem.reembed() == 0
    mem.close()


model_folder = os.environ.get("KANERVA_TEST_MINILM")
needs_model = pytest.mark.skipif(not model_folder, reason="set KANERVA_TEST_MINILM to a folder with the model")


@needs_model
def test_vectors_match_the_cached_sentence_transformers_output():
    pytest.importorskip("onnxruntime")
    root = Path(__file__).resolve().parent.parent
    if not (root / "experiments" / "cache" / "repomem-new.npy").exists():
        pytest.skip("needs the benchmark caches")
    from experiments.datasets import digest
    from experiments.embed_cache import TextCache
    from experiments.registry import load

    embedder = embed.OnnxEmbedder(Path(model_folder))
    for name, data in (("repomem-new", "repomem-new"), ("locomo", "locomo")):
        cache = TextCache(name)
        texts = sorted({t.text for s in load(data)[:3] for t in s.writes if digest(t.text) in cache.index})
        sample = [texts[i] for i in np.random.default_rng(0).choice(len(texts), 300, replace=False)]
        reference = np.stack([cache.vector(t) for t in sample])
        reference /= np.linalg.norm(reference, axis=1, keepdims=True)
        assert (embedder.encode(sample) * reference).sum(axis=1).min() > 0.9999


@needs_model
def test_long_texts_are_truncated_as_the_model_expects():
    pytest.importorskip("onnxruntime")
    embedder = embed.OnnxEmbedder(Path(model_folder))
    words = "the quick brown fox jumps over the lazy dog " * 20
    assert np.allclose(embedder.encode([words * 40]), embedder.encode([words * 80]), atol=1e-5)


@needs_model
def test_the_dense_model_applies(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    seen = []
    real = memory.load_model
    monkeypatch.setattr(memory, "load_model", lambda name: seen.append(name) or real(name))
    mem = Memory(tmp_path / "m.sqlite", embedder=embed.OnnxEmbedder(Path(model_folder)))
    mem.write("Run the suite with make test-all from the repo root", scope="s", kind="semantic")
    assert "make test-all" in mem.retrieve("how do I run the tests?", 300, scope="s").render()
    assert set(seen) == {"dense"}
    mem.close()
