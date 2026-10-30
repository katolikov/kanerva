import json

import numpy as np

from kanerva import Memory, RetrievalConfig
from kanerva.embed import HashingEmbedder
from kanerva.relevance import FEATURES, ItemView, TreeModel, feature_matrix, load_model
from kanerva.select import SelectionProblem, budgeted_select
from kanerva.text import query_window


def _matrix(query, texts, **kw):
    n = len(texts)
    items = [ItemView(t, max(len(t.split()), 1), float(i), "s0") for i, t in enumerate(texts)]
    ranks = np.arange(n, dtype=float)
    zeros = np.zeros(n)
    return feature_matrix(query, items, ranks, ranks, zeros + 1, zeros + 0.5, zeros, zeros, zeros, zeros, zeros, **kw)


def test_feature_matrix_columns_and_query_item_signals():
    x = _matrix("When did Alice move to Paris in 2021?", ["user: Alice moved to Paris in 2021", "assistant: ok"])
    assert x.shape == (2, len(FEATURES))
    assert x[0, FEATURES.index("entity_overlap")] == 1.0
    assert x[0, FEATURES.index("number_overlap")] == 1.0
    assert x[0, FEATURES.index("when_date")] == 1.0
    assert x[1, FEATURES.index("entity_overlap")] == 0.0
    assert x[0, FEATURES.index("role_user")] == 1.0
    assert x[1, FEATURES.index("role_assistant")] == 1.0


def test_speaker_in_query_ignores_generic_roles():
    x = _matrix("What did Caroline say about the user?", ["Caroline: hello", "user: hi"])
    assert x[0, FEATURES.index("speaker_in_query")] == 1.0
    assert x[1, FEATURES.index("speaker_in_query")] == 0.0


def test_tree_model_round_trip_routes_rows_to_leaves():
    use = tuple(FEATURES[:2])
    model = TreeModel(
        feature=np.array([[1, 0, 0]]), threshold=np.array([[0.5, np.inf, np.inf]]), left=np.array([[1, 1, 2]]),
        right=np.array([[2, 1, 2]]), value=np.array([[0.0, -2.0, 2.0]]), baseline=0.0, depth=2, use=use,
    )
    x = np.zeros((2, len(FEATURES)))
    x[1, 1] = 1.0
    p = TreeModel.from_json(json.loads(json.dumps(model.to_json()))).predict(x)
    assert p[0] < 0.5 < p[1]
    assert abs(p[0] + p[1] - 1.0) < 1e-9


def test_shipped_models_are_probabilities():
    for name in ("dense", "lexical", "rerank", "static"):
        model = load_model(name)
        assert model is not None
        p = model.predict(_matrix("what is the port?", ["user: the port is 8080", "assistant: noted"]))
        assert ((p >= 0) & (p <= 1)).all()


def test_neighbours_join_the_pool_within_session_and_scope():
    emb = HashingEmbedder()
    cfg = RetrievalConfig(neighbour_top=5, use_bandit=False)
    mem = Memory(embedder=emb, config=cfg)
    a = mem.write("deployment uses blue green rollout", session="s1", scope="p")
    b = mem.write("sounds fine thanks", session="s1", scope="p")
    c = mem.write("sounds fine thanks again", session="s2", scope="p")
    d = mem.write("thanks for the note", session="s1", scope="q")
    qv = emb.encode(["blue green deployment"])[0]
    prepared = mem.prepare("blue green deployment", 1e9, "p", cfg, qv)
    assert a in prepared.candidates.ids
    assert b in prepared.candidates.ids
    assert c not in prepared.candidates.ids or c != a + 1
    assert d not in prepared.candidates.ids


def test_context_signals_use_same_session_neighbours_only():
    emb = HashingEmbedder()
    mem = Memory(embedder=emb, config=RetrievalConfig(use_bandit=False))
    one = mem.write("blue green deployment", session="s1")
    two = mem.write("unrelated words here", session="s1")
    three = mem.write("unrelated words there", session="s2")
    qv = emb.encode(["blue green deployment"])[0]
    near, _, session = mem.context_signals([two, three, one], qv, None)
    assert near[0] > 0.9 and near[1] == 0.0
    assert session[0] > 0.9 and session[1] < 0.5


def test_session_column_is_added_to_old_databases(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    mem = Memory(path)
    mem.db.execute("ALTER TABLE items DROP COLUMN session")
    mem.close()
    db = sqlite3.connect(path)
    assert "session" not in {r[1] for r in db.execute("PRAGMA table_info(items)")}
    db.close()
    mem = Memory(path)
    assert mem.write("still works", session="s") is not None
    mem.close()


def test_query_window_prefers_the_sentences_with_query_terms():
    text = "Nothing here. Filler words again. The port is 8080 for staging. More filler follows."
    excerpt, kept, total = query_window(text, ["port", "stag"], [0.5, 0.5], 12, lambda s: len(s.split()))
    assert "8080" in excerpt and kept > 0 and total >= kept


def test_version_groups_allow_one_version_of_an_item():
    p = SelectionProblem(
        weights=np.array([0.9, 0.5, 0.45]), costs=np.array([10.0, 10.0, 2.0]), affinity=np.zeros((3, 1)),
        facet_weights=np.zeros(1), modular=1.0, groups=np.array([0, 1, 1]),
    )
    chosen = budgeted_select(p, 100.0, (1.0,))
    assert sorted(chosen) in ([0, 1], [0, 2])


def test_excerpts_stay_within_budget_and_mark_truncation():
    emb = HashingEmbedder()
    cfg = RetrievalConfig(excerpt_threshold=20, excerpt_tokens=12, use_bandit=False, item_overhead=0, model="dense")
    mem = Memory(embedder=emb, config=cfg, count_tokens=lambda s: len(s.split()))
    long_text = " ".join(f"Sentence number {i} says nothing." for i in range(20)) + " The port is 8080."
    mem.write(long_text)
    qv = emb.encode(["what is the port"])[0]
    result = mem.retrieve("what is the port", 15, query_vector=qv, record=False)
    assert result.items and result.items[0].truncated
    assert "8080" in result.items[0].text
    assert result.tokens <= 15


def test_lexical_memory_uses_the_lexical_model():
    mem = Memory()
    mem.write("the staging port is 8080")
    mem.write("lunch was good")
    result = mem.retrieve("which port does staging use", 100, record=False)
    assert result.items and "8080" in result.items[0].text or any("8080" in i.text for i in result.items)
