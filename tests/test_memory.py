import numpy as np
import pytest

from kanerva import Memory, RetrievalConfig
from kanerva.consolidate import consolidate, expand, extract_summary, forget
from kanerva.embed import HashingEmbedder
from kanerva.memory import DAY

T0 = 1_700_000_000.0


@pytest.fixture
def mem():
    m = Memory(embedder=HashingEmbedder())
    yield m
    m.close()


def test_retrieve_respects_the_budget(mem):
    for i in range(50):
        mem.write(f"fact number {i} about the deployment pipeline and staging", at=T0 + i)
    result = mem.retrieve("deployment pipeline", 60, now=T0 + 100)
    assert 0 < result.tokens <= 60
    assert result.tokens == sum(i.tokens + mem.config.item_overhead for i in result.items)


def test_identical_writes_are_deduplicated_and_reconfirm(mem):
    a = mem.write("Tests run with make test", at=T0)
    b = mem.write("tests   run with make TEST", at=T0 + 10 * DAY)
    assert a == b
    assert mem.stats()["active"] == 1
    row = mem.db.execute("SELECT confirmed_at FROM items WHERE id=?", (a,)).fetchone()
    assert row["confirmed_at"] == T0 + 10 * DAY


def test_keyed_write_supersedes_and_keeps_history(mem):
    old = mem.write("The API timeout is 30 seconds", key="config:timeout", at=T0)
    new = mem.write("The API timeout is 60 seconds", key="config:timeout", at=T0 + DAY)
    result = mem.retrieve("what is the API timeout?", 40, now=T0 + 2 * DAY)
    assert result.items[0].id == new
    assert mem.get(old).superseded and not mem.get(new).superseded


def test_historical_query_reaches_old_versions(mem):
    mem.write("The API timeout is 30 seconds", key="config:timeout", at=T0)
    new = mem.write("The API timeout is 60 seconds", key="config:timeout", at=T0 + DAY)
    live = mem.retrieve("what is the API timeout?", 200, now=T0 + 2 * DAY)
    past = mem.retrieve("what was the API timeout previously?", 200, now=T0 + 2 * DAY)
    assert live.items[0].id == new
    assert any(i.superseded and i.weight > 0.01 for i in past.items)


def test_a_value_change_back_creates_a_new_live_version(mem):
    mem.write("Mode is fast", key="config:mode", at=T0)
    mem.write("Mode is safe", key="config:mode", at=T0 + DAY)
    third = mem.write("Mode is fast", key="config:mode", at=T0 + 2 * DAY)
    assert not mem.get(third).superseded
    assert mem.stats()["superseded"] == 2


def test_secrets_never_reach_the_store(mem):
    mem.write("deploy with token ghp_abcdefghijklmnopqrstuvwxyz0123456789 today", at=T0)
    rows = mem.db.execute("SELECT text, source FROM items").fetchall()
    assert "ghp_" not in rows[0]["text"]
    assert "github-token" in rows[0]["source"]
    assert not mem.db.execute("SELECT * FROM items_fts WHERE items_fts MATCH 'ghp_abcdefghijklmnopqrstuvwxyz0123456789'").fetchall()


def test_scopes_are_isolated_but_global_is_shared(mem):
    mem.write("alpha uses Postgres", scope="project:a", at=T0)
    mem.write("beta uses Postgres too", scope="project:b", at=T0)
    mem.write("everyone prefers Postgres", scope="global", at=T0)
    texts = [i.text for i in mem.retrieve("Postgres", 200, scope="project:a", now=T0 + 1).items]
    assert "alpha uses Postgres" in texts and "everyone prefers Postgres" in texts
    assert "beta uses Postgres too" not in texts


def test_feedback_updates_policy_and_item_posterior(mem):
    good = mem.write("Use uv to install dependencies", at=T0)
    mem.write("Use pip to install dependencies", at=T0)
    result = mem.retrieve("how to install dependencies", 100, now=T0 + 1)
    before = mem.bandit.theta.copy()
    assert mem.feedback(result.retrieval_id, {good: 1.0}) == len(result.items)
    assert not np.allclose(before, mem.bandit.theta)
    row = mem.db.execute("SELECT used, unused FROM items WHERE id=?", (good,)).fetchone()
    assert row["used"] == 2 and row["unused"] == 1
    assert mem.feedback(result.retrieval_id, {good: 1.0}) == 0


def test_feedback_skips_items_known_to_be_invalid(mem):
    old = mem.write("Build with make", key="cmd:build", at=T0)
    result = mem.retrieve("how do I build the project?", 100, now=T0 + 1)
    mem.write("Build with ninja", key="cmd:build", at=T0 + 2)
    updates = mem.bandit.updates
    mem.feedback(result.retrieval_id, {old: 0.0})
    assert mem.bandit.updates == updates


def test_policy_state_persists(tmp_path):
    path = tmp_path / "m.sqlite"
    mem = Memory(path)
    item = mem.write("Persist this", at=T0)
    result = mem.retrieve("persist", 100, now=T0 + 1)
    mem.feedback(result.retrieval_id, {item: 1.0})
    theta = mem.bandit.theta.copy()
    mem.close()
    again = Memory(path)
    assert np.allclose(again.bandit.theta, theta)
    assert again.stats()["active"] == 1
    again.close()


def test_hazard_learned_per_class(mem):
    for i in range(30):
        mem.write(f"value {i}", key="volatile:x", at=T0 + i * DAY)
    for i in range(30):
        mem.write(f"stable fact {i}", key=f"stable:{i}", at=T0)
    hazards = mem.refresh_hazards(T0 + 30 * DAY)
    assert hazards["volatile"].survival(30) < hazards["stable"].survival(30)


def test_edit_updates_search(mem):
    item = mem.write("Old wording about kittens", at=T0)
    assert mem.edit(item, "New wording about puppies")
    assert [i.id for i, _ in [(x, 0) for x in mem.retrieve("puppies", 100, now=T0 + 1).items]] == [item]
    assert not mem.lexical_search("kittens", 5)


def test_delete_removes_everything(mem):
    item = mem.write("Temporary note", at=T0)
    assert mem.delete([item]) == 1
    assert mem.get(item) is None
    assert not mem.lexical_search("temporary", 5)


def test_archive_hides_from_retrieval_but_keeps_data(mem):
    item = mem.write("Archived knowledge", at=T0)
    mem.archive([item])
    assert not mem.retrieve("archived knowledge", 100, now=T0 + 1).items
    assert mem.get(item) is not None


def test_empty_store_and_empty_query():
    mem = Memory()
    assert mem.retrieve("anything", 100, now=T0).items == []
    mem.write("something", at=T0)
    assert mem.retrieve("???", 100, now=T0 + 1).items == []
    mem.close()


def test_certificate_upper_bounds_the_objective(mem):
    for i in range(30):
        mem.write(f"deployment fact {i} on host {i % 3}", at=T0 + i)
    result = mem.retrieve("deployment host", 60, now=T0 + 50, certificate=True)
    assert result.upper_bound >= result.objective - 1e-9


def test_consolidation_is_reversible_and_saves_tokens(mem):
    base = "The staging deploy failed because the migration timed out on the orders table."
    ids = [mem.write(f"{base} Attempt {i} was at night.", at=T0 + i) for i in range(4)]
    report = consolidate(mem, threshold=0.5, rate=0.5)
    assert report.clusters == 1 and report.tokens_after < report.tokens_before
    assert mem.stats()["archived"] == 4
    summary = mem.retrieve("staging deploy migration", 100, now=T0 + 10).items[0]
    assert summary.kind == "semantic"
    assert {e["id"] for e in expand(mem, summary.id)} >= set(ids)


def test_extract_summary_distortion_falls_with_rate():
    embedder = HashingEmbedder()
    sentences = [f"Sentence about topic {i} with detail {i * 7}." for i in range(12)]
    count = lambda s: len(s.split())
    distortions = [extract_summary(sentences, embedder, count, budget)[1] for budget in (8, 24, 48, 200)]
    assert distortions == sorted(distortions, reverse=True)
    assert distortions[-1] == pytest.approx(0.0, abs=1e-6)


def test_forget_archives_idle_then_lowest_density(mem):
    mem.write("Old idle note", at=T0)
    keep = mem.write("Pinned rule", at=T0, pinned=True)
    recent = mem.write("Recent note", at=T0 + 300 * DAY)
    assert forget(mem, T0 + 301 * DAY, idle_days=100) == 1
    assert mem.stats()["archived"] == 1
    assert mem.get(keep) and mem.get(recent)
    mem.write("Another recent note that is quite a bit longer than the others are", at=T0 + 300 * DAY)
    forget(mem, T0 + 301 * DAY, capacity_tokens=8)
    assert mem.stats()["active_tokens"] <= 8 + mem.get(keep).tokens


def test_lexical_only_store_works_without_embeddings():
    mem = Memory()
    mem.write("Lexical search is enough for identifiers like parseConfigFile", at=T0)
    result = mem.retrieve("where is parseConfigFile?", 100, now=T0 + 1, config=RetrievalConfig(gamma=0.0))
    assert result.items
    mem.close()


def _validity(mem, query, now):
    prepared = mem.prepare(query, now)
    return dict(zip(prepared.candidates.ids, prepared.candidates.validity))


def test_soft_supersession_drops_older_unkeyed_near_duplicates(mem):
    old = mem.write("The cache lives in redis on port 6379 for the billing service", at=T0, kind="semantic")
    new = mem.write("The cache lives in redis on port 6380 for the billing service", at=T0 + DAY, kind="semantic")
    validity = _validity(mem, "which port does the billing cache use?", T0 + 2 * DAY)
    assert old not in validity and new in validity


def test_a_twin_must_hold_the_name_the_query_asks_about(mem):
    asked = mem.write("dictAdd(dict *d, void *key) is the signature, defined in src/dict.c", at=T0, kind="semantic")
    other = mem.write("dictAddRaw(dict *d, void *key) is the signature, defined in src/dict.c", at=T0 + DAY, kind="semantic")
    validity = _validity(mem, "What parameters does dictAdd take?", T0 + 2 * DAY)
    assert asked in validity and other in validity


def test_a_note_that_only_says_less_is_not_a_newer_version(mem):
    longer = mem.write("skip_whitespace() is the signature, defined in src/lexer.hpp", at=T0, kind="semantic")
    shorter = mem.write("skip_whitespace is defined in src/lexer.hpp", at=T0 + DAY, kind="semantic")
    validity = _validity(mem, "What is the signature of skip_whitespace?", T0 + 2 * DAY)
    assert longer in validity and shorter in validity


def test_one_dropped_word_can_be_a_new_version(mem):
    old = mem.write("connect(host, port, timeout) is the signature, defined in src/client.c", at=T0, kind="semantic")
    new = mem.write("connect(host, port) is the signature, defined in src/client.c", at=T0 + DAY, kind="semantic")
    validity = _validity(mem, "What is the signature of connect?", T0 + 2 * DAY)
    assert old not in validity and new in validity


def test_similar_events_do_not_replace_each_other(mem):
    old = mem.write("commit 1a2b3c: fix the retry loop in the client (files: src/client.py)", at=T0)
    new = mem.write("commit 4d5e6f: fix the retry loop in the client (files: src/client.py, tests)", at=T0 + DAY)
    validity = _validity(mem, "retry loop in the client", T0 + 2 * DAY)
    assert validity[old] == pytest.approx(validity[new], rel=0.05)


def test_an_event_does_not_replace_a_fact(mem):
    fact = mem.write("The cache lives in redis on port 6379 for the billing service", at=T0, kind="semantic")
    event = mem.write("The cache lives in redis on port 6379 for the billing service today", at=T0 + DAY)
    validity = _validity(mem, "which port does the billing cache use?", T0 + 2 * DAY)
    assert validity[fact] == pytest.approx(validity[event], rel=0.05)


def test_soft_supersession_leaves_keyed_items_alone(mem):
    text = "get_user(user_id, timeout={}) is the signature, defined in src/users.py"
    a = mem.write(text.format(30), key="sig:get_user", at=T0)
    b = mem.write(text.format(31), key="sig:get_admin", at=T0 + DAY)
    validity = _validity(mem, "get_user signature", T0 + 2 * DAY)
    assert validity[a] == pytest.approx(validity[b], rel=0.05)


def test_reranker_signals_reach_the_features():
    calls = []

    def reranker(query, texts):
        calls.append(len(texts))
        return [10.0 if "uv" in t else -10.0 for t in texts]

    mem = Memory(embedder=HashingEmbedder(), reranker=reranker)
    good = mem.write("Install dependencies with uv sync", at=T0)
    mem.write("Install dependencies with pip install", at=T0)
    prepared = mem.prepare("how to install dependencies", T0 + 1)
    assert calls == [2]
    index = prepared.candidates.ids.index(good)
    assert prepared.candidates.rerank_rank[index] == 0
    assert prepared.candidates.rerank_score[index] > 0.99
    mem.close()


def test_reranker_with_unknown_scores_is_tolerated():
    mem = Memory(embedder=HashingEmbedder(), reranker=lambda q, texts: [float("nan")] * len(texts))
    mem.write("Anything about deployment", at=T0)
    assert mem.retrieve("deployment", 100, now=T0 + 1).items
    mem.close()
