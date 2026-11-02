"""Scaling a note by how often it helped against how often the model expected it to."""

import json

import numpy as np

from kanerva.learning import (
    FLOORS,
    GATE_SESSIONS,
    HELPED_QUERIES,
    LIFT_RANGE,
    PROBE_RATE,
    PROBE_UNJUDGED,
    SHADOW_CAP,
    Tuning,
    best_floor,
    best_shift,
    compare,
    learning_on,
    pool,
    probes,
    reach,
    shadow_update,
    tuning,
)
from kanerva.memory import Memory
from kanerva.policy import RetrievalConfig

CONFIG = RetrievalConfig(use_dense=False, gamma=0.0, neighbour_top=0, use_bandit=False, learn=False, item_lift=True)
QUERY = "how does the upload retry work"


def store() -> tuple[Memory, int, int]:
    memory = Memory(":memory:", config=CONFIG)
    a = memory.write("The upload retry waits two seconds between attempts.", at=1.0, kind="semantic")
    b = memory.write("The upload retry gives up after five attempts.", at=2.0, kind="semantic")
    return memory, a, b


def weight(memory: Memory, item: int) -> float:
    prepared = memory.prepare(QUERY, 10.0)
    return float(prepared.problem.weights[prepared.candidates.ids.index(item)])


def ask_and_reward(memory: Memory, rewards: dict[int, float]) -> None:
    result = memory.retrieve(QUERY, 200, now=10.0)
    assert result.retrieval_id is not None
    memory.feedback(result.retrieval_id, rewards)


def test_a_note_with_no_record_keeps_the_model_weight():
    memory, a, _ = store()
    plain = memory.prepare(QUERY, 10.0, config=CONFIG.with_(item_lift=False))
    lifted = memory.prepare(QUERY, 10.0)
    assert np.allclose(plain.problem.weights, lifted.problem.weights)


def test_a_note_that_helps_rises_and_one_that_never_helps_falls():
    memory, a, b = store()
    before_a, before_b = weight(memory, a), weight(memory, b)
    for _ in range(20):
        ask_and_reward(memory, {a: 1.0})  # on this very query
    assert weight(memory, a) > before_a and weight(memory, b) < before_b


def test_one_miss_moves_a_note_by_less_than_a_third():
    memory, a, _ = store()
    before = weight(memory, a)
    ask_and_reward(memory, {})
    assert before > weight(memory, a) > 2 / 3 * before


def test_a_store_from_before_the_column_opens(tmp_path):
    path = tmp_path / "old.db"
    memory = Memory(path, config=CONFIG)
    memory.write("a note", at=1.0, kind="semantic")
    memory.commit()
    memory.db.execute("ALTER TABLE items DROP COLUMN expected")
    memory.commit()
    memory.close()
    reopened = Memory(path, config=CONFIG)
    assert "expected" in {r["name"] for r in reopened.db.execute("PRAGMA table_info(items)")}
    reopened.close()


def test_two_processes_can_add_a_column_at_once(tmp_path):
    path = tmp_path / "old.db"
    first = Memory(path, config=CONFIG)
    first.db.execute("ALTER TABLE items DROP COLUMN expected")
    first.commit()
    second = Memory(path, config=CONFIG)
    first._add_column("expected", "expected REAL NOT NULL DEFAULT 0")
    assert "expected" in {r["name"] for r in first.db.execute("PRAGMA table_info(items)")}
    second.close()
    first.close()


def test_a_new_note_inherits_how_its_group_did_in_its_project():
    memory = Memory(":memory:", config=CONFIG)
    proven = memory.write("Test command: pytest -x", at=1.0, kind="semantic", key="cmd:test:pytest", scope="p")
    idle = memory.write("Edited src/upload.py", at=2.0, kind="episodic", scope="p")
    memory.db.execute("UPDATE items SET hits = 20, chances = 10 WHERE id = ?", (proven,))
    memory.db.execute("UPDATE items SET hits = 0, chances = 10 WHERE id = ?", (idle,))
    pooled = pool(memory.db)
    assert pooled.lift("p", "test", 0.0, 0.0) > 1.0 > pooled.lift("p", "edit", 0.0, 0.0)


def test_another_project_starts_from_the_record_across_projects():
    memory = Memory(":memory:", config=CONFIG)
    a = memory.write("Test command: pytest -x", at=1.0, kind="semantic", key="cmd:test:pytest", scope="a")
    b = memory.write("Edited src/upload.py", at=2.0, kind="episodic", scope="a")
    memory.db.execute("UPDATE items SET hits = 20, chances = 10 WHERE id = ?", (a,))
    memory.db.execute("UPDATE items SET hits = 0, chances = 10 WHERE id = ?", (b,))
    pooled = pool(memory.db)
    assert pooled.lift("fresh", "test", 0.0, 0.0) > 1.0 > pooled.lift("fresh", "edit", 0.0, 0.0)


def test_lift_stays_in_range():
    memory = Memory(":memory:", config=CONFIG)
    a = memory.write("Test command: pytest -x", at=1.0, kind="semantic", key="cmd:test:pytest", scope="p")
    b = memory.write("Edited src/upload.py", at=2.0, kind="episodic", scope="p")
    memory.db.execute("UPDATE items SET hits = 1000, chances = 1000 WHERE id = ?", (a,))
    memory.db.execute("UPDATE items SET hits = 0, chances = 1000 WHERE id = ?", (b,))
    pooled = pool(memory.db)
    assert pooled.lift("p", "edit", 0.0, 1000.0) == LIFT_RANGE[0]
    assert pooled.lift("p", "test", 1000.0, 1.0) == LIFT_RANGE[1]


def test_uniform_shyness_of_an_agent_moves_nothing():
    """An agent that marks a tenth of what the model expects says nothing about any one note."""
    memory = Memory(":memory:", config=CONFIG)
    notes = [memory.write(f"note {i}", at=float(i), kind="semantic", scope="p") for i in range(4)]
    for item in notes:
        memory.db.execute("UPDATE items SET hits = 10, chances = 100 WHERE id = ?", (item,))
    pooled = pool(memory.db)
    assert abs(pooled.lift("p", "semantic", 10.0, 100.0) - 1.0) < 0.05


def test_shadow_waits_for_enough_pairs_and_turns_on_for_a_real_lead():
    memory = Memory(":memory:", config=CONFIG)
    shadow_update(memory.db, "p", 10.0, 14.0, 20.0)
    assert not learning_on(memory.db, "p")
    for _ in range(3):
        shadow_update(memory.db, "p", 10.0, 14.0, 20.0)
    assert learning_on(memory.db, "p")


def test_shadow_stays_off_on_a_tie_and_stops_when_the_lead_goes():
    memory = Memory(":memory:", config=CONFIG)
    for _ in range(5):
        shadow_update(memory.db, "p", 10.0, 10.0, 20.0)
    assert not learning_on(memory.db, "p")
    for _ in range(5):
        shadow_update(memory.db, "q", 10.0, 14.0, 20.0)
    assert learning_on(memory.db, "q")
    for _ in range(40):
        shadow_update(memory.db, "q", 14.0, 10.0, 20.0)
    assert not learning_on(memory.db, "q")


def test_a_new_project_follows_the_record_across_projects_until_it_has_its_own():
    memory = Memory(":memory:", config=CONFIG)
    for _ in range(5):
        shadow_update(memory.db, "a", 10.0, 14.0, 20.0)
    assert learning_on(memory.db, "b")
    for _ in range(4):
        shadow_update(memory.db, "b", 14.0, 10.0, 20.0)
    assert not learning_on(memory.db, "b")


def test_compare_counts_ties_as_half():
    assert compare([(0.3, 0.2, 0.1, 0.2), (0.2, 0.2, 0.3, 0.2)]) == (1.5, 1.0, 2.0)


def test_verdict_scores_orderings_before_its_record_changes_them():
    memory, a, b = store()
    memory.db.execute("UPDATE items SET hits = 40, chances = 10 WHERE id = ?", (a,))
    memory.db.execute("UPDATE items SET hits = 0, chances = 10 WHERE id = ?", (b,))
    result = memory.retrieve(QUERY, 200, now=10.0, scope=None)
    memory.shadow_check("p", [result.retrieval_id], {a})
    record = json.loads(memory.db.execute("SELECT value FROM meta WHERE key = 'shadow:p'").fetchone()["value"])
    assert record["pairs"] == 1.0 and record["learned"] >= record["model"]


def rows_of(*groups):
    """(value, helped, count) triples, each row in a session of its own."""
    return [(value, helped, 1.0, f"s{i}-{j}") for i, (value, helped, count) in enumerate(groups) for j in range(count)]


def test_the_gate_stays_put_on_little_or_even_evidence():
    assert best_shift(rows_of((0.2, 1, 1), (0.7, 0, 1))) == 0.0
    assert best_shift([(m / 10, int(i % 4 == 0), 1.0, str(i)) for i, m in enumerate(range(-10, 10))]) == 0.0


def test_the_gate_tightens_when_what_it_lets_through_does_not_help():
    assert best_shift(rows_of((0.2, 0, 20), (1.2, 1, 10))) > 0.0


def test_the_gate_opens_when_prompts_just_below_it_would_have_helped():
    assert best_shift(rows_of((-0.3, 1, 10), (0.5, 1, 10))) < 0.0


def test_one_long_session_cannot_move_the_gate():
    assert best_shift([(0.2, 0, 1.0, "one")] * 28) == 0.0


def test_a_project_without_sessions_of_its_own_follows_the_others():
    memory = Memory(":memory:", config=CONFIG)
    for i in range(30):
        memory.db.execute("INSERT INTO gate_log VALUES (?, 'a', ?, ?, 0.2, 1, 1, '[]', 0)", (i, f"s{i}", float(i)))
    for i in range(30, 40):
        memory.db.execute("INSERT INTO gate_log VALUES (?, 'a', ?, ?, 1.2, 1, 1, '[]', 1)", (i, f"s{i}", float(i)))
    assert tuning(memory.db, "b").shift > 0.0
    assert tuning(memory.db, "a").shift == tuning(memory.db, "b").shift


def test_a_project_probes_more_until_it_has_its_own_sessions():
    memory = Memory(":memory:", config=CONFIG)
    assert tuning(memory.db, "a").probe == PROBE_RATE[0]
    for i in range(GATE_SESSIONS):
        memory.db.execute("INSERT INTO gate_log VALUES (?, 'a', ?, ?, 0.2, 1, 1, '[]', 1)", (i, f"s{i}", float(i)))
    assert tuning(memory.db, "a").probe == PROBE_RATE[1]
    assert tuning(memory.db, "b").probe == PROBE_RATE[0]


def test_probing_stops_where_verdicts_do_not_come():
    memory = Memory(":memory:", config=CONFIG)
    for i in range(PROBE_UNJUDGED):
        memory.db.execute("INSERT INTO gate_log VALUES (?, 'a', NULL, ?, 0.2, 4, 4, '[]', NULL)", (i, float(i)))
    assert tuning(memory.db, "a").probe == 0.0
    assert tuning(memory.db, "b", can_ask=False).probe == 0.0


def test_the_floor_falls_when_notes_under_it_help():
    rows = [(0.3, 1, 1.0, f"s{i}") for i in range(6)] + [(0.3, 0, 1.0, f"t{i}") for i in range(4)] + [(2.0, 1, 1.0, f"u{i}") for i in range(5)]
    assert best_floor(rows) < FLOORS[0]
    assert best_floor([(0.3, 0, 1.0, f"s{i}") for i in range(30)] + [(2.0, 1, 1.0, f"u{i}") for i in range(5)]) == FLOORS[0]


def test_a_prompt_only_a_probe_could_show_stands_for_the_prompts_without_one():
    plain = [(-0.3, 1, 1.0, f"s{i}") for i in range(2)] + [(0.5, 1, 1.0, f"t{i}") for i in range(8)]
    weighted = [(-0.3, 1, 4.0, f"s{i}") for i in range(2)] + [(0.5, 1, 1.0, f"t{i}") for i in range(8)]
    assert best_shift(plain) == 0.0 and best_shift(weighted) < 0.0


def test_one_helped_probe_cannot_move_the_gate():
    assert best_shift([(-0.3, 1, 4.0, "s")] + [(0.5, 1, 1.0, f"t{i}") for i in range(8)]) == 0.0


def test_probing_stays_inside_the_band():
    assert not probes("anything at all", -3.5, 1.0)
    chosen = sum(probes(f"prompt {i}", -0.5, 0.1) for i in range(1000))
    assert 50 < chosen < 150
    assert probes("prompt 7", -0.5, 0.1) == probes("prompt 7", -0.5, 0.1)


def test_no_probing_outside_plugins():
    memory = Memory(":memory:", config=CONFIG)
    assert memory.tuning == Tuning()


def test_a_verdict_reaches_the_gate_log():
    memory, a, _ = store()
    prepared = memory.prepare(QUERY, 10.0)
    result = memory.select(prepared, 200, gated=False)
    ratios = json.dumps([1.0 for _ in result.items])
    memory.db.execute("INSERT INTO gate_log VALUES (?, 'p', NULL, 10.0, 0.5, 1, 1, ?, NULL)", (result.retrieval_id, ratios))
    memory.feedback(result.retrieval_id, {a: 1.0}, session="s")
    assert memory.db.execute("SELECT helped FROM gate_log").fetchone()["helped"] == 1
    assert memory.db.execute("SELECT count(*) AS n, sum(helped) AS good FROM floor_log").fetchone()["good"] == 1


def test_a_note_found_by_hand_counts_for_itself_and_not_for_the_rate():
    memory, a, b = store()
    memory.db.execute("UPDATE items SET hits = 2, chances = 4 WHERE id = ?", (b,))
    before = pool(memory.db).base
    memory.credit({a})
    row = memory.db.execute("SELECT found, chances FROM items WHERE id = ?", (a,)).fetchone()
    assert (row["found"], row["chances"]) == (1.0, 0.0)
    assert pool(memory.db).base == before


def test_a_note_that_misled_beside_a_file_counts_as_a_wasted_chance():
    memory, a, _ = store()
    memory.credit((), {a})
    assert memory.db.execute("SELECT chances FROM items WHERE id = ?", (a,)).fetchone()["chances"] == 1.0


def test_older_signals_do_not_count_as_verdicts():
    memory, a, _ = store()
    memory.db.execute("UPDATE items SET used = 9, expected = 3 WHERE id = ?", (a,))
    assert pool(memory.db).base == 1.0 and not pool(memory.db).groups


def test_an_edit_does_not_copy_a_note_s_record():
    memory, a, _ = store()
    memory.db.execute("UPDATE items SET hits = 2, chances = 3, found = 1 WHERE id = ?", (a,))
    assert memory.edit(a, "The upload retry waits three seconds between attempts.")
    assert memory.db.execute("SELECT sum(hits) AS h, sum(chances) AS c FROM items").fetchone()["h"] == 2.0


def test_one_verdict_adds_a_bounded_number_of_pairs():
    assert compare([(0.3, 0.2, 0.3, 0.2)] * 40)[2] == SHADOW_CAP


def test_a_verdict_belongs_to_its_project():
    memory, a, _ = store()
    memory.verdict([a], scope="x")
    assert memory.open_verdict("y") is None
    at, helped, misled = memory.open_verdict("x")
    assert helped == {a} and not misled
    memory.close_verdict("x", at)
    assert memory.open_verdict("x") is None


def test_a_helped_mark_does_not_make_an_old_note_look_new():
    memory, a, _ = store()
    before = memory.db.execute("SELECT confirmed_at FROM items WHERE id = ?", (a,)).fetchone()["confirmed_at"]
    ask_and_reward(memory, {a: 1.0})
    assert memory.db.execute("SELECT confirmed_at FROM items WHERE id = ?", (a,)).fetchone()["confirmed_at"] == before


def test_a_rise_needs_a_query_like_the_ones_a_note_helped_on():
    assert reach(3.0, 0.0) == 1.0
    assert reach(3.0, 1.0) == 3.0
    assert 1.0 < reach(3.0, 0.5) < 3.0


def test_a_fall_needs_no_match():
    assert reach(0.5, 0.0) == 0.5


def test_a_note_rises_on_a_like_query_and_not_on_another():
    memory, a, _ = store()
    memory.db.execute("UPDATE items SET hits = 5, chances = 1 WHERE id = ?", (a,))
    memory.db.execute("INSERT INTO helped_queries VALUES (?, 1.0, ?, NULL)", (a, QUERY))
    like = memory.helped_matches([a], QUERY, None)[a]
    other = memory.helped_matches([a], "where do the deployment scripts live", None)[a]
    assert like > other


def test_a_verdict_remembers_the_query_it_helped_on():
    memory, a, _ = store()
    ask_and_reward(memory, {a: 1.0})
    assert memory.db.execute("SELECT query FROM helped_queries WHERE item_id = ?", (a,)).fetchone()["query"] == QUERY


def test_a_note_keeps_only_its_latest_helped_queries():
    memory, a, _ = store()
    for at in range(HELPED_QUERIES + 3):
        memory._remember_helped(a, float(at), f"query {at}", None)
    assert memory.db.execute("SELECT count(*) AS n FROM helped_queries").fetchone()["n"] == HELPED_QUERIES
