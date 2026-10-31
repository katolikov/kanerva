"""Scaling a note by how often it helped against how often the model expected it to."""

import numpy as np

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
        ask_and_reward(memory, {a: 1.0})
    assert weight(memory, a) > before_a and weight(memory, b) < before_b


def test_one_miss_barely_moves_a_note():
    memory, a, _ = store()
    before = weight(memory, a)
    ask_and_reward(memory, {})
    assert weight(memory, a) > 0.8 * before


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
