"""The write path without keys: similar notes shown on write, and the agent's answer that one replaces another."""

import json
from pathlib import Path

import numpy as np
import pytest

from kanerva import Memory, hooks, memory
from kanerva.embed import HashingEmbedder
from kanerva.mcp_server import SHORT_NOTE_CHARS, Server
from kanerva.memory import DAY

ROOT = Path(__file__).resolve().parent.parent
T0 = 1_700_000_000.0
SCOPE = "project-a"
SECRET = "ghp_" + "a1B2c3D4e5" * 4


@pytest.fixture
def mem():
    m = Memory()
    yield m
    m.close()


@pytest.fixture
def server(mem):
    return Server(mem, scope=SCOPE)


def fact(mem, text, at=T0, **extra):
    return mem.write(text, scope=SCOPE, kind="semantic", at=at, **extra)


def test_the_old_sentence_is_shown_with_no_embedder(mem):
    old = fact(mem, "The API timeout is 30 seconds")
    new = fact(mem, "The API timeout is 60 seconds", at=T0 + DAY)
    found = mem.similar("The API timeout is 60 seconds", SCOPE, exclude=(new,))
    assert [i.id for i in found] == [old]
    assert 0.0 < found[0].weight <= 1.0


def test_a_one_word_edit_is_shown_among_other_notes(mem):
    for i, text in enumerate(("Lint runs with ruff", "Ports are listed in compose.yaml", "Releases are cut on Fridays")):
        fact(mem, text, at=T0 + i)
    old = fact(mem, "Deploys go through the staging branch", at=T0 + 5)
    new = fact(mem, "Deploys go through the release branch", at=T0 + 6)
    assert [i.id for i in mem.similar("Deploys go through the release branch", SCOPE, exclude=(new,))] == [old]


def test_unrelated_notes_are_not_shown(mem):
    fact(mem, "The API timeout is 30 seconds")
    fact(mem, "Deploys go through the staging branch")
    assert mem.similar("Translations live in the locale folder", SCOPE) == []


def test_only_live_facts_of_the_same_scope_are_shown(mem):
    live = fact(mem, "The API timeout is 30 seconds")
    mem.write("The API timeout is 40 seconds", scope=SCOPE, kind="episodic", at=T0)
    mem.write("The API timeout is 50 seconds", scope="global", kind="semantic", at=T0)
    mem.write("The API timeout is 70 seconds", scope="project-b", kind="semantic", at=T0)
    archived = fact(mem, "The API timeout is 80 seconds")
    mem.archive([archived])
    closed = fact(mem, "The API timeout is 90 seconds")
    assert mem.supersede(closed, fact(mem, "The API timeout is 95 seconds", at=T0 + 1))
    ids = [i.id for i in mem.similar("The API timeout is 60 seconds", SCOPE, limit=10)]
    assert live in ids and closed not in ids and archived not in ids
    assert all(mem.get(i).kind == "semantic" and mem.get(i).scope == SCOPE for i in ids)


def test_the_limit_and_the_order(mem):
    ids = [fact(mem, f"The API timeout is {n} seconds", at=T0 + n) for n in (10, 20, 30, 40)]
    fact(mem, "The API retry timeout for uploads is 50 seconds", at=T0 + 50)
    found = mem.similar("The API timeout is 60 seconds", SCOPE, limit=2)
    assert len(found) == 2 and {i.id for i in found} <= set(ids)
    assert found[0].weight >= found[1].weight


def test_a_long_summary_is_not_a_version_of_a_short_note(mem):
    words = " ".join(f"topic{i}" for i in range(40))
    fact(mem, f"The API timeout is 30 seconds, and the session covered {words}", source={"compact": True})
    assert mem.similar("The API timeout is 60 seconds", SCOPE) == []


def test_notes_with_no_words_find_nothing(mem):
    fact(mem, "The API timeout is 30 seconds")
    assert mem.similar("   ", SCOPE) == [] and mem.similar("the of and", SCOPE) == []


def test_looking_leaves_the_store_as_it_was(mem):
    fact(mem, "The API timeout is 30 seconds")
    tables = ("items", "accesses", "retrievals")
    before = [mem.db.execute(f"SELECT count(*) AS n FROM {t}").fetchone()["n"] for t in tables]
    assert mem.similar("The API timeout is 60 seconds", SCOPE)
    assert [mem.db.execute(f"SELECT count(*) AS n FROM {t}").fetchone()["n"] for t in tables] == before


class Vectors:
    """Two texts count as one direction when they share their first word."""

    dim = 8

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            out[row, sum(map(ord, text.split()[0].lower())) % self.dim] = 1.0
        return out


def test_vectors_add_notes_that_words_miss():
    mem = Memory(embedder=Vectors())
    old = mem.write("Payments settle through Stripe nightly", scope=SCOPE, kind="semantic", at=T0)
    mem.write("Logs rotate weekly", scope=SCOPE, kind="semantic", at=T0)
    new = mem.write("Payments moved over to Adyen", scope=SCOPE, kind="semantic", at=T0 + DAY)
    assert [i.id for i in mem.similar("Payments moved over to Adyen", SCOPE, exclude=(new,))] == [old]
    mem.close()


def test_vectors_of_an_earlier_embedder_do_no_harm():
    mem = Memory(embedder=HashingEmbedder())
    old = mem.write("The API timeout is 30 seconds", scope=SCOPE, kind="semantic", embedding=np.ones(4, dtype=np.float32))
    assert [i.id for i in mem.similar("The API timeout is 60 seconds", SCOPE)] == [old]
    mem.close()


def test_a_replaced_note_stays_as_history(mem):
    old = fact(mem, "The API timeout is 30 seconds")
    new = fact(mem, "The API timeout is 60 seconds", at=T0 + DAY)
    assert mem.supersede(old, new, at=T0 + DAY)
    row = mem.db.execute("SELECT * FROM items WHERE id=?", (old,)).fetchone()
    assert row["valid_to"] == T0 + DAY and row["superseded_by"] == new and row["tier"] == "active"
    assert mem.stats()["superseded"] == 1 and mem.get(old).superseded and not mem.get(new).superseded
    live = mem.retrieve("what is the API timeout?", 200, scope=SCOPE, now=T0 + 2 * DAY)
    past = mem.retrieve("what was the API timeout previously?", 200, scope=SCOPE, now=T0 + 2 * DAY)
    assert [i.id for i in live.items] == [new]
    assert old in [i.id for i in past.items]


def test_the_new_note_takes_over_the_key(mem):
    old = fact(mem, "The API timeout is 30 seconds", key="config:timeout")
    new = fact(mem, "The API timeout is 60 seconds", at=T0 + DAY)
    assert mem.supersede(old, new)
    third = fact(mem, "The API timeout is 90 seconds", at=T0 + 2 * DAY, key="config:timeout")
    assert mem.get(new).superseded and not mem.get(third).superseded


def test_a_note_keeps_its_own_key(mem):
    old = fact(mem, "The API timeout is 30 seconds", key="config:old")
    new = fact(mem, "The API timeout is 60 seconds", at=T0 + DAY, key="config:timeout")
    assert mem.supersede(old, new)
    assert mem.get(new).key == "config:timeout"


def test_a_value_that_comes_back_can_close_the_newer_one(mem):
    first = fact(mem, "Mode is fast")
    second = fact(mem, "Mode is safe", at=T0 + DAY)
    assert fact(mem, "Mode is fast", at=T0 + 2 * DAY) == first  # a reconfirmation, so the id is older than the note it closes
    assert mem.supersede(second, first)
    assert mem.get(second).superseded and not mem.get(first).superseded


def test_a_mistaken_mark_changes_nothing(mem):
    old = fact(mem, "The API timeout is 30 seconds")
    new = fact(mem, "The API timeout is 60 seconds", at=T0 + DAY)
    elsewhere = mem.write("The API timeout is 70 seconds", scope="project-b", kind="semantic", at=T0 + DAY)
    gone = fact(mem, "The API timeout is 75 seconds", at=T0 + DAY)
    mem.archive([gone])
    assert not mem.supersede(old, old)
    assert not mem.supersede(old, 9999) and not mem.supersede(9999, new)
    assert not mem.supersede(new, old)  # swapped arguments
    assert not mem.supersede(old, elsewhere)
    assert not mem.supersede(old, gone) and not mem.supersede(gone, new)
    assert mem.stats()["superseded"] == 0
    assert mem.supersede(old, new) and not mem.supersede(old, new)
    assert mem.db.execute("SELECT superseded_by FROM items WHERE id=?", (old,)).fetchone()[0] == new


def test_the_near_twin_rule_keeps_working_for_notes_that_were_not_marked(mem):
    old = fact(mem, "The API timeout is 30 seconds")
    new = fact(mem, "The API timeout is 60 seconds", at=T0 + DAY)
    result = mem.retrieve("API timeout", 200, scope=SCOPE, now=T0 + 2 * DAY)
    assert new in [i.id for i in result.items] and not mem.get(old).superseded


def test_remember_stays_small_with_nothing_similar(server):
    assert server.call("memory_remember", {"text": "The API timeout is 30 seconds"}) == {"id": 1}


def test_remember_lists_similar_notes_with_one_instruction(server):
    old = server.call("memory_remember", {"text": "The API timeout is 30 seconds"})["id"]
    reply = server.call("memory_remember", {"text": "The API timeout is 60 seconds"})
    assert set(reply) == {"id", "similar", "hint"}
    assert reply["similar"] == [{"id": old, "text": "The API timeout is 30 seconds"}]
    assert f"replaced_by={reply['id']}" in reply["hint"] and "memory_forget" in reply["hint"]
    assert len(json.dumps(reply)) < 500


def test_the_second_step_is_one_small_call(server):
    old = server.call("memory_remember", {"text": "The API timeout is 30 seconds"})["id"]
    new = server.call("memory_remember", {"text": "The API timeout is 60 seconds"})["id"]
    assert server.call("memory_forget", {"id": old, "replaced_by": new}) == {"replaced": True}
    assert server.mem.get(old).superseded
    assert server.call("memory_stats", {})["active"] == 2
    found = server.call("memory_search", {"query": "what is the API timeout?"})
    assert [m["id"] for m in found["memories"]] == [new]
    assert [n["id"] for n in server.call("memory_remember", {"text": "The API timeout is 90 seconds"})["similar"]] == [new]


def test_a_mark_never_deletes(server):
    old = server.call("memory_remember", {"text": "The API timeout is 30 seconds"})["id"]
    new = server.call("memory_remember", {"text": "The API timeout is 60 seconds"})["id"]
    assert server.call("memory_forget", {"id": old, "replaced_by": new, "hard": True}) == {"replaced": True}
    assert server.mem.get(old) is not None


def test_a_wrong_mark_says_so(server):
    old = server.call("memory_remember", {"text": "The API timeout is 30 seconds"})["id"]
    other = server.mem.write("The API timeout is 70 seconds", scope="project-b", kind="semantic")
    assert server.call("memory_forget", {"id": old, "replaced_by": other}) == {"replaced": False}
    assert not server.mem.get(old).superseded


def test_forgetting_works_as_before(server):
    item = server.call("memory_remember", {"text": "The API timeout is 30 seconds"})["id"]
    assert server.call("memory_forget", {"id": item}) == {"removed": 1}
    assert server.mem.get(item) is not None
    assert server.call("memory_forget", {"id": item, "hard": True}) == {"removed": 1}
    assert server.mem.get(item) is None


def test_a_note_is_not_similar_to_itself(server):
    server.call("memory_remember", {"text": "The API timeout is 30 seconds"})
    assert server.call("memory_remember", {"text": "the api  timeout is 30 SECONDS"}) == {"id": 1}


def test_a_key_closes_its_chain_before_the_list_is_made(server):
    server.call("memory_remember", {"text": "The API timeout is 30 seconds", "key": "config:timeout"})
    reply = server.call("memory_remember", {"text": "The API timeout is 60 seconds", "key": "config:timeout"})
    assert set(reply) == {"id"}


def test_a_global_note_only_sees_global_notes(server):
    server.call("memory_remember", {"text": "The API timeout is 30 seconds"})
    assert set(server.call("memory_remember", {"text": "The API timeout is 60 seconds", "scope": "global"})) == {"id"}


def test_an_empty_note_lists_nothing(server):
    assert server.call("memory_remember", {"text": "   "}) == {"id": None}


def test_long_notes_are_cut_in_the_reply(server):
    text = "The API timeout for uploads " + "and downloads across the fleet " * 20 + "is 30 seconds"
    server.call("memory_remember", {"text": text})
    reply = server.call("memory_remember", {"text": text.replace("30", "60")})
    assert len(reply["similar"][0]["text"]) <= SHORT_NOTE_CHARS + 3


def test_the_list_shows_the_filtered_text_only(server):
    server.call("memory_remember", {"text": f"The deploy token for staging is {SECRET}"})
    reply = server.call("memory_remember", {"text": f"The deploy token for staging is {SECRET[:-1]}x and it rotates"})
    assert SECRET not in json.dumps(reply) and SECRET[:-1] not in json.dumps(reply)
    assert "[REDACTED:github-token]" in reply["similar"][0]["text"]


def test_a_note_older_than_a_filter_pattern_is_filtered_on_the_way_out(server):
    item = server.call("memory_remember", {"text": "The deploy token of the staging cluster is unknown"})["id"]
    server.mem.db.execute("UPDATE items SET text=? WHERE id=?", (f"The deploy token of the staging cluster is {SECRET}", item))
    found = server.mem.similar("The deploy token of the staging cluster was rotated", SCOPE, overlap=0.0)
    assert found and SECRET not in found[0].text and "[REDACTED:github-token]" in found[0].text


def test_a_secret_in_the_new_note_does_not_decide_the_search(mem):
    fact(mem, f"The upload bucket key is {SECRET}")
    assert mem.similar(SECRET, SCOPE) == []
    row = mem.db.execute("SELECT count(*) AS n FROM retrievals").fetchone()
    assert row["n"] == 0


def test_the_tool_schema_offers_the_second_step(server):
    tools = {t["name"]: t for t in server.handle({"id": 1, "method": "tools/list"})["result"]["tools"]}
    assert "replaced_by" in tools["memory_forget"]["inputSchema"]["properties"]
    assert tools["memory_forget"]["inputSchema"]["required"] == ["id"]


def test_the_wrap_up_prompt_leaves_room_for_the_second_step():
    assert "memory_remember" in hooks.WRAPUP_PROMPT and "no key" in hooks.WRAPUP_PROMPT
    assert hooks.WRAPUP_PROMPT.index("replaces") < hooks.WRAPUP_PROMPT.index("Then stop")


@pytest.mark.parametrize("path", [
    "plugins/claude-code/skills/remember/SKILL.md", "plugins/codex/skills/remember/SKILL.md",
    "plugins/codex/skills/forget/SKILL.md", "plugins/mcp/README.md",
])
def test_the_agents_are_told(path):
    assert "replaced_by" in (ROOT / path).read_text()


def test_the_server_tells_the_host_when_to_save(tmp_path):
    server = Server(Memory(tmp_path / "memory.sqlite"), scope="project:x")
    answer = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert "memory_remember" in answer["result"]["instructions"]


def vector(*values):
    v = np.array(values + (0.0,) * (384 - len(values)), dtype=np.float32)
    return v / np.linalg.norm(v)


def test_a_paraphrase_is_listed_by_its_vector(tmp_path):
    """An agent's two notes on one fact can share few words, and one may be much longer than the other."""
    mem = Memory(tmp_path / "memory.sqlite")
    long = "User brews with a French press and has tried several beans lately, mostly light roasts from a local shop, at one spoon per six ounces"
    old = mem.write(long, kind="semantic", embedding=vector(1.0, 0.0), scope="s")
    far = mem.write("Deploys go through the staging branch on Fridays", kind="semantic", embedding=vector(0.0, 1.0), scope="s")
    new = mem.write("Coffee ratio is now five ounces", kind="semantic", embedding=vector(0.6, 0.1), scope="s")
    listed = [item.id for item in mem.similar("Coffee ratio is now five ounces", "s", exclude=(new,), query_vector=vector(0.6, 0.1))]
    assert old in listed and far not in listed
    mem.close()


def test_the_server_says_fixing_old_notes_is_part_of_saving(tmp_path):
    server = Server(Memory(tmp_path / "memory.sqlite"), scope="project:x")
    answer = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})["result"]["instructions"]
    assert "out of date" in answer and "same turn" in answer
    server.call("memory_remember", {"text": "The API timeout is 30 seconds"})
    hint = server.call("memory_remember", {"text": "The API timeout is 60 seconds"})["hint"]
    assert "memory_edit" in hint and "without asking" in hint


def test_neither_kind_of_match_crowds_out_the_other():
    assert memory.alternate([1, 2, 3, 4], [9]) == [1, 9, 2, 3, 4]
    assert memory.alternate([1], [8, 9]) == [1, 8, 9]
    assert memory.alternate([], [8]) == [8]
