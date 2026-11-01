"""What a store learns from its agent's verdicts, on top of the shipped relevance model.

The model knows about notes in general. A verdict says what happened to one note in one project, and such evidence is scarce, so
it is pooled on three levels. A note starts from the record of its group in its project, and that record starts from the same
group across all projects. Groups are what lets a new note inherit anything: a fresh test command starts from how test commands
have done. With no evidence every level says 1, and the model's weight stands.

Lift runs in the shadow first. Each verdict also scores the two orderings of the notes it covers, the model's and the learned
one, using only what was known before that verdict. Lift starts to change what gets injected in a project once the learned
ordering is ahead there, or across projects while the project has too little of its own, and it stops if that changes. Only shown
notes get verdicts, so the check cannot see a note that either ordering left out.

The gate and the floor under note weights need evidence the store cannot get by waiting, because a note they stop is never
shown. So a few prompts are probed from the start: they get a wider list, and verdicts on it say what the settings hide. A prompt
that only a probe let through stands for as many prompts as the probe rate implies. Each session counts once, however many
prompts it held, and probing stops in a project where verdicts do not come.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Iterable

BASE_PRIOR = 1.0  # weak: how an agent's marks scale against the model's chances is a fact about the agent, not about notes
GROUP_PRIOR = 10.0  # expected uses at which a group's own record weighs as much as the level above it
NOTE_PRIOR = 1.0  # low, because a real agent's mark is rarely wrong: 96.7% right on a test built to check it
LIFT_RANGE = (0.25, 4.0)
SHADOW_PAIRS = 40.0  # a level needs this many compared pairs before it can decide
SHADOW_MARGIN = 0.02  # the learned ordering must win this share of pairs more than the model's, so that noise cannot flip it
SHADOW_DECAY = 0.98  # per verdict, so that an old lead fades and learning can stop again
SHADOW_CAP = 10.0  # pairs one verdict can add, so that one long session cannot decide alone
ALL_PROJECTS = "*"
PROBE_BAND = 3.0  # a prompt below the gate by no more than this can still be probed
PROBE_RATE = (0.25, 0.1)  # the share of prompts probed while a project has few judged sessions, and after
PROBE_FLOOR = 0.1  # a probe shows notes down to this share of the weight floor, so that verdicts reach what the floor hides
PROBE_UNJUDGED = 8  # probes with no verdict before a project stops probing
GATE_WORTH = 0.25  # speaking pays if at least this share of prompts bring a note that helps
GATE_SHIFTS = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0)
NOTE_WORTH = 0.05  # a note pays if at least this share of notes at its weight help
FLOORS = (1.0, 0.5, 0.25, 0.1)  # shares of the shipped weight floor
GATE_SESSIONS = 8  # judged sessions a level needs before it can move a setting
GATE_LEAD = 2.0  # sessions of gain over the shipped setting before it moves, so that noise cannot
GATE_WINDOW = 200


def note_group(key: str | None, kind: str, text: str) -> str:
    if text.startswith("Edited "):
        return "edit"
    if key:
        return "test" if key.startswith("cmd:test") else "build" if key.startswith("cmd:build") else "keyed"
    return kind


def _shrunk(hits: float, chances: float, prior: float, strength: float) -> float:
    return (hits + strength * prior) / (chances + strength)


@dataclass
class Pool:
    """Hits and chances by group and by project and group. ``base`` is the rate over all notes.

    Chances count in the model's units and hits in the agent's, and the two need not agree, since an agent marks fewer notes
    than a benchmark labels. Dividing by ``base`` removes that, so that only differences between notes count.
    """

    base: float = 1.0
    groups: dict[str, float] = field(default_factory=dict)
    projects: dict[tuple[str, str], float] = field(default_factory=dict)

    def prior(self, scope: str, group: str) -> float:
        return self.projects.get((scope, group), self.groups.get(group, 1.0))

    def lift(self, scope: str, group: str, hits: float, chances: float) -> float:
        low, high = LIFT_RANGE
        return min(high, max(low, _shrunk(hits, self.base * chances, self.prior(scope, group), NOTE_PRIOR)))


def pool(db: sqlite3.Connection) -> Pool:
    """Hits with no chance on record, from a note the agent found itself, count for that note and not for its group or the rate."""
    rows = db.execute("SELECT scope, key, kind, substr(text, 1, 7) AS head, hits, chances FROM items WHERE hits != 0 OR chances != 0").fetchall()
    by_group: dict[str, list[float]] = {}
    by_project: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        group = note_group(row["key"], row["kind"], row["head"])
        for table, name in ((by_group, group), (by_project, (row["scope"], group))):
            entry = table.setdefault(name, [0.0, 0.0])
            entry[0] += row["hits"]
            entry[1] += row["chances"]
    hits, chances = sum(e[0] for e in by_group.values()), sum(e[1] for e in by_group.values())
    base = _shrunk(hits, chances, 1.0, BASE_PRIOR)
    groups = {g: _shrunk(h, base * c, 1.0, GROUP_PRIOR) for g, (h, c) in by_group.items()}
    projects = {(s, g): _shrunk(h, base * c, groups[g], GROUP_PRIOR) for (s, g), (h, c) in by_project.items()}
    return Pool(base, groups, projects)


def _record(db: sqlite3.Connection, level: str) -> dict:
    row = db.execute("SELECT value FROM meta WHERE key = ?", (f"shadow:{level}",)).fetchone()
    return json.loads(row["value"]) if row else {"model": 0.0, "learned": 0.0, "pairs": 0.0, "on": False}


def compare(pairs: Iterable[tuple[float, float, float, float]]) -> tuple[float, float, float]:
    """Wins of each ordering over (helped note, idle note) pairs, given as model and learned weight of each. A tie is half."""
    model, learned, n = 0.0, 0.0, 0.0
    for m_good, m_idle, l_good, l_idle in pairs:
        model += 1.0 if m_good > m_idle else 0.5 if m_good == m_idle else 0.0
        learned += 1.0 if l_good > l_idle else 0.5 if l_good == l_idle else 0.0
        n += 1.0
    scale = min(1.0, SHADOW_CAP / n) if n else 1.0
    return model * scale, learned * scale, n * scale


def shadow_update(db: sqlite3.Connection, scope: str, model: float, learned: float, pairs: float) -> None:
    for level in (scope, ALL_PROJECTS):
        record = _record(db, level)
        record["model"] = SHADOW_DECAY * record["model"] + model
        record["learned"] = SHADOW_DECAY * record["learned"] + learned
        record["pairs"] = SHADOW_DECAY * record["pairs"] + pairs
        lead = (record["learned"] - record["model"]) / max(record["pairs"], 1.0)
        if record["pairs"] >= SHADOW_PAIRS:
            record["on"] = lead >= SHADOW_MARGIN if not record["on"] else lead > 0.0
        db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (f"shadow:{level}", json.dumps(record)))


def learning_on(db: sqlite3.Connection, scope: str) -> bool:
    """The project's own record decides once it has enough pairs, and until then the record across projects."""
    own = _record(db, scope)
    return bool(own["on"] if own["pairs"] >= SHADOW_PAIRS else _record(db, ALL_PROJECTS)["on"])


@dataclass(frozen=True)
class Tuning:
    """How a project's prompts are answered: the shift of the gate, the share of the weight floor, and the share of prompts probed."""

    shift: float = 0.0
    floor: float = 1.0
    probe: float = 0.0


def probes(query: str, below: float, rate: float) -> bool:
    """Whether a prompt gets a wider list. Its text decides, so that a repeat behaves the same."""
    digest = hashlib.blake2b(query.encode(), digest_size=8).digest()
    return below >= -PROBE_BAND and int.from_bytes(digest, "little") / 2**64 < rate


def _best(options: tuple[float, ...], stay: float, rows: list[tuple], cost: float) -> float:
    """The setting that keeps the most worth over judged sessions, or none unless it clearly beats staying.

    Rows are (value, helped, weight, session). Each session counts once, so one long session cannot move a setting.
    """
    sessions: dict[str, list[tuple]] = {}
    for row in rows:
        sessions.setdefault(row[3], []).append(row)

    def worth(limit: float) -> float:
        return sum(sum(w * (helped - cost) for value, helped, w, _ in group if value >= limit) / sum(r[2] for r in group) for group in sessions.values())

    best = max(options, key=lambda o: (worth(o), -abs(o - stay)))
    return best if worth(best) >= worth(stay) + GATE_LEAD else stay


def best_shift(rows: list[tuple]) -> float:
    return _best(GATE_SHIFTS, 0.0, rows, GATE_WORTH)


def best_floor(rows: list[tuple]) -> float:
    return _best(FLOORS, FLOORS[0], rows, NOTE_WORTH)


def _judged(db: sqlite3.Connection, table: str, value: str, weight: str, scope: str) -> tuple[list[tuple], bool]:
    """A project's rows once it has enough judged sessions, and until then the rows across projects."""
    query = f"SELECT {value}, helped, {weight}, session FROM {table} WHERE helped IS NOT NULL{{}} ORDER BY at DESC LIMIT ?"
    rows = [tuple(r) for r in db.execute(query.format(" AND scope = ?"), (scope, GATE_WINDOW))]
    if len({r[3] for r in rows}) >= GATE_SESSIONS:
        return rows, True
    return [tuple(r) for r in db.execute(query.format(""), (GATE_WINDOW,))], False


def tuning(db: sqlite3.Connection, scope: str, can_ask: bool = True) -> Tuning:
    prompts, own = _judged(db, "gate_log", "margin", "prompt_weight", scope)
    notes, _ = _judged(db, "floor_log", "ratio", "note_weight", scope)
    enough = len({r[3] for r in prompts}) >= GATE_SESSIONS
    unjudged = db.execute("SELECT count(*) FROM gate_log WHERE scope = ? AND note_weight > 1 AND helped IS NULL", (scope,)).fetchone()[0]
    probe = PROBE_RATE[own] if can_ask and unjudged < PROBE_UNJUDGED else 0.0
    return Tuning(best_shift(prompts) if enough else 0.0, best_floor(notes) if enough else FLOORS[0], probe)
