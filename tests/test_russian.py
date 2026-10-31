"""Russian as a supported language: the stemmer, stems in the index, query shapes, old stores and the multilingual model."""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from kanerva import embed, memory, relevance, text
from kanerva.embed import HashingEmbedder, fetch_minilm, fetch_multilingual, multilingual_folder, multilingual_ready
from kanerva.memory import SCHEMA_VERSION, Memory, open_store
from kanerva.relevance import ItemView, feature_matrix
from kanerva.text import content_terms, fts_query, light_stem, russian_stem, search_text, tokenize

ROOT = Path(__file__).resolve().parent.parent

# Pairs from the PostgreSQL 16 build of the Snowball stemmer, and from the sample vocabulary on snowballstem.org.
SNOWBALL = {
    "вагоны": "вагон", "важные": "важн", "валяются": "валя", "павловцы": "павловц", "абдивели": "абдивел",
    "мелодию": "мелод", "образуют": "образ", "создан": "созда", "указанных": "указа", "прочитав": "прочита",
    "прочитавшись": "прочита", "улыбаясь": "улыб", "сильнейший": "сильн", "длинный": "длин", "ёлка": "елк", "всё": "все",
    "знание": "знан", "настройками": "настройк", "австралия": "австрал", "данных": "дан", "хорошие": "хорош",
    "я": "я", "кк": "кк", "ь": "ь",
}

# These strings hold no Russian, so a change to the digest of what the helpers return for them is a change for English.
ENGLISH = [
    "The API timeout is 30 seconds, set in config/settings.yaml on 2024-03-15.",
    "parseHTTPResponse_v2 returns a RetryPolicy; see src/kanerva/memory.py line 42",
    "When did I last run the migrations? I'm sure it was before the week of May 3.",
    "How many tests failed yesterday, and how long did the CI take?",
    "What time does my deploy start? What percentage of requests fail?",
    "Previously we used to pin numpy; originally it was 1.26, changed from 1.24 at first.",
    "user: I've moved the old cache to Redis. dev: Done, the history stays.",
    "Melanie went camping on Monday, June 20th, 2023 with her kids in Yosemite",
    "日本語のテキストと한국어 and 中文 mixed with English words",
    "café naïve résumé Zürich Straße ÉCOLE",
    "e=mc^2 && x != y || foo->bar(); #include <stdio.h>",
    "studies running tested boxes cats class glasses processes classes",
    "I i me my mine myself MY Me we our",
    "",
    "   ",
    "a an the of to",
]
ENGLISH_DIGEST = "26a0a6ec1b48e2865fe5abed722e08c6e229a53e10ea7e033990e56ef44e75e4"


def english_row(s: str) -> list:
    terms = content_terms(s)
    return [
        tokenize(s), terms, search_text(s), fts_query(terms), [light_stem(t) for t in tokenize(s)],
        text.approx_tokens(s), text.split_sentences(s), sorted(relevance.stems(s)), sorted(relevance._entities(s)),
        bool(relevance._DATE.search(s)), bool(relevance._TEMPORAL.search(s)), bool(relevance._HOW_MANY.search(s)),
        bool(relevance._FIRST_PERSON.search(s)),
    ]


def test_english_text_gives_the_frozen_results():
    rows = json.dumps([english_row(s) for s in ENGLISH], ensure_ascii=False)
    assert hashlib.sha256(rows.encode()).hexdigest() == ENGLISH_DIGEST


def test_english_questions_keep_their_shapes():
    assert relevance._WHEN.search("what year was that?") and relevance._WHEN.search("WHEN did it start")
    assert not relevance._WHEN.search("whenever you can")
    assert memory._HISTORICAL.search("what was the old port?") and not memory._HISTORICAL.search("what is the port?")


def test_the_english_stopwords_are_all_still_there():
    english = """a about above after again against all am an and any are aren as at be because been before being
    below between both but by can could did didn do does doesn doing don down during each few for from further had hadn
    has hasn have haven having he her here hers herself him himself his how i if in into is isn it its itself just ll me
    might more most mustn my myself no nor not now of off on once only or other our ours ourselves out over own re s
    same shan she should shouldn so some such t than that the their theirs them themselves then there these they this
    those through to too under until up us ve very was wasn we were weren what when where which while who whom why will
    with won would wouldn you your yours yourself yourselves""".split()
    assert {t for t in text.STOPWORDS if t.isascii()} == set(english)


@pytest.mark.parametrize("word, stem", SNOWBALL.items())
def test_stems_match_snowball(word, stem):
    assert russian_stem(word) == stem


def test_the_stem_goes_into_the_index_after_its_word():
    assert search_text("Запросы к базе") == "запросы запрос к базе баз"
    assert search_text("HTTP_запроса") == "http_запроса http запроса запрос"


def test_tokens_stay_as_written_because_a_stem_of_a_stem_can_differ():
    assert tokenize("австралия") == ["австралия"]
    assert russian_stem(russian_stem("австралия")) != russian_stem("австралия")


def test_the_query_asks_for_words_and_stems_once():
    assert fts_query(["запросы", "запросов", "запрос", "кэш"]) == '"запросы" OR "запросов" OR "запрос" OR "кэш"'


def test_words_outside_russian_get_no_stem():
    assert search_text("київ 2019г getПользователя") == "київ 2019г getпользователя"
    assert light_stem("київ") == "київ"


def test_russian_stopwords_stay_out_of_query_terms():
    assert content_terms("что такое кэш и где лежат настройки базы данных") == ["кэш", "лежат", "настройки", "базы", "данных"]
    assert content_terms("который из них") == []


def test_inflected_words_find_the_note():
    mem = Memory(":memory:")
    note = mem.write("Настройки базы данных лежат в файле config.yaml", scope="s", kind="semantic")
    mem.write("Сборка идёт через make", scope="s", kind="semantic")
    assert [i for i, _ in mem.lexical_search("настройками базе", 5, "s")] == [note]
    assert note in [item.id for item in mem.retrieve("где настройка базу", 200, scope="s").items]
    mem.close()


def test_a_word_with_yo_finds_the_same_word_without_it():
    mem = Memory(":memory:")
    note = mem.write("Зеленая ёлка стоит у входа", scope="s")
    assert [i for i, _ in mem.lexical_search("елку зелёную", 5, "s")] == [note]
    mem.close()


@pytest.mark.parametrize("word", ["австралия", "настройками", "базы", "знание", "прочитавшись"])
def test_facets_use_a_stem_the_index_holds(word):
    mem = Memory(":memory:")
    note = mem.write(f"Здесь есть слово {word} и другие", scope="s", kind="semantic")
    terms, weights = mem._facets(word, "s")
    assert terms == [russian_stem(word)] and weights.tolist() == [1.0]
    assert mem.db.execute("SELECT count(*) AS n FROM items_fts WHERE items_fts MATCH ?", (f'"{terms[0]}"',)).fetchone()["n"] == 1
    cand = mem.candidates(word, 0.0, "s", mem.config)[0]
    assert cand.ids == [note] and cand.term_cover.tolist() == [1.0]
    mem.close()


def test_excerpts_pick_the_sentence_with_the_inflected_word():
    long = "Первый абзац про совсем иное. " * 20 + "Настройки базы лежат в файле. " + "Последний абзац тоже про иное. " * 20
    excerpt, kept, total = text.query_window(long, [light_stem("настройками")], [1.0], 40, text.approx_tokens)
    assert "Настройки базы" in excerpt and kept == total == 1.0


@pytest.mark.parametrize(
    "pattern, query",
    [
        (relevance._TEMPORAL, "что изменилось за последнюю неделю"),
        (relevance._TEMPORAL, "Когда вышел релиз"),
        (relevance._TEMPORAL, "какой был первый запрос"),
        (relevance._TEMPORAL, "сколько лет проекту"),
        (relevance._TEMPORAL, "как долго идёт сборка"),
        (relevance._HOW_MANY, "Сколько тестов упало"),
        (relevance._HOW_MANY, "как часто падает сборка"),
        (relevance._HOW_MANY, "какой процент запросов"),
        (relevance._WHEN, "Когда это было"),
        (relevance._WHEN, "в каком году вышла книга"),
        (relevance._WHEN, "какого числа релиз"),
        (relevance._FIRST_PERSON, "Я запускаю тесты"),
        (relevance._FIRST_PERSON, "что мне делать"),
        (relevance._FIRST_PERSON, "МОЙ конфиг"),
        (relevance._DATE, "релиз 12 мая"),
        (relevance._DATE, "март 2023"),
        (relevance._DATE, "случилось в 2019г."),
        (relevance._DATE, "вчера вечером"),
        (relevance._DATE, "во вторник"),
        (memory._HISTORICAL, "какой был старый порт"),
        (memory._HISTORICAL, "что было раньше"),
        (memory._HISTORICAL, "предыдущая версия конфига"),
        (memory._HISTORICAL, "значение изменилось с 5 на 7"),
        (memory._HISTORICAL, "история изменений"),
    ],
)
def test_russian_query_shapes(pattern, query):
    assert pattern.search(query)


@pytest.mark.parametrize(
    "pattern, query",
    [
        (relevance._TEMPORAL, "датчик температуры"),
        (relevance._TEMPORAL, "годный вариант"),
        (relevance._TEMPORAL, "первичный ключ"),
        (relevance._HOW_MANY, "как настроить сборку"),
        (relevance._WHEN, "какой файл открыть"),
        (relevance._FIRST_PERSON, "мойка окон"),
        (relevance._FIRST_PERSON, "ямочный ремонт"),
        (relevance._DATE, "маршрут 5"),
        (relevance._DATE, "среди прочего"),
        (relevance._DATE, "неделимый блок"),
        (relevance._DATE, "выходные данные"),
        (memory._HISTORICAL, "старт сервера"),
        (memory._HISTORICAL, "стартовая страница"),
        (memory._HISTORICAL, "сначала запусти тесты"),
        (memory._HISTORICAL, "исторический район"),
    ],
)
def test_russian_lookalikes_are_not_query_shapes(pattern, query):
    assert not pattern.search(query)


def test_shapes_reach_the_features():
    def row(query: str, note: str) -> np.ndarray:
        zeros, ranks = np.zeros(1), np.array([0.0])
        return feature_matrix(query, [ItemView(note, 8, 0.0, "s")], ranks, ranks, zeros, zeros, zeros, zeros, zeros, zeros, zeros)[0]

    when, count, first = relevance.FEATURES.index("when_date"), relevance.FEATURES.index("count_number"), relevance.FEATURES.index("first_person_user")
    assert row("когда был релиз", "релиз вышел 12 мая")[when] == 1.0
    assert row("когда был релиз", "релиз вышел")[when] == 0.0
    assert row("сколько тестов упало", "упало 3 теста")[count] == 1.0
    assert row("что я говорил", "user: привет")[first] == 1.0


def old_store(path: Path) -> None:
    """A real file as a version without stems left it."""
    mem = Memory(path)
    for line in ("Настройки базы лежат в config.yaml", "The API timeout is 30 seconds", "Сборка идёт через make"):
        mem.write(line, scope="s", kind="semantic")
    for row in mem.db.execute("SELECT id, text FROM items").fetchall():
        mem.db.execute("UPDATE items_fts SET body=? WHERE rowid=?", (" ".join(tokenize(row["text"])), row["id"]))
    mem.db.execute("DELETE FROM meta WHERE key='schema_version'")
    mem.close()


def schema_version(path: Path) -> str | None:
    db = sqlite3.connect(path)
    row = db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    db.close()
    return row[0] if row else None


def test_an_old_store_gets_stems_once(tmp_path, monkeypatch):
    path = tmp_path / "m.sqlite"
    old_store(path)
    assert schema_version(path) is None
    db = sqlite3.connect(path)
    english = db.execute("SELECT rowid, body FROM items_fts WHERE body LIKE '%timeout%'").fetchone()
    assert not db.execute("SELECT 1 FROM items_fts WHERE items_fts MATCH '\"настройк\"'").fetchone()
    db.close()

    seen = []
    real = memory.search_text
    monkeypatch.setattr(memory, "search_text", lambda s: seen.append(s) or real(s))
    mem = Memory(path)
    assert sorted(seen) == ["Настройки базы лежат в config.yaml", "Сборка идёт через make"]
    assert len(mem.lexical_search("настройками", 5, "s")) == 1
    assert mem.db.execute("SELECT count(*) AS n FROM items_fts").fetchone()["n"] == 3
    assert tuple(mem.db.execute("SELECT rowid, body FROM items_fts WHERE body LIKE '%timeout%'").fetchone()) == english
    mem.db.execute("INSERT INTO items_fts (items_fts) VALUES ('integrity-check')")
    mem.close()
    assert schema_version(path) == str(SCHEMA_VERSION)

    seen.clear()
    mem = Memory(path)
    assert seen == []
    mem.close()


def test_a_new_store_needs_no_upgrade(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(memory, "search_text", lambda s: calls.append(s) or "")
    path = tmp_path / "m.sqlite"
    Memory(path).close()
    assert calls == [] and schema_version(path) == str(SCHEMA_VERSION)


def test_a_store_written_by_a_newer_version_is_left_alone(tmp_path, monkeypatch):
    path = tmp_path / "m.sqlite"
    old_store(path)
    db = sqlite3.connect(path)
    db.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION + 1),))
    db.commit()
    db.close()
    calls = []
    monkeypatch.setattr(memory, "search_text", lambda s: calls.append(s) or "")
    Memory(path).close()
    assert calls == [] and schema_version(path) == str(SCHEMA_VERSION + 1)


def test_processes_that_open_an_old_store_together_upgrade_it_once(tmp_path):
    path = tmp_path / "m.sqlite"
    old_store(path)
    code = "import sys; from kanerva.memory import Memory; Memory(sys.argv[1]).close()"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    jobs = [subprocess.Popen([sys.executable, "-c", code, str(path)], env=env, stderr=subprocess.PIPE) for _ in range(4)]
    assert all(job.wait() == 0 for job in jobs), [job.stderr.read() for job in jobs]
    db = sqlite3.connect(path)
    assert db.execute("SELECT count(*) FROM items_fts").fetchone()[0] == 3
    assert db.execute("SELECT count(*) FROM items_fts WHERE items_fts MATCH '\"настройк\"'").fetchone()[0] == 1
    db.close()


def test_a_busy_store_opens_and_upgrades_later(tmp_path, monkeypatch):
    path = tmp_path / "m.sqlite"
    old_store(path)
    other = sqlite3.connect(path)
    other.execute("BEGIN IMMEDIATE")
    connect = sqlite3.connect
    monkeypatch.setattr(memory.sqlite3, "connect", lambda name: connect(name, timeout=0.05))
    Memory(path).close()
    other.rollback()
    other.close()
    assert schema_version(path) is None
    monkeypatch.undo()
    Memory(path).close()
    assert schema_version(path) == str(SCHEMA_VERSION)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("KANERVA_HOME", str(tmp_path / "kanerva"))
    monkeypatch.setenv("KANERVA_EMBEDDER", "multilingual")
    return tmp_path


class Spawned:
    def __init__(self, monkeypatch):
        self.calls = []
        monkeypatch.setattr(embed.subprocess, "Popen", lambda args, **kwargs: self.calls.append(args))


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


def test_a_missing_multilingual_model_means_lexical_search_and_one_download(home, monkeypatch):
    spawned = Spawned(monkeypatch)
    for _ in range(3):
        mem = open_store()
        assert mem.embedder is None and mem.config.use_dense is False
        mem.close()
    assert len(spawned.calls) == 1 and spawned.calls[0][-3:] == ["-m", "kanerva", "fetch-model"]


def test_a_ready_multilingual_model_is_used(home, monkeypatch):
    for name in embed.MULTILINGUAL_FILES:
        path = multilingual_folder() / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    monkeypatch.setattr(embed, "MultilingualEmbedder", lambda: HashingEmbedder(385))
    mem = open_store()
    assert mem.embedder.dim == 385 and mem.config.use_dense is True
    mem.close()


def test_fetch_model_downloads_the_model_the_setting_names(home, monkeypatch):
    content = {name: name.encode() for name in embed.MULTILINGUAL_FILES}
    monkeypatch.setattr(embed, "MULTILINGUAL_FILES", {n: hashlib.sha256(d).hexdigest() for n, d in content.items()})
    calls = fake_hub(monkeypatch, content)
    assert fetch_minilm() == multilingual_folder() and multilingual_ready()
    assert {c[0] for c in calls} == {"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}
    assert {c[2] for c in calls} == {embed.MULTILINGUAL_REVISION} and len(embed.MULTILINGUAL_REVISION) == 40


def test_a_tampered_multilingual_file_is_rejected_and_removed(tmp_path, monkeypatch):
    fake_hub(monkeypatch, {name: b"tampered" for name in embed.MULTILINGUAL_FILES})
    with pytest.raises(ValueError):
        fetch_multilingual(tmp_path)
    assert not (tmp_path / embed.MULTILINGUAL_MODEL).exists()


def test_the_two_models_have_vectors_of_different_lengths():
    assert embed.MultilingualEmbedder.dim != embed.OnnxEmbedder.dim


def test_changing_models_does_not_mix_vectors(tmp_path):
    path = tmp_path / "m.sqlite"
    old = Memory(path, embedder=HashingEmbedder(embed.OnnxEmbedder.dim))
    old.write("The deploy goes through the staging branch", scope="s", kind="semantic")
    old.close()
    new = Memory(path, embedder=HashingEmbedder(embed.MultilingualEmbedder.dim))
    assert new._matrix("s")[0].size == 0
    assert new.reembed() == 1 and new._matrix("s")[0].size == 1
    new.close()


@pytest.mark.parametrize("copy", ["codex", "copilot", "hermes", "gemini", "qwen"])
def test_launcher_copies_match_their_source(copy):
    source = (ROOT / "plugins" / "claude-code" / "scripts" / "run.sh").read_text()
    assert "minilm|multilingual)" in source
    assert (ROOT / "plugins" / copy / "scripts" / "run.sh").read_text() == source


@pytest.mark.parametrize("copy", ["openclaw", "pi", "cline", "amp"])
def test_typescript_launcher_copies_match_their_source(copy):
    source = (ROOT / "plugins" / "opencode" / "src" / "core.ts").read_text()
    assert 'env.KANERVA_EMBEDDER === "multilingual"' in source
    assert (ROOT / "plugins" / copy / "src" / "core.ts").read_text() == source


model_folder = os.environ.get("KANERVA_TEST_MULTILINGUAL")
needs_model = pytest.mark.skipif(not model_folder, reason="set KANERVA_TEST_MULTILINGUAL to a folder with the model")


@needs_model
def test_a_russian_question_finds_an_english_note(tmp_path):
    pytest.importorskip("onnxruntime")
    mem = Memory(tmp_path / "m.sqlite", embedder=embed.MultilingualEmbedder(Path(model_folder)))
    mem.write("The nightly backup is stored in the S3 bucket", scope="s", kind="semantic")
    mem.write("Run the test suite with make test-all", scope="s", kind="semantic")
    best = mem.dense_search("где лежит ночная резервная копия?", 2, "s")
    assert mem.db.execute("SELECT text FROM items WHERE id=?", (best[0][0],)).fetchone()["text"].startswith("The nightly")
    mem.close()


@needs_model
def test_multilingual_vectors_are_normalised_with_a_zero_last_column():
    pytest.importorskip("onnxruntime")
    vectors = embed.MultilingualEmbedder(Path(model_folder)).encode(["Привет, мир", "Hello world, this one is longer"])
    assert vectors.shape == (2, 385) and vectors[:, -1].tolist() == [0.0, 0.0]
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


@needs_model
def test_a_vector_does_not_depend_on_the_texts_around_it():
    pytest.importorskip("onnxruntime")
    embedder = embed.MultilingualEmbedder(Path(model_folder))
    short, long = "Привет, мир", "Длинный текст, рядом с которым короткий получил бы другой вектор " * 4
    assert np.array_equal(embedder.encode([short]), embedder.encode([short, long])[:1])


@needs_model
def test_long_texts_are_truncated_as_the_model_expects():
    pytest.importorskip("onnxruntime")
    embedder = embed.MultilingualEmbedder(Path(model_folder))
    words = "быстрая коричневая лиса прыгает через ленивую собаку " * 20
    assert np.array_equal(embedder.encode([words * 40]), embedder.encode([words * 80]))
