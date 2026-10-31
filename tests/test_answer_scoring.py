from types import SimpleNamespace

import pytest

from experiments.answer_test import (
    SETS,
    contains,
    gold_forms,
    old_values,
    said_unknown,
    sample,
    scorable,
    token_f1,
    version_given,
)


def query(question="", answer="", category="fact-constant-changed", qid="q", stale=()):
    return SimpleNamespace(question=question, answer=answer, category=category, qid=qid, stale=set(stale))


def test_prose_ignores_case_order_articles_and_number_words():
    assert contains("Sunsets.", "sunsets")
    assert contains("May 7, 2023", "7 May 2023")
    assert contains("Two", "2")
    assert contains("the Coconut milk and sugar", "sugar, coconut milk")
    assert contains("2023-07-21", "21 July 2023")
    assert contains("July 2023", "2023-07-21") is False
    assert contains("December 17", "December 17th")
    assert not contains("coconut milk", "coconut milk, sugar")


def test_words_of_the_question_do_not_count():
    question = "What is the deadline for completing gift purchases?"
    gold = "The deadline for gift purchases is December 17th."
    assert contains("December 17th", gold, question)
    assert token_f1("December 17th", gold, question) == 1.0
    assert token_f1("December 18th", gold, question) == pytest.approx(0.5)


def test_a_gold_inside_the_question_still_counts():
    question = "Would she prefer a national park or a theme park?"
    assert contains("national park", "National park", question)
    assert not contains("the theme park", "National park", question)


def test_code_is_compared_as_one_run_of_tokens():
    assert contains("It is 10.", "10", code=True)
    assert not contains("100", "10", code=True)
    assert contains("`src/requests/utils.py`.", "src/requests/utils.py", code=True)
    assert contains(">=1.21.1, <1.27", ">=1.21.1,<1.27", code=True)
    assert contains('"__attr_factory_%s"', "'__attr_factory_%s'", code=True)
    assert not contains("params=None, url", "url, params=None", code=True)


def test_f1_of_an_unrelated_or_empty_answer_is_zero():
    assert token_f1("purple", "sunsets") == 0.0
    assert token_f1("", "sunsets") == 0.0


def test_refusals():
    assert said_unknown("Unknown.")
    assert said_unknown("")
    assert said_unknown("The notes do not contain this.")
    assert said_unknown("I don’t know")
    assert said_unknown("The song is not named in the notes.")
    assert said_unknown("The notes do not specify the frequency.")
    assert not said_unknown("Sunsets")


@pytest.mark.parametrize(
    "answer, expected",
    [
        (">=1.2,<2", "current"),
        (">=1.2", "old"),
        ("It was >=1.2 and is now >=1.2,<2", "both"),
        ("unknown", "neither"),
    ],
)
def test_a_value_inside_a_longer_value_is_not_an_answer(answer, expected):
    assert version_given(answer, ">=1.2,<2", [">=1.2"]) == expected


def test_the_longer_old_value_wins_too():
    assert version_given("a, b=1", "a", ["a, b=1"]) == "old"


def test_alternative_gold():
    assert gold_forms("26 days. 27 days (including the last day) is also acceptable.") == ["26 days", "27 days"]
    assert gold_forms("Sunsets") == ["Sunsets"]


def test_old_values_skip_a_move_that_keeps_the_value():
    moved = query(answer="a, b", category="fact-signature-changed", stale=["f(a, b) is the signature, defined in old.py"])
    assert old_values(moved) == []
    changed = query(answer="a, b", category="fact-signature-changed", stale=["f(a) is the signature, defined in new.py"])
    assert old_values(changed) == ["a"]
    assert old_values(query(answer="3", stale=["X in a.py is set to 2"])) == ["2"]
    assert old_values(query(answer="a/new.py", category="fact-location-changed", stale=["X is defined in a/old.py"])) == ["a/old.py"]
    assert old_values(query(answer=">=2", category="fact-dependency-changed", stale=["The project requires X >=1"])) == [">=1"]


def test_what_can_be_scored():
    locomo, lme, repomem = SETS["locomo"], SETS["lme"], SETS["repomem"]
    assert scorable(query("Q?", "Sunsets", "1"), locomo)
    assert not scorable(query("Q?", "Yes; it's classical music", "3"), locomo)
    assert not scorable(query("Q?", "", "1"), locomo)
    assert not scorable(query("Q?", " ".join(f"w{i}" for i in range(9)), "1"), locomo)
    assert not scorable(query("Q?", "17", "multi-session", "x_abs"), lme)
    assert scorable(query("Q?", "26 days. 27 days (including the last day) is also acceptable.", "multi-session"), lme)
    assert not scorable(query("Q?", "2023-07-21", "2"), locomo)
    assert not scorable(query("Q?", "26 days", "temporal-reasoning"), lme)
    assert not scorable(query("Q?", "=", "fact-dependency-stable"), repomem)
    assert not scorable(query("Q?", "x" * 121, "fact-signature-stable"), repomem)
    assert scorable(query("Q?", "src/a.py", "fact-location-stable"), repomem)


def test_sample_is_stable_and_a_smaller_one_is_a_prefix():
    streams = [
        SimpleNamespace(name=f"s{i}", queries=[query("Q?", "x", c, f"{c}-{k}") for c in ("1", "4") for k in range(5)])
        for i in range(4)
    ]
    large = sample(streams, SETS["locomo"], 20)
    assert large == sample(streams, SETS["locomo"], 20)
    assert len(set(large)) == 20
    assert sample(streams, SETS["locomo"], 8) == large[:8]
