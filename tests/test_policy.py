import math

import numpy as np
import pytest

from kanerva.bandit import LinearBandit
from kanerva.secrets_filter import find_secrets, redact
from kanerva.text import approx_tokens, content_terms, split_identifier, tiktoken_counter
from kanerva.validity import HazardPosterior, activation_score, base_level_activation, supersession_survival


def test_bandit_recovers_a_linear_model():
    rng = np.random.default_rng(0)
    truth = np.array([0.2, -0.1, 0.5])
    bandit = LinearBandit(np.zeros(3), ridge=1.0)
    x = rng.normal(size=(2000, 3))
    bandit.update(x, x @ truth + 0.01 * rng.normal(size=2000))
    assert np.allclose(bandit.theta, truth, atol=0.01)


def test_bandit_prior_is_the_mean_before_any_data():
    theta0 = np.array([0.1, 0.2])
    assert np.allclose(LinearBandit(theta0).theta, theta0)


def test_bandit_incremental_matches_batch_ridge():
    rng = np.random.default_rng(1)
    theta0 = rng.normal(size=4)
    x = rng.normal(size=(30, 4))
    y = rng.normal(size=30)
    bandit = LinearBandit(theta0, ridge=2.0)
    bandit.update(x, y)
    v = 2.0 * np.eye(4) + x.T @ x
    expected = theta0 + np.linalg.solve(v, x.T @ (y - x @ theta0))
    assert np.allclose(bandit.theta, expected)


def test_bandit_bonus_shrinks_with_data_and_weights_stay_in_unit_interval():
    bandit = LinearBandit(np.array([0.5, 0.0]), ridge=1.0, beta=1.0)
    x = np.array([[1.0, 0.0]])
    before = bandit.score(x)[0]
    bandit.update(np.repeat(x, 50, axis=0), np.full(50, 0.5))
    assert bandit.score(x)[0] < before
    assert ((bandit.score(np.random.default_rng(2).normal(size=(50, 2))) >= 0) & (bandit.score(x) <= 1)).all()


def test_bandit_state_round_trip():
    bandit = LinearBandit(np.array([0.1, 0.2]), beta=0.3)
    bandit.update(np.eye(2), np.ones(2))
    copy = LinearBandit.from_state(bandit.state())
    assert np.allclose(copy.theta, bandit.theta) and copy.beta == 0.3


def test_lomax_survival_matches_formula_and_uses_evidence():
    posterior = HazardPosterior(alpha=2.0, beta=10.0)
    assert posterior.survival(0) == 1.0
    assert posterior.survival(10) == pytest.approx(2.0**-2)
    stable = HazardPosterior(alpha=1.0, beta=10.0, changes=0, exposure=10000)
    volatile = HazardPosterior(alpha=1.0, beta=10.0, changes=100, exposure=1000)
    assert stable.survival(30) > 0.99 > volatile.survival(30)


def test_lomax_tail_is_a_power_law():
    posterior = HazardPosterior(alpha=1.5, beta=1.0)
    slope = (math.log(posterior.survival(1e6)) - math.log(posterior.survival(1e5))) / math.log(10)
    assert slope == pytest.approx(-1.5, abs=1e-3)


def test_activation_rewards_recency_and_frequency():
    assert base_level_activation([1.0]) > base_level_activation([10.0])
    assert base_level_activation([1.0, 2.0]) > base_level_activation([1.0])
    assert 0.0 < activation_score([30.0]) < 1.0
    assert activation_score([]) == 0.0


def test_supersession_survival():
    assert supersession_survival([], 0.8) == 1.0
    assert supersession_survival([(1.0, 1.0)], 0.8) == pytest.approx(0.2)


@pytest.mark.parametrize(
    "secret",
    [
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
        "sk-proj-abcdefghijklmnopqrstuvwxyz012345",
        "xoxb-123456789012-abcdefghijkl",
        "postgres://admin:hunter22@db.example.com/prod",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTYifQ.abcDEF123456",
        "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----",
        "api_key = 'Zx9fQ2mK7pL4vR8s'",
    ],
)
def test_secrets_are_redacted(secret):
    text, kinds = redact(f"before {secret} after")
    assert kinds
    assert "REDACTED" in text
    assert secret not in text


@pytest.mark.parametrize(
    "safe",
    [
        "Use `pytest -x` to run the tests",
        "password = os.environ['DB_PASSWORD']",
        "token_budget = 4096",
        "The function get_user_token returns a session",
    ],
)
def test_ordinary_text_is_not_flagged(safe):
    assert find_secrets(safe) == []


def test_identifier_splitting_and_terms():
    assert split_identifier("parseHTTPResponse_v2") == ["parse", "http", "response", "v2"]
    terms = content_terms("What does parseHTTPResponse do with the body?")
    assert "parse" in terms and "body" in terms and "the" not in terms


def test_approx_tokens_tracks_tiktoken():
    count = tiktoken_counter()
    samples = [
        "The quick brown fox jumps over the lazy dog. " * 5,
        "def retrieve(self, query: str, budget: int) -> RetrievalResult:\n    return self.select(query, budget)\n",
        "commit 11798fd0: fix handling of empty headers (files: src/flask/app.py, tests/test_basic.py)",
    ]
    for text in samples:
        assert abs(approx_tokens(text) - count(text)) / count(text) < 0.35
