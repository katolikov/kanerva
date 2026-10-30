import itertools

import numpy as np
import pytest

from kanerva.select import SelectionProblem, budgeted_select, marginal_gains, optimality_certificate, value


def random_problem(rng, n=9, facets=4, gamma=0.3, modular=0.5):
    vectors = rng.normal(size=(n, 5))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    weights = rng.uniform(size=facets + 1)
    return SelectionProblem(
        weights=rng.uniform(size=n),
        costs=rng.integers(1, 12, size=n).astype(float),
        affinity=rng.uniform(size=(n, facets + 1)),
        facet_weights=weights / weights.sum(),
        similarity=np.clip(vectors @ vectors.T, 0.0, 1.0),
        prior_weights=rng.uniform(size=n),
        gamma=gamma,
        modular=modular,
    )


def brute_force(p, budget):
    best, best_set = 0.0, []
    n = p.weights.shape[0]
    for r in range(n + 1):
        for subset in itertools.combinations(range(n), r):
            if p.costs[list(subset)].sum() <= budget:
                v = value(p, list(subset))
                if v > best:
                    best, best_set = v, list(subset)
    return best, best_set


@pytest.mark.parametrize("seed", range(40))
def test_guarantee_and_certificate_against_brute_force(seed):
    rng = np.random.default_rng(seed)
    p = random_problem(rng)
    budget = float(rng.integers(5, 25))
    chosen = budgeted_select(p, budget, (1.0,))
    opt, _ = brute_force(p, budget)
    assert p.costs[chosen].sum() <= budget
    assert value(p, chosen) >= 0.5 * (1 - 1 / np.e) * opt - 1e-9
    assert optimality_certificate(p, chosen, budget) >= opt - 1e-9


@pytest.mark.parametrize("seed", range(10))
def test_objective_is_monotone_submodular(seed):
    rng = np.random.default_rng(seed)
    p = random_problem(rng, n=7)
    for _ in range(50):
        small = list(np.flatnonzero(rng.random(7) < 0.3))
        large = sorted(set(small) | set(np.flatnonzero(rng.random(7) < 0.5).tolist()))
        outside = [i for i in range(7) if i not in large]
        if not outside:
            continue
        i = outside[0]
        gain_small = value(p, small + [i]) - value(p, small)
        gain_large = value(p, large + [i]) - value(p, large)
        assert gain_large <= gain_small + 1e-9
        assert gain_large >= -1e-9


def test_marginal_gains_match_value_differences():
    rng = np.random.default_rng(3)
    p = random_problem(rng, n=6)
    chosen = [1, 4]
    miss = np.ones(p.affinity.shape[1])
    best = np.zeros(6)
    for i in chosen:
        miss = miss * (1 - p.weights[i] * p.affinity[i])
        best = np.maximum(best, p.similarity[i])
    gains = marginal_gains(p, miss, best)
    for i in (0, 2, 3, 5):
        assert gains[i] == pytest.approx(value(p, chosen + [i]) - value(p, chosen))


def test_more_cost_exponents_never_lower_the_objective():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        p = random_problem(rng)
        budget = 15.0
        assert value(p, budgeted_select(p, budget, (1.0, 0.5, 0.0))) >= value(p, budgeted_select(p, budget, (1.0,))) - 1e-12


def test_item_larger_than_budget_is_never_chosen():
    p = SelectionProblem(np.array([1.0, 0.1]), np.array([100.0, 5.0]), np.ones((2, 1)), np.ones(1))
    assert budgeted_select(p, 10.0) == [1]
    assert budgeted_select(p, 1.0) == []
