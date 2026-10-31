"""Temporal validity and activation models.

A fact of class k is assumed to stop holding at a hazard rate drawn from Gamma(alpha_k, beta_k).
After ``changes`` observed invalidations over ``exposure`` days the posterior is
Gamma(alpha + changes, beta + exposure), and the posterior predictive probability of surviving
another dt days is the Lomax tail (1 + dt / (beta + exposure)) ** -(alpha + changes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class HazardPosterior:
    alpha: float = 1.0
    beta: float = 180.0  # days
    changes: float = 0.0
    exposure: float = 0.0

    def survival(self, dt_days: float) -> float:
        scale = self.beta + self.exposure
        return (1.0 + max(dt_days, 0.0) / scale) ** -(self.alpha + self.changes)


def base_level_activation(ages_days: list[float], decay: float = 0.5) -> float:
    """ACT-R base-level activation B = ln(sum_j t_j^-d), ages floored at 1/24 day."""
    if not ages_days:
        return -math.inf
    return math.log(sum(max(age, 1 / 24) ** -decay for age in ages_days))


def activation_score(ages_days: list[float], decay: float = 0.5) -> float:
    """Map activation to (0, 1) with a fixed logistic centred at a 30-day-old single access."""
    a = base_level_activation(ages_days, decay)
    if a == -math.inf:
        return 0.0
    centre = -decay * math.log(30.0)
    return 1.0 / (1.0 + math.exp(-(a - centre)))


def supersession_survival(pairs: list[tuple[float, float]], strength: float) -> float:
    """Survival of an item given newer near-duplicates.

    ``pairs`` holds (similarity, gate) for each newer item, with gate in [0, 1]. Each is treated
    as an independent chance ``strength * similarity * gate`` of having replaced the item.
    """
    survive = 1.0
    for similarity, gate in pairs:
        survive *= 1.0 - strength * similarity * gate
    return survive
