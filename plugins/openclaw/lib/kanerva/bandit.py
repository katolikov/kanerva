"""Ridge linear bandit over item features with a prior mean (warm start)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinearBandit:
    """theta = theta0 + V^-1 b, with V = lambda I + sum x x^T and b = sum (y - x^T theta0) x."""

    theta0: np.ndarray
    ridge: float = 16.0
    beta: float = 0.0
    inv_v: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    updates: int = 0

    def __post_init__(self) -> None:
        d = self.theta0.shape[0]
        self.inv_v = np.eye(d) / self.ridge
        self.b = np.zeros(d)

    @property
    def theta(self) -> np.ndarray:
        return self.theta0 + self.inv_v @ self.b

    def score(self, x: np.ndarray, optimistic: bool = True) -> np.ndarray:
        """Clipped item weights for a feature matrix of shape (n, d)."""
        mean = x @ self.theta
        if optimistic and self.beta > 0:
            width = np.sqrt(np.einsum("ij,jk,ik->i", x, self.inv_v, x).clip(min=0.0))
            mean = mean + self.beta * width
        return np.clip(mean, 0.0, 1.0)

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Sherman-Morrison updates for the rows of x with rewards y."""
        for row, reward in zip(x, y):
            v = self.inv_v @ row
            self.inv_v -= np.outer(v, v) / (1.0 + row @ v)
            self.b += (reward - row @ self.theta0) * row
            self.updates += 1

    def state(self) -> dict:
        return {
            "theta0": self.theta0.tolist(),
            "ridge": self.ridge,
            "beta": self.beta,
            "inv_v": self.inv_v.tolist(),
            "b": self.b.tolist(),
            "updates": self.updates,
        }

    @classmethod
    def from_state(cls, state: dict) -> "LinearBandit":
        bandit = cls(np.array(state["theta0"]), state["ridge"], state["beta"])
        bandit.inv_v = np.array(state["inv_v"])
        bandit.b = np.array(state["b"])
        bandit.updates = state["updates"]
        return bandit
