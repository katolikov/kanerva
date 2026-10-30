"""Paired tests on the independent unit: sign-flip permutation, bootstrap intervals and Holm correction."""

from __future__ import annotations

import numpy as np

BOOT = 2000
PERMUTATIONS = 20000


def unit_of(name: str, stream: str) -> str | None:
    """The independent unit: a conversation, a repository, or, with ``None``, a question."""
    if name.startswith("repomem"):
        return stream.rsplit("-", 1)[0]
    return stream if name in ("locomo", "beam") else None


def sign_flip_p(d: np.ndarray, rng: np.random.Generator) -> float:
    """Two-sided sign-flip permutation test on the mean; exact for up to 16 units."""
    observed = abs(d.mean())
    n = d.size
    if observed == 0:
        return 1.0
    if n <= 16:
        signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        signs = rng.choice((-1.0, 1.0), size=(PERMUTATIONS, n))
    return float((np.abs((signs * d).mean(axis=1)) >= observed - 1e-12).mean())


def paired(a: dict, b: dict, rng: np.random.Generator, name: str) -> dict:
    keys = sorted(set(a) & set(b))
    x = np.array([a[k] for k in keys], dtype=float)
    y = np.array([b[k] for k in keys], dtype=float)
    d = x - y
    n = len(d)
    samples = d[rng.integers(0, n, size=(BOOT, n))].mean(axis=1)
    units = [unit_of(name, k[0]) for k in keys]
    cluster_ci = None
    positive = None
    if units[0] is not None:
        labels = sorted(set(units))
        groups = np.array([d[[i for i, u in enumerate(units) if u == c]].mean() for c in labels])
        sizes = np.array([sum(1 for u in units if u == c) for c in labels], dtype=float)
        picks = rng.integers(0, len(labels), size=(BOOT, len(labels)))
        boots = (groups[picks] * sizes[picks]).sum(axis=1) / sizes[picks].sum(axis=1)
        cluster_ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        positive = f"{int((groups > 0).sum())}/{len(labels)}"
        p = sign_flip_p(groups, rng)
    else:
        p = sign_flip_p(d, rng)
    return {"mean_a": float(x.mean()), "mean_b": float(y.mean()), "diff": float(d.mean()),
            "ci": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
            "cluster_ci": cluster_ci, "units_positive": positive, "p": p, "n": n}


def holm(entries: list[dict]) -> None:
    order = sorted(range(len(entries)), key=lambda i: entries[i]["p"])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, entries[i]["p"] * (len(order) - rank)))
        entries[i]["p_holm"] = running
