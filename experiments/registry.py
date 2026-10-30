"""Dataset name to loader."""

from __future__ import annotations

from experiments.datasets import Stream, load_locomo, load_longmemeval


def load(name: str) -> list[Stream]:
    if name == "lme":
        return load_longmemeval()
    if name == "locomo":
        return load_locomo()
    if name == "repomem":
        from experiments.repomem import load_repomem

        return load_repomem()
    if name == "repomem-holdout":
        from experiments.repomem import load_repomem

        return load_repomem(holdout=True)
    raise ValueError(name)
