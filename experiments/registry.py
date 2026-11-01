"""Dataset name to loader."""

from __future__ import annotations

from experiments.datasets import Stream, load_lme_updates, load_locomo, load_longmemeval


def load(name: str) -> list[Stream]:
    if name == "lme":
        return load_longmemeval()
    if name == "lme-updates":
        return load_lme_updates()
    if name == "locomo":
        return load_locomo()
    if name == "beam":
        from experiments.datasets import load_beam

        return load_beam()
    if name == "beam-dev":
        from experiments.datasets import load_beam

        return load_beam(("1M",))
    if name == "repomem-new":
        from experiments.repomem import load_repomem

        return load_repomem(split="new")
    if name == "repomem-final":
        from experiments.repomem import load_repomem

        return load_repomem(split="final")
    if name in ("repomem-c-dev", "repomem-c-test", "repomem-c-final", "repomem-c-last"):
        from experiments.repomem import load_repomem

        return load_repomem(split=name.removeprefix("repomem-"))
    if name == "repomem-other":
        from experiments.repomem import load_repomem

        return load_repomem(split="other")
    if name == "repomem-fresh":
        from experiments.repomem import load_repomem

        return load_repomem(split="fresh")
    if name == "repomem":
        from experiments.repomem import load_repomem

        return load_repomem()
    if name == "repomem-holdout":
        from experiments.repomem import load_repomem

        return load_repomem(holdout=True)
    raise ValueError(name)
