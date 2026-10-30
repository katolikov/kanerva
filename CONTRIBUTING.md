# Contributing

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,experiments]"
.venv/bin/pytest -q
```

## Rules for changes

- The memory layer stays local. It may not call a paid API or require an LLM.
- Experiments use the library as shipped. If a method needs a change, make it in `src/kanerva`.
- Every number in the paper comes from `results/*.json`. `scripts/check_numbers.py` enforces this.
- A retrieval change needs a before and after from `make reproduce-fast`, with the cases where it loses stated.
- Hyperparameters are tuned on dev splits only. Do not read test results while tuning.
- Citations must resolve. `scripts/verify_refs.py` checks them.
- After changing `src/kanerva`, run `scripts/sync_plugin.sh` so that the Claude Code plugin carries the same code.
