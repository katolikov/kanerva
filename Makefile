PY := .venv/bin/python
export PYTHONPATH := src:.

.PHONY: install test data reproduce reproduce-fast figures paper check plugin

install:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev,experiments]"

test:
	.venv/bin/pytest -q
	cd plugins/opencode && npm test

plugin:
	scripts/sync_plugin.sh

data:
	$(PY) scripts/fetch_data.py
	$(PY) -m experiments.repomem
	$(PY) -m experiments.embed_cache lme
	$(PY) -m experiments.embed_cache locomo
	$(PY) -m experiments.embed_cache repomem

reproduce:
	$(PY) -m experiments.reproduce

reproduce-fast:
	$(PY) -m experiments.reproduce --fast

figures:
	$(PY) -m experiments.analysis

paper:
	cd paper && ../.tools/tectonic -X compile paper.tex --keep-logs

check:
	$(PY) scripts/check_numbers.py
