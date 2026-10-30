PY := .venv/bin/python
export PYTHONPATH := src:.

.PHONY: install test data reproduce reproduce-fast figures paper check plugin fit

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
	$(PY) -m experiments.embed_cache repomem-holdout
	$(PY) -m experiments.embed_cache lme minishlab/potion-base-8M
	$(PY) -m experiments.embed_cache locomo minishlab/potion-base-8M
	$(PY) -m experiments.embed_cache repomem minishlab/potion-base-8M

reproduce:
	$(PY) -m experiments.reproduce

reproduce-fast:
	$(PY) -m experiments.reproduce --fast

figures:
	$(PY) -m experiments.analysis

paper:
	$(PY) scripts/make_macros.py
	cd paper && ../.tools/tectonic paper.tex --keep-logs --reruns 3

check:
	$(PY) scripts/check_numbers.py

fit:
	$(PY) -m experiments.fit_prior

tools:
	mkdir -p .tools
	curl -sL "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-aarch64-apple-darwin.tar.gz" | tar -xz -C .tools
