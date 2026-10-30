PY := .venv/bin/python
export PYTHONPATH := src:.

.PHONY: install test data reproduce figures paper check plugin fit

install:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev,experiments]"

test:
	.venv/bin/pytest -q
	cd plugins/opencode && npm ci && npm run typecheck && npm test

plugin:
	sh scripts/sync_plugin.sh

data:
	$(PY) scripts/fetch_data.py
	$(PY) -m experiments.repomem
	for name in lme locomo repomem repomem-holdout repomem-fresh beam; do \
	  $(PY) -m experiments.embed_cache $$name; \
	  $(PY) -m experiments.embed_cache $$name minishlab/potion-base-8M; \
	done

reproduce:
	$(PY) -m experiments.reproduce

figures:
	$(PY) -m experiments.figures

paper:
	$(PY) scripts/make_macros.py
	cd paper && ../.tools/tectonic paper.tex --keep-logs --reruns 3

check:
	$(PY) scripts/check_numbers.py

fit:
	$(PY) -m experiments.fit_relevance

tools:
	mkdir -p .tools
	curl -sL "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-aarch64-apple-darwin.tar.gz" | tar -xz -C .tools
