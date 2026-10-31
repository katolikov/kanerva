PY := .venv/bin/python
TECTONIC_TARGET := $(if $(filter Darwin,$(shell uname -s)),aarch64-apple-darwin,x86_64-unknown-linux-gnu)
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
	for name in lme locomo repomem repomem-holdout repomem-fresh repomem-new repomem-other repomem-final repomem-c-dev repomem-c-test repomem-c-final beam; do \
	  $(PY) -m experiments.embed_cache $$name; \
	  $(PY) -m experiments.embed_cache $$name minishlab/potion-base-8M; \
	done
	git show 63e6b25:src/kanerva/relevance_model.json > experiments/cache/relevance_model_0.4.json

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
	curl -sL "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-$(TECTONIC_TARGET).tar.gz" | tar -xz -C .tools
