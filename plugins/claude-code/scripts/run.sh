#!/bin/sh
# Runs the vendored kanerva package. Uses uv when present so that numpy is available without a global install.
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/lib${PYTHONPATH:+:$PYTHONPATH}"
if command -v uv >/dev/null 2>&1; then
  if [ "$KANERVA_EMBEDDER" = "static" ]; then
    exec uv run --quiet --no-project --with 'numpy>=1.26' --with 'model2vec>=0.6' python -m kanerva "$@"
  fi
  exec uv run --quiet --no-project --with 'numpy>=1.26' python -m kanerva "$@"
fi
exec python3 -m kanerva "$@"
