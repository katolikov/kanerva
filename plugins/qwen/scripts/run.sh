#!/bin/sh
# Runs the bundled kanerva package. uv supplies NumPy when it is available.
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/lib${PYTHONPATH:+:$PYTHONPATH}"
export KANERVA_PROJECT_DIR="${KANERVA_PROJECT_DIR:-${CLAUDE_PROJECT_DIR:-${DEVIN_PROJECT_DIR:-$PWD}}}"

# Run from the plugin folder, so that a project's own modules and .python-version stay out of the way.
cd "$ROOT" || exit 0

run() {
  if command -v uv >/dev/null 2>&1; then
    if [ "$KANERVA_EMBEDDER" = "static" ]; then
      uv run --quiet --no-project --with 'numpy>=1.26' --with 'model2vec>=0.6' python -m kanerva "$@"
    else
      uv run --quiet --no-project --with 'numpy>=1.26' python -m kanerva "$@"
    fi
  else
    python3 -m kanerva "$@"
  fi
}

case "$1" in
  hook|hook-host)
    # A memory failure must never block the agent.
    run "$@" 2>/dev/null || true
    exit 0
    ;;
esac
run "$@"
