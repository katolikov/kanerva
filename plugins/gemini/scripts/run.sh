#!/bin/sh
# Runs the bundled kanerva package. uv supplies NumPy when it is available.
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/lib${PYTHONPATH:+:$PYTHONPATH}"
export KANERVA_PROJECT_DIR="${KANERVA_PROJECT_DIR:-${CLAUDE_PROJECT_DIR:-${DEVIN_PROJECT_DIR:-$PWD}}}"

# Run from the plugin folder, so that a project's own modules and .python-version stay out of the way.
cd "$ROOT" || exit 0

run() {
  if command -v uv >/dev/null 2>&1; then
    case "$KANERVA_EMBEDDER" in
      static) extras="--with model2vec>=0.6" ;;
      minilm) extras="--with onnxruntime>=1.18 --with tokenizers>=0.19" ;;
      *) extras="" ;;
    esac
    # shellcheck disable=SC2086
    uv run --quiet --no-project --with 'numpy>=1.26' $extras python -m kanerva "$@"
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
