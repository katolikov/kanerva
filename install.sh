#!/bin/sh
# Installs Kanerva for Claude Code, OpenCode or both.
#   curl -fsSL https://raw.githubusercontent.com/katolikov/kanerva/main/install.sh | sh -s -- opencode
set -eu

REPO="${KANERVA_REPO:-https://github.com/katolikov/kanerva.git}"
SRC="${KANERVA_SRC:-$HOME/.kanerva/src}"
TARGET="${1:-all}"

fetch() {
  command -v git >/dev/null 2>&1 || { echo "kanerva: git is required" >&2; exit 1; }
  if [ -d "$SRC/.git" ]; then
    git -C "$SRC" pull --quiet --ff-only
  else
    mkdir -p "$(dirname "$SRC")"
    git clone --quiet --depth 1 "$REPO" "$SRC"
  fi
}

check_python() {
  if ! command -v uv >/dev/null 2>&1 && ! python3 -c "import numpy" >/dev/null 2>&1; then
    echo "kanerva: install uv (https://docs.astral.sh/uv/) or NumPy for python3 first" >&2
    exit 1
  fi
}

install_opencode() {
  fetch
  dir="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/plugins"
  mkdir -p "$dir"
  printf 'export { default } from "%s/plugins/opencode/src/index.ts"\n' "$SRC" > "$dir/kanerva.ts"
  echo "Kanerva plugin added to $dir. Restart OpenCode to load it."
}

install_claude() {
  command -v claude >/dev/null 2>&1 || { echo "kanerva: the claude command was not found" >&2; exit 1; }
  claude plugin marketplace add katolikov/kanerva
  claude plugin install kanerva@kanerva
  echo "Kanerva plugin installed. Restart Claude Code to load it."
}

check_python
case "$TARGET" in
  opencode) install_opencode ;;
  claude) install_claude ;;
  all)
    command -v opencode >/dev/null 2>&1 && install_opencode
    command -v claude >/dev/null 2>&1 && install_claude
    ;;
  *)
    echo "usage: install.sh [claude|opencode|all]" >&2
    exit 2
    ;;
esac
