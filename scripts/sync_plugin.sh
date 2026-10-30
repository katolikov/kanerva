#!/bin/sh
# Copies the package into the Claude Code plugin. With --check, fails if the copy is out of date.
set -e
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
SRC="$ROOT/src/kanerva"
DST="$ROOT/plugins/claude-code/lib/kanerva"
if [ "$1" = "--check" ]; then
  diff -r -x __pycache__ "$SRC" "$DST"
  exit 0
fi
mkdir -p "$DST"
rm -rf "$DST"
cp -R "$SRC" "$DST"
find "$DST" -name __pycache__ -type d -prune -exec rm -rf {} +
