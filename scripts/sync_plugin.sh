#!/bin/sh
# Copies the package into both plugins. With --check, fails if a copy is out of date.
set -e
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
SRC="$ROOT/src/kanerva"
for DST in "$ROOT/plugins/claude-code/lib/kanerva" "$ROOT/plugins/opencode/lib/kanerva"; do
  if [ "$1" = "--check" ]; then
    diff -r -x __pycache__ "$SRC" "$DST"
    continue
  fi
  rm -rf "$DST"
  mkdir -p "$(dirname "$DST")"
  cp -R "$SRC" "$DST"
  find "$DST" -name __pycache__ -type d -prune -exec rm -rf {} +
done
