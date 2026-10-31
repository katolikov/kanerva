#!/bin/sh
# Copies the package, the launcher and the TypeScript core into the plugins.
# With --check, fails if a copy is out of date.
set -e
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
SRC="$ROOT/src/kanerva"
RUN="$ROOT/plugins/claude-code/scripts/run.sh"
CORE="$ROOT/plugins/opencode/src/core.ts"

WITH_LIBRARY="claude-code opencode copilot openclaw"
WITH_LAUNCHER="copilot"
WITH_CORE="openclaw"

for NAME in $WITH_LIBRARY; do
  DST="$ROOT/plugins/$NAME/lib/kanerva"
  if [ "$1" = "--check" ]; then
    diff -r -x __pycache__ "$SRC" "$DST"
    continue
  fi
  rm -rf "$DST"
  mkdir -p "$(dirname "$DST")"
  cp -R "$SRC" "$DST"
  find "$DST" -name __pycache__ -type d -prune -exec rm -rf {} +
done

copy_file() {
  if [ "$1" = "--check" ]; then
    diff "$2" "$3"
    return
  fi
  mkdir -p "$(dirname "$3")"
  cp -p "$2" "$3"
}

for NAME in $WITH_LAUNCHER; do
  copy_file "${1:-}" "$RUN" "$ROOT/plugins/$NAME/scripts/run.sh"
done
for NAME in $WITH_CORE; do
  copy_file "${1:-}" "$CORE" "$ROOT/plugins/$NAME/src/core.ts"
done
