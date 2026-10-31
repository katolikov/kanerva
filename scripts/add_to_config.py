#!/usr/bin/env python3
"""Merges a Kanerva config template into an agent's JSON config and leaves the rest of the file alone."""

from __future__ import annotations

import json
import sys
from pathlib import Path

MARK = "scripts/run.sh"


def ours(entry) -> bool:
    return MARK in json.dumps(entry)


def merge(base, extra):
    if isinstance(base, dict) and isinstance(extra, dict):
        return {**base, **{key: merge(base[key], value) if key in base else value for key, value in extra.items()}}
    if isinstance(base, list) and isinstance(extra, list):
        return [entry for entry in base if not ours(entry)] + extra
    return extra


def install(template: Path, target: Path, runner: str) -> None:
    extra = json.loads(template.read_text(encoding="utf-8").replace("{{RUN}}", runner))
    base = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merge(base, extra), indent=2) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: add_to_config.py <template> <target> <runner>", file=sys.stderr)
        return 2
    template, target, runner = argv
    try:
        install(Path(template), Path(target).expanduser(), runner)
    except ValueError as exc:
        print(f"kanerva: {target} is not valid JSON, so it was left as it is: {exc}", file=sys.stderr)
        return 1
    print(f"Updated {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
