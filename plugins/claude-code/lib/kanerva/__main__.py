"""``python -m kanerva``: hook entry points and the CLI."""

from __future__ import annotations

import json
import sys


def main() -> int:
    argv = sys.argv[1:]
    if argv[:1] == ["hook"]:
        from .hooks import main as hook_main

        return hook_main(argv[1:])
    if argv[:1] == ["hook-raw"]:
        # Host-neutral adapter entry: payload on stdin, context as plain text on stdout.
        from .hooks import handle

        try:
            context = handle(argv[1], json.load(sys.stdin))
            if context:
                sys.stdout.write(context)
        except Exception as exc:
            print(f"kanerva hook: {exc}", file=sys.stderr)
        return 0
    from .cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
