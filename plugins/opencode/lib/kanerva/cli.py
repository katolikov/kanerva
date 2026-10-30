"""Command line: inspect, edit and maintain the memory store."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .consolidate import consolidate, expand, forget
from .memory import Memory, open_store
from .paths import project_scope


def _scope(args) -> str | None:
    if args.all:
        return None
    return "global" if args.scope == "global" else project_scope()


def export_markdown(mem: Memory, scope: str | None) -> str:
    lines = ["# Memory export", ""]
    for item in mem.list_items(scope, limit=100000):
        key = f" key={item.key}" if item.key else ""
        mark = " superseded" if item.superseded else ""
        lines.append(f"- [#{item.id} {item.scope} {item.kind}{key}{mark}] {item.text}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kanerva", description=__doc__)
    parser.add_argument("--scope", choices=["project", "global"], default="project")
    parser.add_argument("--all", action="store_true", help="all scopes")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stats")
    p = sub.add_parser("add")
    p.add_argument("text")
    p.add_argument("--key")
    p.add_argument("--pin", action="store_true")
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--budget", type=int, default=800)
    p = sub.add_parser("list")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--archived", action="store_true")
    p = sub.add_parser("show")
    p.add_argument("id", type=int)
    p = sub.add_parser("edit")
    p.add_argument("id", type=int)
    p.add_argument("text")
    p = sub.add_parser("forget")
    p.add_argument("id", type=int)
    p.add_argument("--hard", action="store_true", help="delete permanently")
    p = sub.add_parser("maintain")
    p.add_argument("--idle-days", type=float)
    p.add_argument("--capacity", type=int)
    p = sub.add_parser("export")
    p.add_argument("path", nargs="?")
    sub.add_parser("mcp", help="run the MCP server on stdio")
    args = parser.parse_args(argv)

    if args.command == "mcp":
        from .mcp_server import serve

        serve()
        return 0

    mem = open_store()
    try:
        scope = _scope(args)
        if args.command == "stats":
            print(json.dumps(mem.stats(), indent=1))
        elif args.command == "add":
            print(mem.write(args.text, scope=scope or "global", kind="semantic", key=args.key, pinned=args.pin))
        elif args.command == "search":
            print(mem.retrieve(args.query, args.budget, scope=scope, record=False).render())
        elif args.command == "list":
            tier = "archive" if args.archived else "active"
            for item in mem.list_items(scope, tier=tier, limit=args.limit):
                print(item.render())
        elif args.command == "show":
            print(json.dumps(expand(mem, args.id), indent=1, ensure_ascii=False))
        elif args.command == "edit":
            print(mem.edit(args.id, args.text))
        elif args.command == "forget":
            print(mem.delete([args.id]) if args.hard else mem.archive([args.id]))
        elif args.command == "maintain":
            report = consolidate(mem, scope=scope)
            archived = forget(mem, time.time(), args.idle_days, args.capacity, scope)
            print(json.dumps(report.__dict__ | {"archived": archived}, indent=1))
        elif args.command == "export":
            text = export_markdown(mem, scope)
            if args.path:
                Path(args.path).write_text(text)
            else:
                sys.stdout.write(text)
        mem.commit()
    finally:
        mem.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
