"""Minimal MCP server over stdio (newline-delimited JSON-RPC 2.0), with no third-party dependencies."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable

from . import __version__
from .consolidate import consolidate, expand, forget
from .memory import Memory, open_store
from .paths import project_scope

PROTOCOL_VERSION = "2025-06-18"
SHORT_NOTE_CHARS = 120  # of a similar note, which the agent pays for on each write

TOOLS: dict[str, dict] = {
    "memory_search": {
        "description": "Retrieve the memories most useful for a query under a token budget.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "budget": {"type": "integer", "default": 800}},
            "required": ["query"],
        },
    },
    "memory_remember": {
        "description": (
            "Store a durable fact, decision, convention or bug lesson. Give a stable `key` such as "
            "'cmd:test' or 'config:db-url' when the fact may change later: a new write with the same key "
            "supersedes the old one and keeps it as history."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "key": {"type": "string"},
                "scope": {"type": "string", "enum": ["project", "global"], "default": "project"},
                "pinned": {"type": "boolean", "default": False},
            },
            "required": ["text"],
        },
    },
    "memory_expand": {
        "description": "Show the full text of a memory id, with its archived sources.",
        "inputSchema": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
    },
    "memory_feedback": {
        "description": "Report which memory ids from a retrieval were actually useful.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "retrieval_id": {"type": "integer"},
                "used_ids": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["retrieval_id", "used_ids"],
        },
    },
    "memory_edit": {
        "description": "Replace the text of a memory id.",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
            "required": ["id", "text"],
        },
    },
    "memory_forget": {
        "description": (
            "Archive a memory id, or delete it permanently with `hard`. With `replaced_by`, the id of a newer note, "
            "it is kept as history instead, and `hard` is ignored."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "hard": {"type": "boolean", "default": False},
                "replaced_by": {"type": "integer"},
            },
            "required": ["id"],
        },
    },
    "memory_maintain": {
        "description": "Consolidate near-duplicate episodic items. With `idle_days`, also archive items idle that long.",
        "inputSchema": {"type": "object", "properties": {"idle_days": {"type": "number"}}},
    },
    "memory_stats": {
        "description": "Show store size and supersession counts.",
        "inputSchema": {"type": "object", "properties": {}},
    },
}


def short_note(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SHORT_NOTE_CHARS else text[:SHORT_NOTE_CHARS].rstrip() + "..."


INSTRUCTIONS = (
    "Long-term memory for this project, kept across sessions. When the user asks you to remember something, or a decision "
    "or lesson should outlast this session, save it with memory_remember and not in a file: later sessions get it ranked "
    "under a token budget, and a newer note can replace it. Keeping memory current is part of saving, and installing this "
    "server asks for it: when memory_remember lists a note that yours makes out of date, retire or correct that note in the "
    "same turn, as the result says."
)


class Server:
    def __init__(self, mem: Memory | None = None, scope: str | None = None):
        self.mem = mem or open_store()
        self.scope = scope or project_scope(os.environ.get("KANERVA_PROJECT_DIR"))

    def call(self, name: str, args: dict) -> Any:
        """Run one tool and commit, since a long-lived open transaction would lock out the hooks."""
        try:
            return self._call(name, args)
        finally:
            self.mem.commit()

    def _call(self, name: str, args: dict) -> Any:
        if name == "memory_search":
            result = self.mem.retrieve(args["query"], int(args.get("budget", 800)), scope=self.scope)
            return {
                "retrieval_id": result.retrieval_id,
                "tokens": result.tokens,
                "memories": [{"id": i.id, "text": i.text, "superseded": i.superseded} for i in result.items],
            }
        if name == "memory_remember":
            scope = "global" if args.get("scope") == "global" else self.scope
            item = self.mem.write(
                args["text"], scope=scope, kind="semantic", key=args.get("key"), pinned=bool(args.get("pinned"))
            )
            reply: dict = {"id": item}
            similar = [] if item is None else self.mem.similar(args["text"], scope, exclude=(item,))
            if similar:
                reply["similar"] = [{"id": note.id, "text": short_note(note.text)} for note in similar]
                reply["hint"] = (
                    "A listed note with an older value of the same fact is now out of date. Fixing it is part of saving "
                    f"this note, so do it now without asking. If that fact is all it says, call memory_forget with its id and "
                    f"replaced_by={item}. If it says other things too, call memory_edit with its id and its text without the "
                    "outdated part. Leave notes about anything else alone."
                )
            return reply
        if name == "memory_expand":
            return expand(self.mem, int(args["id"]))
        if name == "memory_feedback":
            used = {int(i): 1.0 for i in args["used_ids"]}
            return {"updated": self.mem.feedback(int(args["retrieval_id"]), used)}
        if name == "memory_edit":
            return {"edited": self.mem.edit(int(args["id"]), args["text"])}
        if name == "memory_forget":
            item = int(args["id"])
            if args.get("replaced_by") is not None:
                return {"replaced": self.mem.supersede(item, int(args["replaced_by"]))}
            removed = self.mem.delete([item]) if args.get("hard") else self.mem.archive([item])
            return {"removed": removed}
        if name == "memory_maintain":
            import time

            report = consolidate(self.mem, scope=self.scope)
            idle = args.get("idle_days")
            archived = forget(self.mem, time.time(), idle_days=None if idle is None else float(idle), scope=self.scope)
            return {"clusters": report.clusters, "sources": report.sources, "archived": archived}
        if name == "memory_stats":
            return self.mem.stats()
        raise KeyError(name)

    def handle(self, message: dict) -> dict | None:
        method = message.get("method")
        request_id = message.get("id")
        if request_id is None:
            return None
        if method == "initialize":
            result: Any = {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "kanerva", "version": __version__},
                "instructions": INSTRUCTIONS,
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": [{"name": n} | spec for n, spec in TOOLS.items()]}
        elif method == "tools/call":
            params = message.get("params", {})
            try:
                value = self.call(params.get("name", ""), params.get("arguments") or {})
                result = {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]}
            except Exception as exc:
                result = {"content": [{"type": "text", "text": f"error: {exc}"}], "isError": True}
        else:
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"unknown method {method}"}}
        return {"jsonrpc": "2.0", "id": request_id, "result": result}


def serve(reader=sys.stdin, writer: Callable[[str], Any] | None = None) -> None:
    server = Server()
    write = writer or (lambda line: (sys.stdout.write(line + "\n"), sys.stdout.flush()))
    for line in reader:
        line = line.strip()
        if not line:
            continue
        try:
            response = server.handle(json.loads(line))
        except json.JSONDecodeError:
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        if response is not None:
            write(json.dumps(response))


if __name__ == "__main__":
    serve()
