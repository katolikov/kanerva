---
name: remember
description: Save a durable fact, decision, convention or bug lesson to memory. Use when the user says to remember something, or after settling a decision worth keeping across sessions.
---

Call the `memory_remember` tool with one self-contained sentence in `text`.

Give a stable `key` when the fact may change later, for example `cmd:test`, `config:db-url` or `decision:auth`. A later write with the same key supersedes this one and keeps it as history. Use `scope: "global"` only for preferences that apply to every project. Never store secrets; the store redacts the ones it detects.
