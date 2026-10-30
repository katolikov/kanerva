---
description: Inspect the local memory store. Use when the user asks what is remembered, wants to search memories, or wants store statistics.
---

Inspect memory with the CLI and report what you find.

- Statistics: `"${CLAUDE_PLUGIN_ROOT}/scripts/run.sh" stats`
- Recent items for this project: `"${CLAUDE_PLUGIN_ROOT}/scripts/run.sh" list --limit 30`
- Search: `"${CLAUDE_PLUGIN_ROOT}/scripts/run.sh" search "$ARGUMENTS"`
- Full text and sources of one item: `"${CLAUDE_PLUGIN_ROOT}/scripts/run.sh" show <id>`
- Readable export: `"${CLAUDE_PLUGIN_ROOT}/scripts/run.sh" export memory-export.md`

Lines read `[#id date] text`. A `superseded` mark means a newer version of the fact exists.
