---
name: memory
description: Inspect the local memory store. Use when the user asks what is remembered, wants to search memories, or wants store statistics.
---

Inspect memory with the `memory_*` tools and report what you find.

- Statistics: `memory_stats`
- Search: `memory_search` with the user's words in `query`
- Full text and sources of one item: `memory_expand` with its id

Lines read `[#id date] text`. A `superseded` mark means a newer version of the fact exists.
