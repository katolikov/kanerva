---
name: forget
description: Forget or correct a memory. Use when the user says a memory is wrong, outdated, or should be removed.
---

Find the item with `memory_search` and show it to the user.

- To correct it: `memory_edit` with its id and the new text.
- To mark it as replaced by a newer note, which keeps it as history: `memory_forget` with its id and `replaced_by` set to the newer id.
- To archive it, which is reversible: `memory_forget` with its id.
- To delete it permanently, only after the user confirms: `memory_forget` with its id and `hard: true`.
