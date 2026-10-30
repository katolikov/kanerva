---
description: Forget or correct a memory. Use when the user says a memory is wrong, outdated, or should be removed.
disable-model-invocation: true
---

Find the item with `"${CLAUDE_PLUGIN_ROOT}/scripts/run.sh" search "$ARGUMENTS"` and show it to the user.

- To correct it: `"${CLAUDE_PLUGIN_ROOT}/scripts/run.sh" edit <id> "new text"`.
- To archive it, which is reversible: `"${CLAUDE_PLUGIN_ROOT}/scripts/run.sh" forget <id>`.
- To delete it permanently, only after the user confirms: `"${CLAUDE_PLUGIN_ROOT}/scripts/run.sh" forget <id> --hard`.
