# Kanerva for Qwen Code

```bash
curl -fsSL https://raw.githubusercontent.com/katolikov/kanerva/main/install.sh | sh -s -- qwen
```

The script clones Kanerva to `~/.kanerva/src` and installs this folder, which you can also do by hand:

```bash
qwen extensions install ~/.kanerva/src/plugins/qwen --consent
```

Tested on Qwen Code 0.26.4 with a local stub model.

- **Notes** are added to each prompt.
- **Tools** are deferred: the model loads `mcp__memory__memory_search` and the rest through `tool_search`.
- **Capture** covers tool calls, stops and compaction.
