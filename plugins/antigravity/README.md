# Kanerva for Antigravity

```bash
curl -fsSL https://raw.githubusercontent.com/katolikov/kanerva/main/install.sh | sh -s -- antigravity
```

The script clones Kanerva to `~/.kanerva/src` and merges these templates into `~/.gemini/config/hooks.json` and `~/.gemini/config/mcp_config.json`.

Docs only: this follows Antigravity's hook docs and has not been run.

Notes load at session start only. `PreInvocation` can inject context, but its input does not include the prompt, so there is nothing to search with. Use `memory_search` for the rest, and add the rules snippet from the [MCP setup](../mcp/README.md) to help the agent remember it.
