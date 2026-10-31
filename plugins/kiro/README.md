# Kanerva for Kiro

```bash
curl -fsSL https://raw.githubusercontent.com/katolikov/kanerva/main/install.sh | sh -s -- kiro
```

The script clones Kanerva to `~/.kanerva/src`, writes `~/.kiro/hooks/kanerva.json` and merges the server into `~/.kiro/settings/mcp.json`.

Docs only: this follows Kiro's hook and MCP docs and has not been run. Global hooks in `~/.kiro/hooks/` need Kiro CLI V3. The docs do not list every stdin field, so the prompt is read from stdin and then from `USER_PROMPT`.
