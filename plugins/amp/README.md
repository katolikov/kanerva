# Kanerva for Amp

```bash
curl -fsSL https://raw.githubusercontent.com/katolikov/kanerva/main/install.sh | sh -s -- amp
```

The script clones Kanerva to `~/.kanerva/src` and adds a small loader to `~/.config/amp/plugins/`. Run `plugins: reload` from the command palette afterwards. To remove it, delete `kanerva.ts` from that folder.

Follows the plugin API that `amp plugins show-docs` prints. Amp loads the plugin, but a full session needs an account, so injection has not been run. Please report what you see.

- **Notes** are added as a hidden message at the start of each turn.
- **Capture** covers tool results and turn ends.
- **Tools** are not included. Amp plugins work in the CLI only, not the editor extension, so add the [MCP server](../mcp/README.md) there.
