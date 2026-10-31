# Kanerva for the Cline CLI

```bash
curl -fsSL https://raw.githubusercontent.com/katolikov/kanerva/main/install.sh | sh -s -- cline
```

The script clones Kanerva to `~/.kanerva/src`, then runs:

```bash
cline plugin install ~/.kanerva/src/plugins/cline --force
cline mcp install kanerva --yes -- ~/.kanerva/src/plugins/claude-code/scripts/run.sh mcp
```

Tested on Cline CLI 3.0.74 with a local stub model.

- **Notes** are added at the start of each run.
- **Tools** appear as `kanerva__memory_search` and so on.
- **Capture** covers each command of `run_commands`, edits and run ends.

Cline plugins run in the SDK, the CLI and Kanban. The VS Code and JetBrains extensions do not load them, so there you only get the tools, through Cline's MCP settings.
