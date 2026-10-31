# Kanerva for Hermes

```bash
hermes plugins install katolikov/kanerva/plugins/hermes --enable
```

This adds notes to each turn and records edits and commands. Tested on Hermes 0.19.0 with a local stub model.

Notes that name a file a session touches are held for the next turn. Unit tests cover this, and it has not been run in a live session.

The eight tools come from Kanerva's MCP server. Hermes needs its `mcp` extra for that:

```bash
pip install 'hermes-agent[mcp]'
```

Then add this to `~/.hermes/config.yaml`, and run `/reload-mcp`:

```yaml
mcp_servers:
  kanerva:
    command: /home/you/.hermes/plugins/kanerva/scripts/run.sh
    args: [mcp]
```

Tools then show up as `mcp__kanerva__memory_search` and so on. Hermes keeps its own `memory` tool, and Kanerva does not replace it.
