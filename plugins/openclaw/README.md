# Kanerva for OpenClaw

```bash
curl -fsSL https://raw.githubusercontent.com/katolikov/kanerva/main/install.sh | sh -s -- openclaw
```

The script clones Kanerva to `~/.kanerva/src`, then does three things you can also do by hand:

```bash
openclaw plugins install ~/.kanerva/src/plugins/openclaw --link --force --accept-capabilities
openclaw config set plugins.entries.kanerva.hooks.allowConversationAccess true --strict-json
openclaw mcp set kanerva '{"command":"'"$HOME"'/.kanerva/src/plugins/claude-code/scripts/run.sh","args":["mcp"]}'
```

`--link` is needed because the plugin ships TypeScript, and OpenClaw only runs that from a linked folder. Conversation access is what lets `before_prompt_build` run for a plugin that is not bundled. Kanerva does not take OpenClaw's memory slot, so it runs next to `memory-core`.

Tested on OpenClaw 2026.10.9 with a local stub model.

- **Notes** are prepended to each prompt.
- **Tools** are named `kanerva__memory_search` and so on, and the model finds them through `tool_search`. OpenClaw's own `memory_search` is a different tool.
- **Capture** covers tool calls, turn ends and the start of compaction. OpenClaw's compaction event has no summary, so compaction summaries are not stored.
- OpenClaw hides `KANERVA_HOME` from MCP servers. If you set it, add it under `env` in the `mcp set` command.

Needs Node 24.22.1 or newer, as OpenClaw does.
