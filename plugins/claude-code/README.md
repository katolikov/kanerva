# Kanerva for Claude Code, Codex and Devin

This plugin serves three agents. Claude Code and Devin read `.claude-plugin/plugin.json` and `.mcp.json`. Codex reads `.codex-plugin/plugin.json` and `.codex-mcp.json`, because it does not expand `${CLAUDE_PLUGIN_ROOT}` in MCP commands.

## Codex

```bash
codex plugin marketplace add katolikov/kanerva
codex plugin add kanerva@kanerva
```

Start Codex, run `/hooks` and trust the Kanerva hooks. Until you do they are skipped without a message, so the tools work but no notes appear. Trust is tied to the plugin's contents, so an update asks again.

- Tested on Codex 0.166 with a local stub model: notes reached the model, and commands were recorded.
- A one-shot `codex exec` can exit before a background write finishes. Long sessions are not affected.
- Codex hides most of the environment from MCP servers. If you set `KANERVA_HOME`, also pass it to the `memory` server, or the tools use `~/.kanerva`.
- In `codex exec`, MCP calls need an approval policy that allows them.

## Devin

Docs only, not run. `install.sh devin` installs this folder as a plugin for the skills and tools, and adds the hooks to `~/.config/devin/config.json`, because Devin looks for plugin hooks in a different place than Claude Code.

## Claude Code

See the [main README](../../README.md#install).
