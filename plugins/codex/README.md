# Kanerva for Codex

```bash
codex plugin marketplace add katolikov/kanerva
codex plugin add kanerva@kanerva
```

Start Codex, run `/hooks` and trust the Kanerva hooks. Until you do they are skipped without a message, so the tools work but no notes appear. Trust is tied to the plugin's contents, so an update asks again.

- Tested on Codex 0.166 with a local stub model, with hook trust bypassed: notes reached the model, a hook's context after a tool call reached it one step later, and the [session wrap-up](../../README.md#session-wrap-up) note was saved.
- Notes about a file a session touches come from `apply_patch`. A stub model is not offered that tool, so this trigger has not been run.
- Without trust, hooks do not run, which was checked. `/hooks` lists them for review, and approving them from that screen has not been run.
- A one-shot `codex exec` can exit before a background write finishes. Long sessions are not affected.
- Codex hides most of the environment from MCP servers. If you set `KANERVA_HOME`, also pass it to the `memory` server, or the tools use `~/.kanerva`.
- In `codex exec`, MCP calls need an approval policy that allows them.

The plugin has its own marketplace file, [`.agents/plugins/marketplace.json`](../../.agents/plugins/marketplace.json), which Codex reads before the Claude Code one.
