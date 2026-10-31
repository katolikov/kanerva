# Kanerva for Devin

```bash
curl -fsSL https://raw.githubusercontent.com/katolikov/kanerva/main/install.sh | sh -s -- devin
```

The script installs [`plugins/claude-code`](../claude-code) as a Devin plugin, for the skills and tools, and merges `hooks.json` into `~/.config/devin/config.json`.

Docs only: this follows Devin's plugin and hook docs and has not been run. Devin's hook payloads have no `cwd`, so the project comes from `DEVIN_PROJECT_DIR`.
