# Kanerva for OpenCode

Long-term memory for OpenCode. Notes from earlier sessions come back when they matter, under a hard token budget, with no LLM calls and no data leaving your machine.

Tested on OpenCode 1.19 and 2.2. Needs [uv](https://docs.astral.sh/uv/) or `python3` with NumPy.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/katolikov/kanerva/main/install.sh | sh -s -- opencode
```

The script clones Kanerva to `~/.kanerva/src` and adds a small loader to `~/.config/opencode/plugins/`. Restart OpenCode and you are done. To update, run it again. To remove it, delete `~/.config/opencode/plugins/kanerva.ts`.

On V2 you can also install it as a package:

```bash
opencode plugin add 'github:katolikov/kanerva#main::path:plugins/opencode'
```

Or point your config at a checkout:

```jsonc
// opencode.jsonc (V2)
{ "plugins": ["/path/to/kanerva/plugins/opencode"] }
```

## What you get

- **Context on every request.** Notes from earlier sessions are added to the system prompt, for the session and for each prompt.
- **Automatic capture.** Edits, test commands and compaction summaries are stored as you work. Secrets are redacted first.
- **Eight tools.** `memory_search`, `memory_remember`, `memory_expand`, `memory_edit`, `memory_forget`, `memory_feedback`, `memory_maintain` and `memory_stats` are available to the model with no MCP setup.

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `KANERVA_EMBEDDER` | off | Set to `static` for vector search with model2vec. |
| `KANERVA_PROMPT_BUDGET` | `600` | Tokens injected per prompt. |
| `KANERVA_SESSION_BUDGET` | `400` | Tokens injected at session start. |
| `KANERVA_HOME` | `~/.kanerva` | Where the database lives. |
| `KANERVA_COMMAND` | auto | Command that starts Kanerva, such as `uvx kanerva`. |
| `KANERVA_TIMEOUT_MS` | `20000` | Give up on a memory call after this long. |

The plugin looks for `KANERVA_COMMAND`, then a `kanerva` command on your `PATH`, then runs its bundled copy through `uv` or `python3`. If memory ever fails, OpenCode carries on without it.

## Development

```bash
npm test
npm run typecheck
```
