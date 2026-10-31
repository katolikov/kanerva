# MCP setup for Goose, Zed and other clients

These clients have no hook that can add notes on their own, so the agent has to call the tools itself. Start by installing the `kanerva` command:

```bash
uv tool install 'kanerva[onnx] @ git+https://github.com/katolikov/kanerva'
```

Tools only, and docs only: these configs follow each client's documentation and have not been run.

## Goose

In `~/.config/goose/config.yaml`:

```yaml
extensions:
  kanerva:
    name: kanerva
    type: stdio
    cmd: kanerva
    args: [mcp]
    enabled: true
    timeout: 300
```

Put the rules below in `AGENTS.md` or `.goosehints`.

## Zed

In `settings.json`:

```json
{
  "context_servers": {
    "kanerva": { "command": "kanerva", "args": ["mcp"] }
  }
}
```

Put the rules below in `AGENTS.md` or `.rules`.

## Rules for the agent

```markdown
This project has long-term memory through the memory_* tools.
- Before you start a task, call memory_search with what you are about to do.
- When you learn a decision, a convention, a command or the cause of a bug, call memory_remember with one short fact. Give it a key if it may change later.
- If memory_remember lists a similar note that yours replaces, call memory_forget with that id and replaced_by set to your note's id.
- If a note turns out to be wrong, fix it with memory_edit or memory_forget.
```

## Which project notes go to

The server uses the folder it starts in. If notes land under the wrong project, set `KANERVA_PROJECT_DIR` in the server's environment. Check with `memory_stats`.
