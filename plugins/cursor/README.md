# Kanerva for Cursor

```bash
curl -fsSL https://raw.githubusercontent.com/katolikov/kanerva/main/install.sh | sh -s -- cursor
```

The script clones Kanerva to `~/.kanerva/src` and merges these templates into `~/.cursor/hooks.json` and `~/.cursor/mcp.json`. Other settings in those files are kept. If a file is not valid JSON it is left alone and the script says so.

Docs only: this follows Cursor's hook reference and has not been run.

- **Notes.** Cursor reads context from `sessionStart` and `postToolUse`, but not from `beforeSubmitPrompt`. So notes for a prompt are held and added to the first tool result of that turn. A reply that uses no tools gets none. Some users on Cursor's forum report that hook context is dropped, so check that it arrives.
- **Tools** come from the MCP server. User-level servers have no project folder variable, so check which project `memory_stats` reports. If it is the wrong one, add the server to the project's `.cursor/mcp.json` with `"env": {"KANERVA_PROJECT_DIR": "${workspaceFolder}"}`.

A plugin for the Cursor marketplace is not included yet, since publishing needs a review by Cursor.
