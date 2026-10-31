# Kanerva for Copilot CLI

```bash
copilot plugin marketplace add katolikov/kanerva
copilot plugin install kanerva@kanerva
```

Tested on Copilot CLI 1.0.105 with a local stub model.

- **Notes.** Copilot drops the output of `userPromptSubmitted`, so notes are added through `userPromptTransformed`. They reach the model at the end of the prompt, and the prompt you see stays the same. Notes also load at session start.
- **Tools.** The eight tools appear as `memory-memory_search` and so on. Copilot starts plugin servers in the plugin folder, so `.mcp.json` passes `${PWD}` to keep notes in your project. Start Copilot from the project folder.
- **Capture.** Edits, test commands and stops are recorded through `postToolUse` and `agentStop`.
- **File notes.** Notes about a file reach the model in the request right after the tool call that opened it, through `postToolUse`.

POSIX shells only. There is no PowerShell hook yet.
