# Kanerva for Gemini CLI

```bash
curl -fsSL https://raw.githubusercontent.com/katolikov/kanerva/main/install.sh | sh -s -- gemini
```

`gemini extensions install` has no option for a subfolder of a repository, so the script clones Kanerva to `~/.kanerva/src` and installs this folder from there. By hand:

```bash
gemini extensions install ~/.kanerva/src/plugins/gemini --consent --skip-settings
```

It asks you to trust the folder. The extension is copied, so update by pulling the clone and running `gemini extensions update kanerva`.

Tested on Gemini CLI 0.66.0 with a local stub model.

- **Notes** are added to each prompt through `BeforeAgent`, and at session start.
- **Tools** appear as `mcp_memory_memory_search` and so on.
- **Capture** covers tool calls, turn ends and compression. The summary after compression is not available to hooks.
