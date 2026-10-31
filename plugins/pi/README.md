# Kanerva for Pi

```bash
curl -fsSL https://raw.githubusercontent.com/katolikov/kanerva/main/install.sh | sh -s -- pi
```

Or by hand:

```bash
pi install git:github.com/katolikov/kanerva
uv tool install 'kanerva[onnx] @ git+https://github.com/katolikov/kanerva'
pi mcp add memory --exposure direct -- kanerva mcp
```

Tested on Pi 1.4.0 with a local stub model.

- **Notes** go in as a hidden message before each run.
- **Tools** need `--exposure direct`. Without it they are only reachable from `codemode` scripts.
- **Capture** covers tool results, run ends and compaction summaries.

Run `/reload` in Pi after installing.
