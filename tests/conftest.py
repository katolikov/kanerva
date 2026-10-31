"""Tests never start a model download."""

import os

os.environ["KANERVA_EMBEDDER"] = "off"
os.environ.pop("CLAUDE_CODE_ENTRYPOINT", None)
