"""Store location and project scopes."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path


def home() -> Path:
    return Path(os.environ.get("KANERVA_HOME", Path.home() / ".kanerva")).expanduser()


def db_path() -> Path:
    return home() / "memory.sqlite"


def project_root(cwd: str | None = None) -> str:
    base = cwd or os.environ.get("KANERVA_PROJECT_DIR") or os.getcwd()
    try:
        out = subprocess.run(
            ["git", "-C", base, "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=2
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return str(Path(base).resolve())


def project_scope(cwd: str | None = None) -> str:
    root = project_root(cwd)
    return f"project:{Path(root).name}-{hashlib.sha1(root.encode()).hexdigest()[:8]}"
