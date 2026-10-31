"""Verify that the paper's numbers come from results/*.json.

1. Regenerates paper/generated/ and the README results block, and fails if anything changes.
2. Fails if result-bearing sections contain literal decimals or percentages outside the allowlist.
3. Fails if any \\R or \\V key used in the paper is undefined.
4. Confirms that every result file records its commit.
"""

from __future__ import annotations

import filecmp
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ROOT / "paper" / "generated"
SECTIONS = ROOT / "paper" / "sections"
RESULT_SECTIONS = ["abstract.tex", "experiments.tex", "conclusion.tex", "limitations.tex", "summaries.tex"]
ALLOWED = {
    "0.316",  # the proved worst-case constant, (1 - 1/e) / 2
    "0.05",  # significance level
    "0.3", "0.5", "0.7", "0.8", "0.85", "0.9", "0.95", "0.97",  # thresholds and rates, which are inputs
    "0.4.1", "0.5.0",  # release numbers
}
ALLOWED_PERCENT = {"20\\%"}  # the reward noise, an input
NO_COMMIT = {"dataset_stats.json", "systems_mem0.json"}
MACRO = re.compile(r"\\R\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}|\\V\{([^}]*)\}")


def main() -> int:
    errors: list[str] = []
    backup = Path(tempfile.mkdtemp()) / "generated"
    shutil.copytree(GENERATED, backup)
    readme = (ROOT / "README.md").read_text()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_macros.py")], check=True)
    for path in sorted(backup.iterdir()):
        if not filecmp.cmp(path, GENERATED / path.name, shallow=False):
            errors.append(f"{path.name} is out of date against results/")
    if readme != (ROOT / "README.md").read_text():
        errors.append("README results block is out of date against results/")

    macros = (GENERATED / "macros.tex").read_text()
    checked = 0
    for path in sorted(SECTIONS.glob("*.tex")):
        text = path.read_text()
        if path.name in RESULT_SECTIONS:
            for match in re.finditer(r"(?<![\w.{-])\d+\.\d+(?:\.\d+)?(?![\w}])|\d+(?:\.\d+)?\\%", text):
                if match.group(0) not in ALLOWED | ALLOWED_PERCENT:
                    errors.append(f"{path.name}: literal number {match.group(0)!r}")
        for m in MACRO.finditer(text):
            key = f"v-{m.group(4)}" if m.group(4) is not None else f"r-{m.group(1)}-{m.group(2)}-{m.group(3)}"
            checked += 1
            if f"\\csname {key}\\endcsname" not in macros:
                errors.append(f"{path.name}: undefined macro key {key}")

    for path in sorted((ROOT / "results").glob("*.json")):
        if path.name not in NO_COMMIT and not json.loads(path.read_text()).get("meta", {}).get("commit"):
            errors.append(f"{path.name}: no commit recorded")

    for line in errors:
        print("FAIL", line)
    print(f"{checked} macro references checked, {len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
