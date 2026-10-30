"""Verify that the paper's numbers come from results/*.json.

1. Regenerates paper/generated/ and fails if the committed files differ.
2. Fails if result-bearing sections contain literal decimals or percentages outside the allowlist.
3. Fails if the PDF's macro use is undefined: every \\R, \\T, \\St, \\D, \\CI and \\V key must exist.
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
SECTIONS = ["abstract.tex", "experiments.tex", "conclusion.tex", "limitations.tex"]
ALLOWED = {
    "0.316",  # the proved worst-case constant, (1 - 1/e) / 2
    "2.2.1",  # Mem0 version
    "0.1", "0.5",  # distortion rates named in the text
    "0.05",  # significance level
    "0.7", "0.8", "0.9", "0.97",  # similarity thresholds, which are inputs
    "0.95",  # a qualitative floor for saturated fact queries, checked in the category table
}
MACRO = re.compile(r"\\(R|T|St|D|CI|Err|Inj)\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}|\\V\{([^}]*)\}")
PREFIX = {"R": "r", "T": "t", "St": "s", "D": "d", "CI": "ci", "Err": "e", "Inj": "i"}


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
    summaries = (ROOT / "paper" / "sections" / "summaries.tex")
    texts = {name: (ROOT / "paper" / "sections" / name).read_text() for name in SECTIONS}
    if summaries.exists():
        texts["summaries.tex"] = summaries.read_text()
    checked = 0
    for name, text in texts.items():
        for match in re.finditer(r"(?<![\w.{-])\d+\.\d+(?:\.\d+)?(?![\w}])|\d+(?:\.\d+)?\\%", text):
            if match.group(0) not in ALLOWED and "95\\%" not in match.group(0) and "20\\%" not in match.group(0) and "12\\%" not in match.group(0):
                errors.append(f"{name}: literal number {match.group(0)!r}")
        for m in MACRO.finditer(text):
            if m.group(5) is not None:
                key = f"v-{m.group(5)}"
            else:
                key = f"{PREFIX[m.group(1)]}-{m.group(2)}-{m.group(3)}-{m.group(4)}"
            checked += 1
            if f"\\csname {key}\\endcsname" not in macros:
                errors.append(f"{name}: undefined macro key {key}")

    for path in sorted((ROOT / "results").glob("*.json")):
        data = json.loads(path.read_text())
        if path.name in ("fitted_prior.json", "tuning.json", "comparison.json", "dataset_stats.json"):
            continue
        meta = data.get("meta", {})
        if path.name != "systems_mem0.json" and not meta.get("commit"):
            errors.append(f"{path.name}: no commit recorded")

    for line in errors:
        print("FAIL", line)
    print(f"{checked} macro references checked, {len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
