"""Download the public benchmarks into data/. LoCoMo is CC BY-NC, so it is fetched, not redistributed."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

DATA = Path(__file__).resolve().parent.parent / "data"
LOCOMO = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


def main() -> None:
    (DATA / "locomo").mkdir(parents=True, exist_ok=True)
    target = DATA / "locomo" / "locomo10.json"
    if not target.exists():
        urllib.request.urlretrieve(LOCOMO, target)
    hf_hub_download("xiaowu0162/longmemeval-cleaned", "longmemeval_s_cleaned.json", repo_type="dataset",
                    local_dir=DATA / "longmemeval")


if __name__ == "__main__":
    main()
