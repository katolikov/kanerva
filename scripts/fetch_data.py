"""Download the public benchmarks into data/. LoCoMo and BEAM have their own licenses, so they are fetched, not redistributed."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

DATA = Path(__file__).resolve().parent.parent / "data"
LOCOMO = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
BEAM = "https://huggingface.co/datasets/Mohammadta/BEAM/resolve/refs%2Fconvert%2Fparquet/default"


def main() -> None:
    (DATA / "locomo").mkdir(parents=True, exist_ok=True)
    target = DATA / "locomo" / "locomo10.json"
    if not target.exists():
        urllib.request.urlretrieve(LOCOMO, target)
    hf_hub_download("xiaowu0162/longmemeval-cleaned", "longmemeval_s_cleaned.json", repo_type="dataset",
                    local_dir=DATA / "longmemeval")
    (DATA / "beam").mkdir(parents=True, exist_ok=True)
    for split in ("100K", "500K", "1M"):
        target = DATA / "beam" / f"{split}.parquet"
        if not target.exists():
            urllib.request.urlretrieve(f"{BEAM}/{split}/0000.parquet", target)


if __name__ == "__main__":
    main()
