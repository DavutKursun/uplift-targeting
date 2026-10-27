"""Check the Criteo uplift file is in data/raw/ and explain how to get it. Never asks for credentials.

Usage: python scripts/download.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.config import KAGGLE_DATASET, RAW, RAW_CANDIDATES
from uplift.data import find_raw

HOW = f"""
No raw file found in {RAW} (looked for {', '.join(RAW_CANDIDATES)}).

Two ways to get the Criteo Uplift Prediction Dataset v2.1 (your Kaggle / Criteo login, never shared in a chat):
  1) Kaggle mirror: https://www.kaggle.com/datasets/{KAGGLE_DATASET} → Data → download the CSV (≈ 3 GB) → data/raw/
     or: kaggle datasets download -d {KAGGLE_DATASET} -p data/raw/ && unzip in data/raw/
  2) Criteo AI Lab: https://ailab.criteo.com/criteo-uplift-prediction-dataset/ → criteo-research-uplift-v2.1.csv.gz (≈ 300 MB) → data/raw/
     (its go.criteo.net link returned 404 on 2026-10-27)
  3) Criteo's own Hugging Face mirror of the same file (311 MB, CC BY-NC-SA 4.0, no login):
     curl -L -o data/raw/criteo-research-uplift-v2.1.csv.gz \
       https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz
Then: python scripts/build_splits.py
"""


def main() -> int:
    p = find_raw()
    if p is None:
        print(HOW)
        return 2
    print(f"found {p.name}: {p.stat().st_size / 1e6:.0f} MB")
    print("next: python scripts/build_splits.py   (first run converts the CSV to parquet in chunks: a few minutes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
