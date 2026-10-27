"""Raw CSV → parquet cache (chunked) → stratified subsample → train/dev/test parquet + summary.json.

Usage: python scripts/build_splits.py [--raw FILE] [--out data/splits] [--rows 4000000 | --full] [--cache data/raw/criteo.parquet]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.config import RAW, SPLIT_RULES, SPLITS
from uplift.data import COLUMNS, assign_splits, csv_to_parquet, find_raw, save_splits, stratified_subsample


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=SPLITS)
    ap.add_argument("--cache", type=Path, default=RAW / "criteo.parquet")
    ap.add_argument("--rows", type=int, default=SPLIT_RULES.subsample_rows)
    ap.add_argument("--full", action="store_true", help="use every row")
    a = ap.parse_args(argv)
    raw = a.raw or find_raw()
    if raw is None or not Path(raw).exists():
        print("raw file not found — run scripts/download.py")
        return 2
    if not a.cache.exists():
        print(f"converting {raw} → {a.cache} in chunks …", flush=True)
        n = csv_to_parquet(Path(raw), a.cache)
        print(f"{n:,} rows written", flush=True)
    df = pd.read_parquet(a.cache, columns=COLUMNS)
    total = len(df)
    if not a.full:
        df = stratified_subsample(df, a.rows)
    df["split"] = assign_splits(df)
    summ = save_splits(df, a.out)
    summ["source_rows"] = int(total)
    (a.out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))
    print(f"written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
