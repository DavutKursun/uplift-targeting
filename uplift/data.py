"""Criteo file → float32 parquet (chunked) → stratified subsample → stratified splits; randomization check; ATE.

The raw CSV is 14M rows; it is read in chunks so an 8 GB laptop never holds the text. Splits are random (no
timestamps exist) and stratified by (treatment, visit) so the 15% control arm is present everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURES, OUTCOMES, RAW, RAW_CANDIDATES, SEED, SPLIT_RULES, SPLITS, TREATMENT, SplitRules

COLUMNS = FEATURES + [TREATMENT, "conversion", "visit", "exposure"]
DTYPES = {**{f: "float32" for f in FEATURES}, TREATMENT: "int8", "conversion": "int8", "visit": "int8", "exposure": "int8"}
SPLIT_NAMES = ("train", "dev", "test")


def find_raw(raw_dir: Path = RAW) -> Path | None:
    for name in RAW_CANDIDATES:
        p = raw_dir / name
        if p.exists():
            return p
    return None


def csv_to_parquet(csv_path: Path, out_path: Path, chunksize: int = 1_000_000) -> int:
    """Stream the CSV into one parquet file with compact dtypes. Returns the row count."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer, n = None, 0
    for chunk in pd.read_csv(csv_path, usecols=COLUMNS, dtype=DTYPES, chunksize=chunksize):
        table = pa.Table.from_pandas(chunk[COLUMNS], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table)
        n += len(chunk)
    if writer is not None:
        writer.close()
    return n


def stratum(df: pd.DataFrame) -> pd.Series:
    return df[TREATMENT].astype(int) * 2 + df["visit"].astype(int)


def stratified_subsample(df: pd.DataFrame, n_rows: int, seed: int = SEED) -> pd.DataFrame:
    """Keep n_rows with the same (treatment, visit) mix as the full file."""
    if n_rows >= len(df):
        return df
    frac = n_rows / len(df)
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(df), dtype=bool)
    s = stratum(df).to_numpy()
    for value in np.unique(s):
        idx = np.flatnonzero(s == value)
        take = rng.choice(idx, size=int(round(len(idx) * frac)), replace=False)
        keep[take] = True
    return df[keep].reset_index(drop=True)


def assign_splits(df: pd.DataFrame, rules: SplitRules = SPLIT_RULES, seed: int = SEED) -> pd.Series:
    """train / dev / test, stratified by (treatment, visit)."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(df), dtype=object)
    s = stratum(df).to_numpy()
    for value in np.unique(s):
        idx = np.flatnonzero(s == value)
        rng.shuffle(idx)
        n_tr, n_dv = int(len(idx) * rules.train), int(len(idx) * rules.dev)
        out[idx[:n_tr]] = "train"
        out[idx[n_tr : n_tr + n_dv]] = "dev"
        out[idx[n_tr + n_dv :]] = "test"
    return pd.Series(out, index=df.index)


def randomization_check(df: pd.DataFrame, features: list[str] = FEATURES) -> pd.DataFrame:
    """Standardized mean difference of every feature between arms; |SMD| < 0.1 is the usual balance bar,
    a real RCT at this size sits far below 0.02."""
    t = df[TREATMENT].to_numpy().astype(bool)
    rows = []
    for f in features:
        x = df[f].to_numpy(dtype=np.float64)
        a, b = x[t], x[~t]
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd = (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0
        rows.append({"feature": f, "mean_treated": float(a.mean()), "mean_control": float(b.mean()), "smd": round(float(smd), 5),
                     "n_distinct": len(np.unique(x))})
    out = pd.DataFrame(rows)
    out["abs_smd"] = out["smd"].abs()
    return out.sort_values("abs_smd", ascending=False).reset_index(drop=True)


def ate(y, t, z: float = 1.96) -> dict:
    """Difference in outcome rates between arms with a Wald CI; relative uplift against the control rate."""
    y, t = np.asarray(y, dtype=float), np.asarray(t).astype(bool)
    y1, y0 = y[t], y[~t]
    p1, p0 = y1.mean(), y0.mean()
    se = np.sqrt(p1 * (1 - p1) / len(y1) + p0 * (1 - p0) / len(y0))
    diff = p1 - p0
    return {"n_treated": len(y1), "n_control": len(y0), "rate_treated": round(float(p1), 6), "rate_control": round(float(p0), 6),
            "ate": round(float(diff), 6), "ci_low": round(float(diff - z * se), 6), "ci_high": round(float(diff + z * se), 6),
            "relative_uplift": round(float(diff / p0), 4) if p0 > 0 else None, "se": round(float(se), 7)}


def summary(df: pd.DataFrame) -> dict:
    out = {"rows": len(df), "treated_share": round(float(df[TREATMENT].mean()), 5)}
    for o in OUTCOMES:
        out[o] = ate(df[o], df[TREATMENT])
    if "split" in df.columns:
        out["splits"] = {s: {"rows": int((df["split"] == s).sum()), "treated_share": round(float(df.loc[df["split"] == s, TREATMENT].mean()), 5),
                             "visit_rate": round(float(df.loc[df["split"] == s, "visit"].mean()), 6)} for s in SPLIT_NAMES if (df["split"] == s).any()}
    return out


def save_splits(df: pd.DataFrame, out_dir: Path = SPLITS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLIT_NAMES:
        df[df["split"] == s].drop(columns=["split"]).to_parquet(out_dir / f"{s}.parquet", index=False)
    summ = summary(df)
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    return summ


def load_split(name: str, splits_dir: Path = SPLITS) -> pd.DataFrame:
    if name not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    p = splits_dir / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run scripts/build_splits.py first")
    return pd.read_parquet(p)


def xty(df: pd.DataFrame, outcome: str):
    """(X as float32 ndarray, treatment int array, outcome int array)."""
    return df[FEATURES].to_numpy(dtype=np.float32), df[TREATMENT].to_numpy().astype(int), df[outcome].to_numpy().astype(int)
