"""The experiment itself: arm sizes, outcome rates by arm, ATE with CIs for both outcomes, randomization check.
Runs on train (never test).

Usage: python scripts/ate_report.py [--split train] [--splits-dir data/splits] [--out eval/results]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.config import OUTCOMES, RESULTS, SPLITS, TREATMENT
from uplift.data import ate, load_split, randomization_check


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "dev"])
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    df = load_split(a.split, a.splits_dir)
    out = {"split": a.split, "rows": len(df), "treated_share": round(float(df[TREATMENT].mean()), 5)}
    for o in OUTCOMES:
        out[o] = ate(df[o], df[TREATMENT])
    rc = randomization_check(df)
    out["randomization"] = {"max_abs_smd": round(float(rc["abs_smd"].max()), 5), "balanced": bool(rc["abs_smd"].max() < 0.1)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"ate_{a.split}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    rc.to_csv(a.out / f"randomization_{a.split}.csv", index=False)
    print(json.dumps(out, indent=2))
    print(rc.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
