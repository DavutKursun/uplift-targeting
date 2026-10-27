"""Paired bootstrap differences between models scored on the same customers (M3). For each pair a − b: the difference in
normalised Qini and in incremental responses per 1,000 contacts at each budget, with CIs from the SAME stratified
resamples (the right test for two models on the same split; two marginal CIs overlapping says nothing).

Usage: python scripts/paired.py --split dev [--outcome visit] [--pairs s_learner-response,r_learner-response]
       [--budgets 0.1,0.2,0.3] [--resamples 200] [--models-dir models] [--out eval/results] [--final]
Writes <out>/paired_<split>_<outcome>.{json,md}. `--split test` needs `--final`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.config import BOOTSTRAP_RESAMPLES, MODELS, PRIMARY_OUTCOME, RESULTS, SPLITS
from uplift.data import ate, load_split, xty
from uplift.metrics import paired_block


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--outcome", default=PRIMARY_OUTCOME, choices=["visit", "conversion"])
    ap.add_argument("--pairs", default="s_learner-response,r_learner-response")
    ap.add_argument("--budgets", default="0.1,0.2,0.3")
    ap.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    if a.split == "test" and not a.final:
        print("refusing: the test split is evaluated once, at Milestone 4, with --final.")
        return 3
    pairs = [tuple(p.split("-")) for p in a.pairs.split(",")]
    budgets = [float(b) for b in a.budgets.split(",")]
    names = sorted({n for p in pairs for n in p})
    df = load_split(a.split, a.splits_dir)
    X, t, y = xty(df, a.outcome)
    scores = {}
    for name in names:
        p = a.models_dir / f"{name}-{a.outcome}.joblib"
        if not p.exists():
            print(f"no bundle {p} — run scripts/train.py")
            return 2
        scores[name] = joblib.load(p)["learner"].predict_uplift(X)
    res = paired_block(y, t, scores, pairs, budgets, a.resamples)
    base = ate(y, t)
    rows = []
    for pair, d in res["pairs"].items():
        for key, v in d.items():
            rows.append({"pair": pair, "statistic": "qini" if key == "qini" else f"per_1000_at_{int(round(float(key) * 100))}pct",
                         "difference": round(v["point"], 4), "ci_low": round(v["ci_low"], 4), "ci_high": round(v["ci_high"], 4), "excludes_zero": v["excludes_zero"]})
    table = pd.DataFrame(rows)
    out = {"split": a.split, "outcome": a.outcome, "n": len(df), "n_resamples": a.resamples, "ate": base, "random_per_1000": round(base["ate"] * 1000, 2),
           "models": res["models"], "pairs": res["pairs"], "table": rows}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"paired_{a.split}_{a.outcome}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = [f"# Paired differences on {a.split} — outcome `{a.outcome}`, {len(df):,} rows, {a.resamples} resamples", "",
          "Same stratified resamples for both models of each pair; a CI that excludes 0 is a paired-bootstrap difference at the 95% level.", "",
          table.to_markdown(index=False), ""]
    (a.out / f"paired_{a.split}_{a.outcome}.md").write_text("\n".join(md), encoding="utf-8")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
