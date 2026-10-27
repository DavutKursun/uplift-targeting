"""Qini / AUUC / policy value with bootstrap CIs and the uplift-by-decile table for every saved model on one split.
`--split test` needs `--final` (used once, at M4).

Usage: python scripts/evaluate.py --split dev [--outcome visit] [--resamples 200] [--models-dir models] [--out eval/results]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.config import BOOTSTRAP_RESAMPLES, MODELS, PRIMARY_OUTCOME, RESULTS, SPLITS
from uplift.data import ate, load_split, xty
from uplift.metrics import evaluate_scores, uplift_by_bin


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--outcome", default=PRIMARY_OUTCOME, choices=["visit", "conversion"])
    ap.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    if a.split == "test" and not a.final:
        print("refusing: the test split is evaluated once, at Milestone 4, with --final.")
        return 3
    df = load_split(a.split, a.splits_dir)
    X, t, y = xty(df, a.outcome)
    paths = sorted(a.models_dir.glob(f"*-{a.outcome}.joblib"))
    if not paths:
        print(f"no models for outcome {a.outcome} in {a.models_dir} — run scripts/train.py")
        return 2
    base = ate(y, t)
    rows, deciles = [], {}
    a.out.mkdir(parents=True, exist_ok=True)
    for p in paths:
        bundle = joblib.load(p)
        name, m = bundle["name"], bundle["learner"]
        s = m.predict_uplift(X)
        ev = evaluate_scores(y, t, s, n_resamples=a.resamples)
        rows.append({"model": name, "qini": round(ev["qini"]["point"], 4), "qini_ci_low": round(ev["qini"]["ci_low"], 4), "qini_ci_high": round(ev["qini"]["ci_high"], 4),
                     "auuc": round(ev["auuc"], 6),
                     **{f"per_1000_at_{int(float(b) * 100)}pct": round(v["point"], 2) for b, v in ev["policy"].items()},
                     **{f"per_1000_at_{int(float(b) * 100)}pct_ci": f"[{v['ci_low']:.1f}, {v['ci_high']:.1f}]" for b, v in ev["policy"].items()},
                     **{f"share_of_total_at_{int(float(b) * 100)}pct": round(v["share_of_total"], 4) if v["share_of_total"] is not None else None
                        for b, v in ev["policy"].items()}})
        tab = uplift_by_bin(y, t, s, predicted=s if name != "response" else None)
        tab.to_csv(a.out / f"deciles_{a.split}_{a.outcome}_{name}.csv", index=False)
        deciles[name] = tab[["bin", "n", "uplift_observed", "uplift_predicted"]].round(5).to_dict(orient="records")
        print(f"{name:20s} Qini {rows[-1]['qini']:.4f} [{rows[-1]['qini_ci_low']:.4f}, {rows[-1]['qini_ci_high']:.4f}]  per 1,000 at 20%: {rows[-1]['per_1000_at_20pct']}", flush=True)
    table = pd.DataFrame(rows).sort_values("qini", ascending=False)
    stamp = {"split": a.split, "outcome": a.outcome, "run_date": str(date.today()), "n": len(df), "n_resamples": a.resamples, "models_dir": str(a.models_dir),
             "ate": base, "random_per_1000": round(base["ate"] * 1000, 2),
             "metrics": table.to_dict(orient="records"), "deciles": deciles}
    (a.out / f"{a.split}-summary-{a.outcome}.json").write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    md = [f"# {a.split} results — outcome `{a.outcome}`, {len(df):,} rows, run {stamp['run_date']}", "",
          f"ATE {base['ate']:.5f} [{base['ci_low']:.5f}, {base['ci_high']:.5f}] → random targeting gives {stamp['random_per_1000']} incremental per 1,000 contacts.", "",
          table.to_markdown(index=False), ""]
    (a.out / f"{a.split}-summary-{a.outcome}.md").write_text("\n".join(md), encoding="utf-8")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
