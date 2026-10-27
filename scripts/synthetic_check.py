"""Ground-truth validation: every learner on a synthetic campaign with known τ(x). PEHE, Spearman with the truth,
Qini on the held-out part, policy value at 20%. A learner that cannot recover a known effect is not trusted on real data.

Usage: python scripts/synthetic_check.py [--n 200000] [--fast] [--out eval/results]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.config import RESULTS
from uplift.data import assign_splits, xty
from uplift.learners import LEARNERS, fast_settings, fit_learner
from uplift.metrics import pehe, policy_value, qini_coefficient, spearman
from uplift.synthetic import make_campaign


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--learners", default="all")
    a = ap.parse_args(argv)
    df = make_campaign(n=a.n, seed=11)
    df["split"] = assign_splits(df)
    tr, dv = df[df["split"] == "train"], df[df["split"] != "train"]
    X, t, y = xty(tr, "visit")
    Xd, td, yd = xty(dv, "visit")
    truth = dv["tau_true"].to_numpy()
    names = list(LEARNERS) if a.learners == "all" else a.learners.split(",")
    rows = [{"learner": "truth", "spearman_with_truth": 1.0, "pehe": 0.0, "qini": round(qini_coefficient(yd, td, truth), 4),
             "incremental_per_1000_at_20pct": round(policy_value(yd, td, truth, 0.2)["incremental_per_1000"], 2), "fit_seconds": 0.0}]
    for name in names:
        t0 = time.time()
        m = fit_learner(name, X, t, y, settings=fast_settings() if a.fast else None)
        s = m.predict_uplift(Xd)
        rows.append({"learner": name, "spearman_with_truth": round(spearman(truth, s), 4), "pehe": round(pehe(truth, s), 5),
                     "qini": round(qini_coefficient(yd, td, s), 4), "incremental_per_1000_at_20pct": round(policy_value(yd, td, s, 0.2)["incremental_per_1000"], 2),
                     "fit_seconds": round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out / "synthetic_check.csv", index=False)
    (a.out / "synthetic_check.json").write_text(json.dumps({"n": a.n, "true_ate": round(float(df["tau_true"].mean()), 5), "rows": rows}, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
