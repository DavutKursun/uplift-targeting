"""Everything the service needs, computed once from DEV and committed under models/serving/ (never the test split):

  policy_grid_dev_visit.json   for budgets 1–50% and the models s_learner, r_learner, response: the score threshold (dev
                               quantile), incremental visits per 1,000 contacts raw and IPW-weighted (cross-fitted propensity
                               on dev), share of the total effect, profit per 1,000 contacts and per 1,000 customers under the
                               stated economics, and for the uplift learners the four-segment shares among the targeted;
                               plus each model's break-even and profit-maximising budgets, the segment thresholds δ and
                               p0_high, the economics and the dev ATE
  customers_dev_sample.csv     2,000 dev customers: an id and the 12 anonymous features, nothing else
  version.json                 git short hash, build date, deployed bundles

Usage: python scripts/export_serving.py [--n-customers 2000] [--out models/serving]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.config import ECONOMICS, FEATURES, MODELS, SEED
from uplift.data import ate, load_split, xty
from uplift.learners import propensity_oof
from uplift.metrics import budget_block
from uplift.policy import SEGMENTS, breakeven, segment, threshold_for_budget

SERVED = ("s_learner", "r_learner", "response")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-customers", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=MODELS / "serving")
    a = ap.parse_args(argv)
    dv = load_split("dev")
    X, t, y = xty(dv, "visit")
    base = ate(y, t)
    delta, p0_high = 0.25 * base["ate"], 2 * base["rate_control"]
    e_hat = propensity_oof(X, t)
    value, cost = ECONOMICS.value_per_visit, ECONOMICS.contact_cost
    grid = [round(i / 100, 2) for i in range(1, 51)]
    n = len(y)
    models, scores = {}, {}
    for name in SERVED:
        b = joblib.load(MODELS / f"{name}-visit.joblib")
        models[name], scores[name] = b["learner"], b["learner"].predict_uplift(X)
    out = {"split": "dev", "outcome": "visit", "n": n, "built": str(date.today()), "ate": base, "random_per_1000": round(base["ate"] * 1000, 2),
           "economics": {"value_per_visit": value, "contact_cost": cost, "marginal_threshold_per_1000": cost / value * 1000, "stated_as": "assumptions"},
           "segments": {"delta": round(delta, 7), "p0_high": round(p0_high, 6), "names": list(SEGMENTS)}, "budgets": grid, "models": {}}
    for name in SERVED:
        s = scores[name]
        blk = budget_block(y, t, s, e_hat, grid, n_resamples=0)
        seg = None
        if name != "response":
            p0, _ = models[name].predict_outcomes(X)
            seg = segment(p0, s, delta, p0_high)
        rows = []
        for b in grid:
            k = blk["n_targeted"][str(b)]
            thr = threshold_for_budget(s, b)
            raw, ipw = blk["raw"][str(b)]["point"], blk["ipw"][str(b)]["point"]
            row = {"budget": b, "threshold": round(float(thr), 7), "n_targeted": k, "per_1000": round(raw, 3), "ipw_per_1000": round(ipw, 3),
                   "share_of_total": round(blk["share_of_total"][str(b)], 4) if blk["share_of_total"][str(b)] is not None else None,
                   "profit_per_1000_contacts": round(value * raw - cost * 1000, 3), "ipw_profit_per_1000_contacts": round(value * ipw - cost * 1000, 3),
                   "profit_per_1000_customers": round((value * raw - cost * 1000) * k / n, 4), "ipw_profit_per_1000_customers": round((value * ipw - cost * 1000) * k / n, 4)}
            if seg is not None:
                targeted = s >= thr
                row["segments_among_targeted"] = {k_: round(float(np.mean(seg[targeted] == k_)), 4) for k_ in SEGMENTS}
            rows.append(row)
        out["models"][name] = {"bundle": f"{name}-visit.joblib", "grid": rows, "breakeven": breakeven(y, t, s, e_hat, ECONOMICS, value),
                               "score_meaning": "estimated uplift in visit probability" if name != "response" else "P(visit | treated), a ranking score, not an uplift"}
        print(f"{name}: threshold at 10% {rows[9]['threshold']:.5f}, per 1,000 {rows[9]['per_1000']:.1f} raw / {rows[9]['ipw_per_1000']:.1f} IPW, "
              f"break-even {out['models'][name]['breakeven']['raw']['breakeven_budget']}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "policy_grid_dev_visit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n, size=a.n_customers, replace=False))
    sample = pd.DataFrame(X[idx], columns=FEATURES).round(6)
    sample.insert(0, "id", range(a.n_customers))
    sample.to_csv(a.out / "customers_dev_sample.csv", index=False)
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=3, check=False).stdout.strip() or "dev"
    except (OSError, subprocess.SubprocessError):
        sha = "dev"
    (a.out / "version.json").write_text(json.dumps({"version": sha, "built": str(date.today()), "deployed": {"visit": "s_learner-visit.joblib", "cross_check": "r_learner-visit.joblib"},
                                                     "note": "thresholds and policy numbers come from the dev split; the test split was used once for the reported results"}, indent=2), encoding="utf-8")
    print(f"wrote {a.out}/policy_grid_dev_visit.json, customers_dev_sample.csv ({a.n_customers} rows), version.json ({sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
