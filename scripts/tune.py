"""Hyperparameters, once, on dev Qini (M2): at most four configurations of (num_leaves, min_data_in_leaf, rounds) for the
S-learner's single classifier and for the τ regression of the R- and DR-learners. The nuisance models of R and DR keep
the defaults and are cross-fitted once per learner, then reused for every configuration, so the comparison isolates the
τ model. Writes eval/results/tuning_dev_<outcome>.{json,csv} and prints the train.py command that retrains each learner
with its chosen settings. Never reads the test split.

Usage: python scripts/tune.py [--outcome visit] [--learners s_learner,r_learner,dr_learner] [--out eval/results]
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

from uplift.config import PRIMARY_OUTCOME, RESULTS, SPLITS
from uplift.data import load_split, xty
from uplift.learners import CLS_PARAMS, REG_PARAMS, SLearner, Settings, _cross_fit_outcome, _predict, _train
from uplift.metrics import auuc, policy_value, qini_coefficient

# (num_leaves, min_data_in_leaf, rounds); the first entry is the committed default
GRID = {
    "s_learner": [(15, 2000, 150), (31, 2000, 300), (63, 1000, 300), (31, 500, 500)],
    "tau": [(7, 5000, 150), (15, 2000, 300), (7, 20000, 300), (3, 10000, 500)],
}


def _score(yd, td, s) -> dict:
    return {"qini": round(qini_coefficient(yd, td, s), 4), "auuc": round(auuc(yd, td, s), 6),
            "per_1000_at_20pct": round(policy_value(yd, td, s, 0.2)["incremental_per_1000"], 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default=PRIMARY_OUTCOME, choices=["visit", "conversion"])
    ap.add_argument("--learners", default="s_learner,r_learner,dr_learner")
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    tr, dv = load_split("train", a.splits_dir), load_split("dev", a.splits_dir)
    X, t, y = xty(tr, a.outcome)
    Xd, td, yd = xty(dv, a.outcome)
    e = float(t.mean())
    rows = []
    for name in a.learners.split(","):
        grid = GRID["s_learner"] if name == "s_learner" else GRID["tau"]
        if name == "s_learner":
            targets = []
            for leaves, min_data, rounds in grid:
                t0 = time.time()
                m = SLearner(e, Settings(cls_params={**CLS_PARAMS, "num_leaves": leaves, "min_data_in_leaf": min_data}, rounds=rounds)).fit(X, t, y)
                targets.append((leaves, min_data, rounds, m.predict_uplift(Xd), time.time() - t0))
        else:
            t0 = time.time()
            if name == "r_learner":
                m_hat = _cross_fit_outcome(X, t, y, Settings(), per_arm=False)
                y_res, t_res = y - m_hat, t - e
                target, weight = y_res / t_res, t_res**2
            elif name == "dr_learner":
                mu0, mu1 = _cross_fit_outcome(X, t, y, Settings(), per_arm=True)
                target, weight = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e), None
            else:
                raise SystemExit(f"tune.py handles s_learner, r_learner, dr_learner; got {name}")
            nuisance_s = time.time() - t0
            print(f"{name}: nuisances cross-fitted once in {nuisance_s:.0f}s", flush=True)
            targets = []
            for leaves, min_data, rounds in grid:
                t0 = time.time()
                tau = _train(X, target, {**REG_PARAMS, "num_leaves": leaves, "min_data_in_leaf": min_data}, rounds, weight=weight)
                targets.append((leaves, min_data, rounds, _predict(tau, Xd), time.time() - t0 + nuisance_s))
        for i, (leaves, min_data, rounds, s, secs) in enumerate(targets):
            rows.append({"learner": name, "config": i, "num_leaves": leaves, "min_data_in_leaf": min_data, "rounds": rounds, "is_default": i == 0,
                         **_score(yd, td, s), "fit_seconds": round(secs, 1)})
            print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    choice = {}
    for name, g in table.groupby("learner"):
        best = g.sort_values(["qini", "is_default"], ascending=[False, False]).iloc[0]
        choice[name] = {k: (int(best[k]) if k != "qini" else float(best[k])) for k in ("config", "num_leaves", "min_data_in_leaf", "rounds", "qini")}
        choice[name]["is_default"] = bool(best["is_default"])
        over = json.dumps({"num_leaves": int(best["num_leaves"]), "min_data_in_leaf": int(best["min_data_in_leaf"])})
        flag = f"--cls-params '{over}' --rounds {int(best['rounds'])}" if name == "s_learner" else f"--reg-params '{over}' --tau-rounds {int(best['rounds'])}"
        choice[name]["retrain"] = f"python scripts/train.py --learners {name} --outcome {a.outcome} {flag}"
    a.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out / f"tuning_dev_{a.outcome}.csv", index=False)
    (a.out / f"tuning_dev_{a.outcome}.json").write_text(json.dumps({"outcome": a.outcome, "grid": GRID, "rows": rows, "choice": choice}, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    for name, c in choice.items():
        print(f"{name}: config {c['config']} (leaves {c['num_leaves']}, min_data {c['min_data_in_leaf']}, rounds {c['rounds']}) dev Qini {c['qini']:.4f}"
              f"{' = default' if c['is_default'] else ''}\n  {c['retrain']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
