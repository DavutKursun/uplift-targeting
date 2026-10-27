"""Fit learners on train, score dev once (Qini, no CI), save each as <models-dir>/<learner>-<outcome>.joblib.

Usage: python scripts/train.py [--learners all|response,s_learner,...] [--outcome visit] [--fast]
                               [--params JSON] [--cls-params JSON] [--reg-params JSON] [--rounds N] [--tau-rounds N]
                               [--estimate-propensity] [--categorical MAX_DISTINCT]
                               [--splits-dir data/splits] [--models-dir models] [--out eval/results]
--params applies to every LightGBM model of a learner; --cls-params only to the outcome / nuisance classifiers and
--reg-params only to the τ regressions (X, TO, R, DR); --tau-rounds sets the τ regressions' rounds (default --rounds).
--estimate-propensity fits e(x) instead of using the randomized share; --categorical N treats the columns with at most
N distinct values on train as categorical in LightGBM. Both are ablations: write them to their own --models-dir.
Never reads the test split.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.config import MODELS, PRIMARY_OUTCOME, RESULTS, SPLITS
from uplift.data import load_split, xty
from uplift.learners import LEARNERS, Settings, categorical_columns, fit_learner, settings_from
from uplift.metrics import auuc, policy_value, qini_coefficient


def build_settings(a, X) -> Settings:
    return settings_from(a.params, a.cls_params, a.reg_params, a.rounds, a.tau_rounds, a.estimate_propensity,
                         categorical_columns(X, a.categorical) if a.categorical else (), fast=a.fast)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", default="all")
    ap.add_argument("--outcome", default=PRIMARY_OUTCOME, choices=["visit", "conversion"])
    ap.add_argument("--fast", action="store_true", help="small-data settings (smoke runs)")
    ap.add_argument("--params", default=None, help="JSON overrides applied to both base-learner parameter sets")
    ap.add_argument("--cls-params", default=None, help="JSON overrides for the outcome / nuisance classifiers")
    ap.add_argument("--reg-params", default=None, help="JSON overrides for the τ regressions")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--tau-rounds", type=int, default=None, help="rounds for the τ regressions (default: --rounds)")
    ap.add_argument("--estimate-propensity", action="store_true", help="ablation: fit e(x) instead of using the randomized share")
    ap.add_argument("--categorical", type=int, default=0, metavar="MAX_DISTINCT",
                    help="ablation: columns with at most MAX_DISTINCT distinct values on train are categorical for LightGBM (0 = off)")
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    tr, dv = load_split("train", a.splits_dir), load_split("dev", a.splits_dir)
    X, t, y = xty(tr, a.outcome)
    Xd, td, yd = xty(dv, a.outcome)
    settings = build_settings(a, X)
    e = float(t.mean())
    names = list(LEARNERS) if a.learners == "all" else a.learners.split(",")
    spec = {"rounds": settings.rounds, "tau_rounds": settings.tau_rounds or settings.rounds, "estimate_propensity": settings.estimate_propensity,
            "categorical": list(settings.categorical), "cls_params": {k: settings.cls_params[k] for k in ("num_leaves", "min_data_in_leaf", "learning_rate")},
            "reg_params": {k: settings.reg_params[k] for k in ("num_leaves", "min_data_in_leaf", "learning_rate")}}
    print(f"train {len(X):,} rows (treated {e:.3f}, {a.outcome} rate {y.mean():.4f}) | dev {len(Xd):,} | learners {names}", flush=True)
    print(f"settings {json.dumps(spec)}", flush=True)
    results = {}
    a.models_dir.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        t0 = time.time()
        m = fit_learner(name, X, t, y, propensity=e, settings=settings)
        fit_s = time.time() - t0
        s = m.predict_uplift(Xd)
        res = {"qini": round(qini_coefficient(yd, td, s), 4), "auuc": round(auuc(yd, td, s), 6),
               "incremental_per_1000_at_20pct": round(policy_value(yd, td, s, 0.2)["incremental_per_1000"], 2), "fit_seconds": round(fit_s, 1),
               "propensity": round(e, 4), "settings": spec}
        path = a.models_dir / f"{name}-{a.outcome}.joblib"
        joblib.dump({"learner": m, "name": name, "outcome": a.outcome, "meta": res}, path)
        results[name] = {**res, "path": str(path)}
        print(f"{name:20s} fit {fit_s:6.1f}s | dev Qini {res['qini']:.4f} AUUC {res['auuc']:.5f} | +{res['incremental_per_1000_at_20pct']:.1f} per 1,000 at 20%", flush=True)
    (a.out / f"train_dev_{a.outcome}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
