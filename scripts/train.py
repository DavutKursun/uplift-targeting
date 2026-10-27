"""Fit learners on train, score dev once (Qini, no CI), save each as models/<learner>-<outcome>.joblib.

Usage: python scripts/train.py [--learners all|response,s_learner,...] [--outcome visit] [--fast] [--params '{"num_leaves": 15}']
                               [--rounds 400] [--estimate-propensity] [--splits-dir data/splits] [--models-dir models] [--out eval/results]
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
from uplift.learners import LEARNERS, Settings, fast_settings, fit_learner
from uplift.metrics import auuc, policy_value, qini_coefficient


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", default="all")
    ap.add_argument("--outcome", default=PRIMARY_OUTCOME, choices=["visit", "conversion"])
    ap.add_argument("--fast", action="store_true", help="small-data settings (smoke runs)")
    ap.add_argument("--params", default=None, help="JSON overrides applied to both base-learner parameter sets")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--estimate-propensity", action="store_true", help="ablation: fit e(x) instead of using the randomized share")
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    tr, dv = load_split("train", a.splits_dir), load_split("dev", a.splits_dir)
    X, t, y = xty(tr, a.outcome)
    Xd, td, yd = xty(dv, a.outcome)
    settings = fast_settings() if a.fast else Settings()
    if a.params:
        over = json.loads(a.params)
        settings.cls_params.update(over)
        settings.reg_params.update(over)
    if a.rounds:
        settings.rounds = a.rounds
    settings.estimate_propensity = a.estimate_propensity
    e = float(t.mean())
    names = list(LEARNERS) if a.learners == "all" else a.learners.split(",")
    print(f"train {len(X):,} rows (treated {e:.3f}, {a.outcome} rate {y.mean():.4f}) | dev {len(Xd):,} | learners {names}", flush=True)
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
               "propensity": round(e, 4), "estimate_propensity": a.estimate_propensity, "rounds": settings.rounds}
        tag = "" if a.estimate_propensity else ""
        path = a.models_dir / f"{name}-{a.outcome}{tag}.joblib"
        joblib.dump({"learner": m, "name": name, "outcome": a.outcome, "meta": res}, path)
        results[name] = {**res, "path": str(path)}
        print(f"{name:20s} fit {fit_s:6.1f}s | dev Qini {res['qini']:.4f} AUUC {res['auuc']:.5f} | +{res['incremental_per_1000_at_20pct']:.1f} per 1,000 at 20%", flush=True)
    (a.out / f"train_dev_{a.outcome}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
