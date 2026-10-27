"""Placebo: shuffle the treatment labels in train (so no real effect can be learned), fit a learner with the deployed
settings, score dev. A sound pipeline gives a Qini whose CI contains 0; anything else means leakage or a metric bug.

Usage: python scripts/placebo.py [--learner s_learner] [--outcome visit] [--fast] [--resamples 200]
       [--params JSON] [--cls-params JSON] [--reg-params JSON] [--rounds N] [--tau-rounds N]
Writes <out>/placebo_<outcome>_<learner>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.config import BOOTSTRAP_RESAMPLES, PRIMARY_OUTCOME, RESULTS, SEED, SPLITS
from uplift.data import load_split, xty
from uplift.learners import fit_learner, settings_from
from uplift.metrics import bootstrap, qini_coefficient


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", default="s_learner")
    ap.add_argument("--outcome", default=PRIMARY_OUTCOME, choices=["visit", "conversion"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    ap.add_argument("--params", default=None)
    ap.add_argument("--cls-params", default=None)
    ap.add_argument("--reg-params", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--tau-rounds", type=int, default=None)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    tr, dv = load_split("train", a.splits_dir), load_split("dev", a.splits_dir)
    X, t, y = xty(tr, a.outcome)
    Xd, td, yd = xty(dv, a.outcome)
    settings = settings_from(a.params, a.cls_params, a.reg_params, a.rounds, a.tau_rounds, fast=a.fast)
    rng = np.random.default_rng(SEED)
    t_shuffled = rng.permutation(t)
    m = fit_learner(a.learner, X, t_shuffled, y, propensity=float(t.mean()), settings=settings)
    s = m.predict_uplift(Xd)
    q = bootstrap(qini_coefficient, yd, td, s, a.resamples)
    spec = {"rounds": settings.rounds, "tau_rounds": settings.tau_rounds or settings.rounds,
            "cls_params": {k: settings.cls_params[k] for k in ("num_leaves", "min_data_in_leaf")}, "reg_params": {k: settings.reg_params[k] for k in ("num_leaves", "min_data_in_leaf")}}
    out = {"learner": a.learner, "outcome": a.outcome, "settings": spec, "qini_placebo": q, "contains_zero": bool(q["ci_low"] <= 0 <= q["ci_high"])}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"placebo_{a.outcome}_{a.learner}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
