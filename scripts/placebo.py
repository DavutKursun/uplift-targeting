"""Placebo: shuffle the treatment labels in train (so no real effect can be learned), fit a learner, score dev.
A sound pipeline gives a Qini whose CI contains 0; anything else means leakage or a metric bug.

Usage: python scripts/placebo.py [--learner r_learner] [--outcome visit] [--fast] [--resamples 200]
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
from uplift.learners import Settings, fast_settings, fit_learner
from uplift.metrics import bootstrap, qini_coefficient


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", default="r_learner")
    ap.add_argument("--outcome", default=PRIMARY_OUTCOME, choices=["visit", "conversion"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    tr, dv = load_split("train", a.splits_dir), load_split("dev", a.splits_dir)
    X, t, y = xty(tr, a.outcome)
    Xd, td, yd = xty(dv, a.outcome)
    rng = np.random.default_rng(SEED)
    t_shuffled = rng.permutation(t)
    m = fit_learner(a.learner, X, t_shuffled, y, propensity=float(t.mean()), settings=fast_settings() if a.fast else Settings())
    s = m.predict_uplift(Xd)
    q = bootstrap(qini_coefficient, yd, td, s, a.resamples)
    out = {"learner": a.learner, "outcome": a.outcome, "qini_placebo": q, "contains_zero": bool(q["ci_low"] <= 0 <= q["ci_high"])}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"placebo_{a.outcome}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
