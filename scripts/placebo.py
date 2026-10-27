"""Placebo: shuffle the treatment labels in train (so no real effect can be learned), fit a learner with the deployed
settings, score dev (or test, once, with --final). A sound pipeline gives a Qini whose CI contains 0; anything else means
leakage or a metric bug.

Usage: python scripts/placebo.py [--learner s_learner] [--outcome visit] [--split dev] [--final] [--fast] [--resamples 200]
       [--params JSON] [--cls-params JSON] [--reg-params JSON] [--rounds N] [--tau-rounds N] [--seed 7]
       [--shuffles K] [--compare-qini Q]
One shuffle writes <out>/placebo_<split>_<outcome>_<learner>[_seedS].json. `--shuffles K` fits K independent shuffles
(seeds 1..K), each once with the same settings, and writes <out>/placebo_null_<split>_<outcome>_<learner>.json with the
spread of their Qini — the placebo null of CLAUDE.md §9; `--compare-qini Q` states whether Q lies outside that spread.
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
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    ap.add_argument("--params", default=None)
    ap.add_argument("--cls-params", default=None)
    ap.add_argument("--reg-params", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--tau-rounds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED, help="seed of the label shuffle")
    ap.add_argument("--shuffles", type=int, default=1, help="K independent shuffles (seeds 1..K) → the placebo null spread")
    ap.add_argument("--compare-qini", type=float, default=None, help="a real model's Qini to place against the null spread")
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    if a.split == "test" and not a.final:
        print("refusing: the test split is scored once, at Milestone 4, with --final.")
        return 3
    tr, dv = load_split("train", a.splits_dir), load_split(a.split, a.splits_dir)
    X, t, y = xty(tr, a.outcome)
    Xd, td, yd = xty(dv, a.outcome)
    settings = settings_from(a.params, a.cls_params, a.reg_params, a.rounds, a.tau_rounds, fast=a.fast)
    spec = {"rounds": settings.rounds, "tau_rounds": settings.tau_rounds or settings.rounds,
            "cls_params": {k: settings.cls_params[k] for k in ("num_leaves", "min_data_in_leaf")}, "reg_params": {k: settings.reg_params[k] for k in ("num_leaves", "min_data_in_leaf")}}
    a.out.mkdir(parents=True, exist_ok=True)

    def one(seed: int) -> dict:
        rng = np.random.default_rng(seed)
        m = fit_learner(a.learner, X, rng.permutation(t), y, propensity=float(t.mean()), settings=settings)
        q = bootstrap(qini_coefficient, yd, td, m.predict_uplift(Xd), a.resamples)
        return {"shuffle_seed": seed, "qini_placebo": q, "contains_zero": bool(q["ci_low"] <= 0 <= q["ci_high"])}

    if a.shuffles <= 1:
        out = {"learner": a.learner, "outcome": a.outcome, "split": a.split, "n": int(len(yd)), "settings": spec, **one(a.seed)}
        tag = "" if a.seed == SEED else f"_seed{a.seed}"
        (a.out / f"placebo_{a.split}_{a.outcome}_{a.learner}{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return 0
    shuffles = []
    for seed in range(1, a.shuffles + 1):
        shuffles.append(one(seed))
        q = shuffles[-1]["qini_placebo"]
        print(f"shuffle {seed}: Qini {q['point']:+.4f} [{q['ci_low']:+.4f}, {q['ci_high']:+.4f}]", flush=True)
    vals = np.array([s_["qini_placebo"]["point"] for s_ in shuffles])
    out = {"learner": a.learner, "outcome": a.outcome, "split": a.split, "n": int(len(yd)), "settings": spec, "n_shuffles": a.shuffles, "shuffles": shuffles,
           "spread": {"min": round(float(vals.min()), 5), "max": round(float(vals.max()), 5), "mean": round(float(vals.mean()), 5), "sd": round(float(vals.std(ddof=1)), 5)},
           "rule": "the null is the spread of Qini over the shuffles; a real model's Qini must lie outside it; each shuffle's bootstrap CI is reported, not used"}
    if a.compare_qini is not None:
        out["compare_qini"] = a.compare_qini
        out["outside_spread"] = bool(a.compare_qini > vals.max() or a.compare_qini < vals.min())
    (a.out / f"placebo_null_{a.split}_{a.outcome}_{a.learner}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k != "shuffles"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
