"""Budget curves and segments: for every saved model, incremental responses per 1,000 contacts and profit under the
stated economics at each budget; segment shares and the observed uplift among predicted sleeping dogs for one chosen model.

Usage: python scripts/policy_eval.py --split dev [--outcome visit] [--segments-model r_learner] [--final]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.config import BUDGETS, ECONOMICS, MODELS, PRIMARY_OUTCOME, RESULTS, SPLITS
from uplift.data import ate, load_split, xty
from uplift.policy import budget_table, segment, segment_shares


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--outcome", default=PRIMARY_OUTCOME, choices=["visit", "conversion"])
    ap.add_argument("--segments-model", default="r_learner")
    ap.add_argument("--delta-share-of-ate", type=float, default=0.25, help="δ for segments as a share of the ATE")
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    if a.split == "test" and not a.final:
        print("refusing: test needs --final")
        return 3
    df = load_split(a.split, a.splits_dir)
    X, t, y = xty(df, a.outcome)
    base = ate(y, t)
    scores, learners = {}, {}
    for p in sorted(a.models_dir.glob(f"*-{a.outcome}.joblib")):
        b = joblib.load(p)
        learners[b["name"]] = b["learner"]
        scores[b["name"]] = b["learner"].predict_uplift(X)
    if not scores:
        print("no models — run scripts/train.py")
        return 2
    value = ECONOMICS.value_per_visit if a.outcome == "visit" else ECONOMICS.value_per_conversion
    table = budget_table(y, t, scores, BUDGETS, ECONOMICS, value_per_response=value)
    a.out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(a.out / f"policy_{a.split}_{a.outcome}.csv", index=False)
    out = {"split": a.split, "outcome": a.outcome, "ate": base, "economics": {"value_per_response": value, "contact_cost": ECONOMICS.contact_cost}}
    if a.segments_model in learners:
        m = learners[a.segments_model]
        p0, _ = m.predict_outcomes(X)
        tau = scores[a.segments_model]
        delta, p0_high = a.delta_share_of_ate * base["ate"], 2 * base["rate_control"]
        seg = segment(p0, tau, delta, p0_high)
        dogs = seg == "sleeping_dog"
        dog_uplift = ate(y[dogs], t[dogs]) if dogs.sum() > 200 and len(np.unique(t[dogs])) == 2 else None
        out["segments"] = {"model": a.segments_model, "delta": round(delta, 6), "p0_high": round(p0_high, 6), "shares": segment_shares(p0, tau, delta, p0_high),
                           "sleeping_dogs_observed": dog_uplift, "share_tau_negative": round(float((tau < 0).mean()), 4)}
    (a.out / f"policy_{a.split}_{a.outcome}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    pivot = table.pivot(index="model", columns="budget", values="incremental_per_1000").round(1)
    print("\nincremental responses per 1,000 contacts by budget:\n", pivot.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
