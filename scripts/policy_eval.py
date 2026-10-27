"""Decisions on one split (M3). Budget curves 5–50% for random, the response model and the deployed pair: incremental
responses per 1,000 contacts raw and IPW-weighted (cross-fitted propensity: the M1 imbalance correction) with CIs, the
share of the total effect, profit under the stated economics with the break-even and profit-maximising budgets. For
one chosen model: the four segments with the observed uplift inside each (the sleeping-dog test), the τ̂ < 0 group,
and five worked customers (one per segment plus one on the budget threshold).

Usage: python scripts/policy_eval.py --split dev [--outcome visit] [--models random,response,s_learner,r_learner]
       [--segments-model s_learner] [--delta-share-of-ate 0.25 | --delta D --p0-high P] [--resamples 200] [--fast] [--final]
Writes policy_<split>_<outcome>.{csv,json} and worked_customers_<split>_<outcome>.{json,md}. `--split test` needs `--final`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.config import BOOTSTRAP_RESAMPLES, ECONOMICS, FEATURES, MODELS, PRIMARY_OUTCOME, RESULTS, SPLITS
from uplift.data import ate, load_split, xty
from uplift.learners import fast_settings, propensity_oof
from uplift.policy import BUDGET_GRID, breakeven, budget_curves, decide, segment, segment_report, threshold_for_budget


def worked_customers(X, p0, tau, seg, thr: float) -> list[dict]:
    """One customer per segment plus one on the 20%-budget threshold, as feature rows and scores. The features are
    anonymous (Criteo's f0–f11), so a row explains nothing by itself; it shows what the service returns."""
    picks = []
    for role, mask, pick in (("persuadable", seg == "persuadable", lambda i: i[np.argmax(tau[i])]),
                             ("sure_thing", seg == "sure_thing", lambda i: i[np.argmax(p0[i])]),
                             ("lost_cause", seg == "lost_cause", lambda i: i[np.argmin(p0[i])]),
                             ("sleeping_dog", seg == "sleeping_dog", lambda i: i[np.argmin(tau[i])])):
        idx = np.flatnonzero(mask)
        if len(idx):
            picks.append((role, int(pick(idx))))
    picks.append(("borderline", int(np.argmin(np.abs(tau - thr)))))
    rows = []
    for role, i in picks:
        rows.append({"role": role, "row": i, **{f: round(float(X[i, j]), 4) for j, f in enumerate(FEATURES)},
                     "p_control": round(float(p0[i]), 5), "p_treated": round(float(np.clip(p0[i] + tau[i], 0, 1)), 5), "uplift": round(float(tau[i]), 5),
                     "segment": str(seg[i]), "decision_at_20pct": str(decide([tau[i]], thr)[0])})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--outcome", default=PRIMARY_OUTCOME, choices=["visit", "conversion"])
    ap.add_argument("--models", default="random,response,s_learner,r_learner")
    ap.add_argument("--segments-model", default="s_learner")
    ap.add_argument("--delta-share-of-ate", type=float, default=0.25, help="δ for segments as a share of the ATE of the split")
    ap.add_argument("--delta", type=float, default=None, help="fixed δ (e.g. the dev value) instead of a share of this split's ATE")
    ap.add_argument("--p0-high", type=float, default=None, help="fixed p0_high (e.g. the dev value) instead of 2 × this split's control rate")
    ap.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    ap.add_argument("--fast", action="store_true", help="small-data settings for the propensity model (smoke runs)")
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
    for name in [m for m in a.models.split(",") if m != "random"]:
        p = a.models_dir / f"{name}-{a.outcome}.joblib"
        if not p.exists():
            print(f"skipping {name}: no bundle at {p}")
            continue
        b = joblib.load(p)
        learners[name], scores[name] = b["learner"], b["learner"].predict_uplift(X)
    if not scores:
        print("no models — run scripts/train.py")
        return 2
    e_hat = propensity_oof(X, t, fast_settings() if a.fast else None)
    from sklearn.metrics import roc_auc_score

    prop = {"auc_oof": round(float(roc_auc_score(t, e_hat)), 4), "p1": round(float(np.percentile(e_hat, 1)), 4), "p99": round(float(np.percentile(e_hat, 99)), 4),
            "mean": round(float(e_hat.mean()), 4)}
    value = ECONOMICS.value_per_visit if a.outcome == "visit" else ECONOMICS.value_per_conversion
    table = budget_curves(y, t, scores, e_hat, BUDGET_GRID, ECONOMICS, value, n_resamples=a.resamples)
    table["raw_minus_ipw"] = table["incremental_per_1000"] - table["ipw_per_1000"]
    table["raw_ci_halfwidth"] = (table["ci_high"] - table["ci_low"]) / 2
    exceed = table[table["raw_minus_ipw"].abs() > table["raw_ci_halfwidth"]][["model", "budget", "incremental_per_1000", "ipw_per_1000", "raw_ci_halfwidth"]]
    a.out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(a.out / f"policy_{a.split}_{a.outcome}.csv", index=False)
    out = {"split": a.split, "outcome": a.outcome, "n": len(df), "n_resamples": a.resamples, "ate": base, "random_per_1000": round(base["ate"] * 1000, 2),
           "economics": {"value_per_response": value, "contact_cost": ECONOMICS.contact_cost, "stated_as": "assumptions, not measured",
                         "marginal_threshold_per_1000": ECONOMICS.contact_cost / value * 1000},
           "propensity": prop, "budgets": list(BUDGET_GRID),
           "raw_vs_ipw": {"max_abs_difference_per_1000": round(float(table["raw_minus_ipw"].abs().max()), 3),
                          "rows_where_difference_exceeds_ci_halfwidth": exceed.round(3).to_dict(orient="records")},
           "breakeven": {name: breakeven(y, t, s, e_hat, ECONOMICS, value) for name, s in {"random": np.random.default_rng(0).random(len(y)), **scores}.items()}}
    if a.segments_model in learners:
        m = learners[a.segments_model]
        p0, _ = m.predict_outcomes(X)
        tau = scores[a.segments_model]
        delta = a.delta if a.delta is not None else a.delta_share_of_ate * base["ate"]
        p0_high = a.p0_high if a.p0_high is not None else 2 * base["rate_control"]
        rep = segment_report(y, t, p0, tau, delta, p0_high)
        rep["model"] = a.segments_model
        rep["thresholds_source"] = "fixed on the command line" if (a.delta is not None or a.p0_high is not None) else "derived from this split"
        rep["share_tau_negative"] = rep["tau_negative"]["share"]
        out["segments"] = rep
        thr = threshold_for_budget(tau, 0.2)
        seg = segment(p0, tau, delta, p0_high)
        rows = worked_customers(X, p0, tau, seg, thr)
        out["worked_customers"] = {"threshold_at_20pct": round(float(thr), 6), "rows": rows}
        wc = pd.DataFrame(rows)
        (a.out / f"worked_customers_{a.split}_{a.outcome}.json").write_text(json.dumps(out["worked_customers"], indent=2), encoding="utf-8")
        md = [f"# Worked customers — {a.segments_model} on {a.split}, outcome `{a.outcome}`", "",
              f"Features f0–f11 are Criteo's anonymous columns. δ = {delta:.5f}, p0_high = {p0_high:.4f}, treat/skip at the 20% budget "
              f"(threshold {thr:.5f}).", "", wc.to_markdown(index=False), ""]
        (a.out / f"worked_customers_{a.split}_{a.outcome}.md").write_text("\n".join(md), encoding="utf-8")
    (a.out / f"policy_{a.split}_{a.outcome}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("ate", "random_per_1000", "economics", "propensity", "raw_vs_ipw", "breakeven")}, indent=2))
    for col in ("incremental_per_1000", "ipw_per_1000", "profit_per_1000_customers"):
        print(f"\n{col} by budget:\n", table.pivot(index="model", columns="budget", values=col).round(2).to_string())
    if "segments" in out:
        print("\nsegments:", json.dumps(out["segments"], indent=2))
        print("\nworked customers:\n", pd.DataFrame(out["worked_customers"]["rows"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
