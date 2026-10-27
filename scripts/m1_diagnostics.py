"""M1 diagnostics behind DESIGN.md §1 and §2. Three sub-commands, all CPU-only, none touches the test split:

  facts      population facts on the full parquet cache (rates by arm, ATE with absolute and relative CIs, SMD table,
             the raw file's treated share by tenth)                    → eval/results/ate_full.json, randomization_full.csv
  balance    does the covariate imbalance move the ATE? on train: OLS, exact strata, cross-fitted propensity (AUC,
             range), Hájek IPW, AIPW                                    → eval/results/randomization_diagnostics_train.json
  synthetic  learner settings against the known truth: the loose starter values vs the committed defaults, at
             n = 200k and 1M, plus a balanced-arm variant               → eval/results/synthetic_settings.{json,csv}

Usage: python scripts/m1_diagnostics.py facts|balance|synthetic [--out eval/results] [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.config import OUTCOMES, RAW, RESULTS, SEED
from uplift.data import COLUMNS, assign_splits, ate, load_split, randomization_check, xty

LGB = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=2000, verbose=-1, seed=SEED)


def _relative_ci(r: dict, z: float = 1.96) -> dict:
    """Delta method on log(p1/p0)."""
    p1, p0, n1, n0 = r["rate_treated"], r["rate_control"], r["n_treated"], r["n_control"]
    se = np.sqrt((1 - p1) / (p1 * n1) + (1 - p0) / (p0 * n0))
    return {**r, "relative_ci_low": round(float(np.exp(np.log(p1 / p0) - z * se) - 1), 4),
            "relative_ci_high": round(float(np.exp(np.log(p1 / p0) + z * se) - 1), 4)}


def facts(out: Path, cache: Path = RAW / "criteo.parquet") -> dict:
    df = pd.read_parquet(cache, columns=COLUMNS)
    n, t = len(df), df["treatment"].to_numpy()
    tenth = n // 10
    res = {"rows": n, "treated_share": round(float(t.mean()), 5),
           "visit_rate_all": round(float(df["visit"].mean()), 6), "conversion_rate_all": round(float(df["conversion"].mean()), 6),
           "exposure_rate_treated": round(float(df.loc[t == 1, "exposure"].mean()), 5), "exposure_rate_control": round(float(df.loc[t == 0, "exposure"].mean()), 5),
           "treated_share_by_file_tenth": [round(float(t[i * tenth : (i + 1) * tenth].mean()), 3) for i in range(10)]}
    for o in OUTCOMES:
        res[o] = _relative_ci(ate(df[o], t))
    rc = randomization_check(df)
    res["randomization"] = {"max_abs_smd": round(float(rc["abs_smd"].max()), 5), "n_features_above_0_02": int((rc["abs_smd"] > 0.02).sum())}
    out.mkdir(parents=True, exist_ok=True)
    (out / "ate_full.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    rc.to_csv(out / "randomization_full.csv", index=False)
    print(json.dumps(res, indent=2))
    print(rc.to_string(index=False))
    return res


def _strata(df: pd.DataFrame, cols: list[str], outcome: str = "visit") -> dict:
    """ATE as the row-weighted mean of within-stratum differences, over strata that contain both arms (exact values, no binning)."""
    g = df.groupby(cols + ["treatment"], observed=True)[outcome].agg(["mean", "size"]).unstack("treatment")
    g = g[g["size"].notna().all(axis=1)]
    w = g["size"].sum(axis=1)
    w = w / w.sum()
    d = g["mean"][1] - g["mean"][0]
    var = g["mean"][1] * (1 - g["mean"][1]) / g["size"][1] + g["mean"][0] * (1 - g["mean"][0]) / g["size"][0]
    est, se = float((w * d).sum()), float(np.sqrt((w**2 * var).sum()))
    return {"ate": round(est, 6), "ci": [round(est - 1.96 * se, 6), round(est + 1.96 * se, 6)], "n_strata": len(g),
            "rows_covered_share": round(float(g["size"].sum().sum() / len(df)), 4)}


def balance(out: Path, split: str = "train") -> dict:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    df = load_split(split)
    n = len(df)
    X, t, y = xty(df, "visit")
    yf = y.astype(float)
    raw = ate(df["visit"], t)
    res = {"split": split, "rows": n, "visit_ate_raw": {"ate": raw["ate"], "ci": [raw["ci_low"], raw["ci_high"]]}}
    # OLS on treatment + standardized features, HC0 SE on the treatment coefficient
    Z = X.astype(np.float64)
    Z = (Z - Z.mean(0)) / Z.std(0)
    A = np.column_stack([np.ones(n), t, Z])
    beta, *_ = np.linalg.lstsq(A, yf, rcond=None)
    resid = yf - A @ beta
    inv = np.linalg.inv(A.T @ A)
    se = float(np.sqrt((inv @ ((A * resid[:, None]).T @ (A * resid[:, None])) @ inv)[1, 1]))
    res["visit_ate_ols_12_features"] = {"ate": round(float(beta[1]), 6), "ci": [round(float(beta[1] - 1.96 * se), 6), round(float(beta[1] + 1.96 * se), 6)]}
    for cols in (["f3"], ["f5"], ["f3", "f5"], ["f3", "f5", "f1"]):
        res["visit_ate_exact_strata_" + "_".join(cols)] = _strata(df, cols)
    top = df["f3"].value_counts().head(5)
    res["f3_top_values"] = [{"f3": float(v), "share": round(float(c / n), 4), "treated_share": round(float(df.loc[df.f3 == v, "treatment"].mean()), 4),
                             "visit_rate": round(float(df.loc[df.f3 == v, "visit"].mean()), 4)} for v, c in top.items()]
    # cross-fitted propensity and per-arm outcome models → AUC, Hájek IPW, AIPW
    t0 = time.time()
    e, mu0, mu1 = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, t * 2 + y):
        e[te] = lgb.train(LGB, lgb.Dataset(X[tr], t[tr]), 150).predict(X[te])
        mu0[te] = lgb.train(LGB, lgb.Dataset(X[tr][t[tr] == 0], y[tr][t[tr] == 0]), 150).predict(X[te])
        mu1[te] = lgb.train(LGB, lgb.Dataset(X[tr][t[tr] == 1], y[tr][t[tr] == 1]), 150).predict(X[te])
    res["treatment_from_features_auc_oof"] = round(float(roc_auc_score(t, e)), 4)
    e = np.clip(e, 0.05, 0.99)
    res["propensity_oof_p1_p99"] = [round(float(np.percentile(e, 1)), 4), round(float(np.percentile(e, 99)), 4)]
    w1, w0 = t / e, (1 - t) / (1 - e)
    res["visit_ate_hajek_ipw"] = round(float((w1 * yf).sum() / w1.sum() - (w0 * yf).sum() / w0.sum()), 6)
    phi = mu1 - mu0 + t * (yf - mu1) / e - (1 - t) * (yf - mu0) / (1 - e)
    a, s = float(phi.mean()), float(phi.std(ddof=1) / np.sqrt(n))
    res["visit_ate_aipw"] = {"ate": round(a, 6), "ci": [round(a - 1.96 * s, 6), round(a + 1.96 * s, 6)]}
    res["nuisance_seconds"] = round(time.time() - t0, 1)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"randomization_diagnostics_{split}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return res


def synthetic(out: Path, quick: bool = False) -> pd.DataFrame:
    from uplift.learners import CLS_PARAMS, LEARNERS, REG_PARAMS, Settings, fit_learner
    from uplift.metrics import pehe, qini_coefficient, spearman
    from uplift.synthetic import make_campaign

    loose = Settings(cls_params={**CLS_PARAMS, "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 500},
                     reg_params={**REG_PARAMS, "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 1000}, rounds=400)
    runs = [("loose", 200_000, 0.85, loose), ("default", 200_000, 0.85, None), ("loose", 200_000, 0.50, loose)]
    if not quick:
        runs += [("loose", 1_000_000, 0.85, loose), ("default", 1_000_000, 0.85, None)]
    rows = []
    for tag, n, share, settings in runs:
        df = make_campaign(n=n, seed=11, treated_share=share)
        df["split"] = assign_splits(df)
        tr, dv = df[df["split"] == "train"], df[df["split"] != "train"]
        X, t, y = xty(tr, "visit")
        Xd, td, yd = xty(dv, "visit")
        truth = dv["tau_true"].to_numpy()
        for name in LEARNERS:
            t0 = time.time()
            s = fit_learner(name, X, t, y, settings=settings).predict_uplift(Xd)
            rows.append({"settings": tag, "n": n, "treated_share": share, "learner": name, "spearman_with_truth": round(spearman(truth, s), 3),
                         "pehe": round(pehe(truth, s), 4), "qini": round(qini_coefficient(yd, td, s), 4), "fit_seconds": round(time.time() - t0, 1)})
            print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "synthetic_settings.csv", index=False)
    (out / "synthetic_settings.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(table.pivot_table(index=["n", "treated_share", "settings"], columns="learner", values="spearman_with_truth").to_string())
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["facts", "balance", "synthetic"])
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--quick", action="store_true", help="synthetic: n = 200k only")
    a = ap.parse_args(argv)
    {"facts": lambda: facts(a.out), "balance": lambda: balance(a.out), "synthetic": lambda: synthetic(a.out, a.quick)}[a.what]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
