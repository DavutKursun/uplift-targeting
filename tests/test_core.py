from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from uplift.config import FEATURES, TREATMENT
from uplift.data import (
    COLUMNS,
    assign_splits,
    ate,
    csv_to_parquet,
    load_split,
    randomization_check,
    save_splits,
    stratified_subsample,
    summary,
)
from uplift.learners import LEARNERS, UPLIFT_LEARNERS
from uplift.metrics import auuc, bootstrap, curves, pehe, perfect_score, policy_value, qini_coefficient, spearman, uplift_by_bin
from uplift.policy import budget_table, decide, segment, segment_shares, threshold_for_budget
from uplift.schema import Customer, UpliftResponse
from uplift.synthetic import write_raw_csv


# --- data ---------------------------------------------------------------------
def test_synthetic_world_has_the_four_segments(world):
    df = world["df"]
    assert 0.8 < df[TREATMENT].mean() < 0.9
    assert (df["tau_true"] < 0).mean() > 0.03  # sleeping dogs exist
    assert (df["tau_true"] > 0.03).mean() > 0.1  # persuadables exist
    assert df.loc[df["p0_true"] > 0.3, "tau_true"].mean() < df["tau_true"].mean()  # sure things move less


def test_ate_recovers_truth_and_randomization_is_balanced(world):
    df = world["df"]
    a = ate(df["visit"], df[TREATMENT])
    assert a["ci_low"] <= df["tau_true"].mean() <= a["ci_high"]
    assert a["relative_uplift"] > 0 and a["n_control"] > 5_000
    rc = randomization_check(df)
    assert rc["abs_smd"].max() < 0.05 and set(rc["feature"]) == set(FEATURES)


def test_splits_are_stratified_and_disjoint(world):
    df = world["df"]
    for s in ("train", "dev", "test"):
        part = df[df["split"] == s]
        assert abs(part[TREATMENT].mean() - df[TREATMENT].mean()) < 0.01
        assert abs(part["visit"].mean() - df["visit"].mean()) < 0.01
    assert abs((df["split"] == "train").mean() - 0.6) < 0.01
    sub = stratified_subsample(df, 20_000)
    assert len(sub) == pytest.approx(20_000, abs=50) and abs(sub[TREATMENT].mean() - df[TREATMENT].mean()) < 0.01
    s = summary(df)
    assert set(s["splits"]) == {"train", "dev", "test"} and "visit" in s and "conversion" in s


def test_csv_roundtrip_in_chunks(tmp_path, world):
    df = world["df"].head(5_000)
    csv = write_raw_csv(df, tmp_path / "raw" / "criteo.csv.gz")
    n = csv_to_parquet(csv, tmp_path / "cache.parquet", chunksize=1_000)
    back = pd.read_parquet(tmp_path / "cache.parquet")
    assert n == 5_000 and list(back.columns) == COLUMNS and str(back["f0"].dtype) == "float32" and str(back["visit"].dtype) == "int8"
    back["split"] = assign_splits(back)
    summ = save_splits(back, tmp_path / "splits")
    assert summ["rows"] == 5_000 and len(load_split("train", tmp_path / "splits")) == summ["splits"]["train"]["rows"]
    with pytest.raises(FileNotFoundError):
        load_split("test", tmp_path / "nope")


# --- learners against the truth ------------------------------------------------
def test_uplift_learners_recover_the_true_effect(world, fitted):
    rows = {}
    for name in UPLIFT_LEARNERS:
        s = fitted[name].predict_uplift(world["Xd"])
        rows[name] = (spearman(world["truth"], s), pehe(world["truth"], s))
        assert rows[name][0] > 0.45, (name, rows[name])
        assert rows[name][1] < 0.07, (name, rows[name])
    best = max(rows, key=lambda k: rows[k][0])
    assert rows[best][0] > 0.7


def test_response_model_targets_sure_things_not_persuadables(world, fitted):
    s = fitted["response"].predict_uplift(world["Xd"])
    assert spearman(world["truth"], s) < 0.25
    assert spearman(world["p0"], s) > 0.6  # it ranks mostly by base rate
    q_resp = qini_coefficient(world["yd"], world["td"], s)
    q_best = max(qini_coefficient(world["yd"], world["td"], fitted[n].predict_uplift(world["Xd"])) for n in UPLIFT_LEARNERS)
    assert q_best > q_resp + 0.03


def test_predict_outcomes_are_probabilities(world, fitted):
    for name, m in fitted.items():
        p0, p1 = m.predict_outcomes(world["Xd"][:500])
        assert p0.shape == p1.shape == (500,) and (0 <= p0).all() and (p0 <= 1).all() and (0 <= p1).all() and (p1 <= 1).all()
        if name != "response":
            assert np.allclose(p1 - p0, np.clip(p0 + m.predict_uplift(world["Xd"][:500]), 0, 1) - p0, atol=1e-6)


def test_learner_registry_and_pickling(world, fitted, tmp_path):
    import joblib

    assert set(LEARNERS) == {"response", "s_learner", "t_learner", "x_learner", "transformed_outcome", "r_learner", "dr_learner"}
    m = fitted["dr_learner"]
    joblib.dump(m, tmp_path / "m.joblib")
    m2 = joblib.load(tmp_path / "m.joblib")
    assert np.allclose(m2.predict_uplift(world["Xd"][:100]), m.predict_uplift(world["Xd"][:100]))


# --- metrics -------------------------------------------------------------------
def test_qini_truth_beats_random_and_perfect_is_one(world):
    yd, td, truth = world["yd"], world["td"], world["truth"]
    rng = np.random.default_rng(0)
    q_truth, q_rand = qini_coefficient(yd, td, truth), qini_coefficient(yd, td, rng.random(len(yd)))
    assert q_truth > 0.05 and abs(q_rand) < 0.06 and q_truth > q_rand + 0.02  # random Qini is noise around 0 at 18k rows
    assert qini_coefficient(yd, td, perfect_score(yd, td)) == pytest.approx(1.0, abs=1e-9)
    assert auuc(yd, td, truth) > auuc(yd, td, rng.random(len(yd)))
    c = curves(yd, td, truth)
    assert c["fraction"].iloc[0] == 0 and c["fraction"].iloc[-1] == pytest.approx(1.0) and len(c) > 50


def test_constant_score_behaves_like_random(world):
    q = qini_coefficient(world["yd"], world["td"], np.zeros(len(world["yd"])))
    assert abs(q) < 0.06


def test_decile_table_and_policy_value(world):
    yd, td, truth = world["yd"], world["td"], world["truth"]
    tab = uplift_by_bin(yd, td, truth, predicted=truth)
    assert len(tab) == 10 and tab["n"].sum() == len(yd)
    assert tab["uplift_observed"].iloc[0] > tab["uplift_observed"].iloc[-1]  # top bin above bottom bin
    assert tab["uplift_observed"].iloc[-1] < 0  # sleeping dogs at the bottom
    pv = policy_value(yd, td, truth, 0.2)
    assert pv["n_targeted"] == round(0.2 * len(yd)) and pv["incremental_per_1000"] > pv["random_per_1000"] > 0
    assert 0 < pv["share_of_total"] <= 1.5


def test_bootstrap_ci_contains_point(world):
    yd, td, truth = world["yd"], world["td"], world["truth"]
    b = bootstrap(qini_coefficient, yd, td, truth, n_resamples=30)
    assert b["ci_low"] <= b["point"] <= b["ci_high"] and b["n_resamples"] == 30


def test_placebo_shuffled_treatment_gives_zero_qini(world):
    from uplift.learners import fast_settings, fit_learner

    rng = np.random.default_rng(1)
    m = fit_learner("transformed_outcome", world["X"], rng.permutation(world["t"]), world["y"], propensity=float(world["t"].mean()), settings=fast_settings(60))
    q = bootstrap(qini_coefficient, world["yd"], world["td"], m.predict_uplift(world["Xd"]), n_resamples=40)
    assert q["ci_low"] <= 0.0 <= q["ci_high"] or abs(q["point"]) < 0.03


# --- policy -------------------------------------------------------------------
def test_threshold_decisions_and_segments(world, fitted):
    s = fitted["s_learner"].predict_uplift(world["Xd"])
    thr = threshold_for_budget(s, 0.2)
    d = decide(s, thr)
    assert abs((d == "treat").mean() - 0.2) < 0.01
    p0, _ = fitted["s_learner"].predict_outcomes(world["Xd"])
    seg = segment(p0, s, delta=0.005, p0_high=0.13)
    shares = segment_shares(p0, s, 0.005, 0.13)
    assert set(np.unique(seg)) <= {"persuadable", "sure_thing", "lost_cause", "sleeping_dog"} and abs(sum(shares.values()) - 1) < 1e-6
    assert shares["persuadable"] > 0.1 and shares["sleeping_dog"] > 0.01
    tab = budget_table(world["yd"], world["td"], {"s_learner": s}, budgets=(0.1, 0.2))
    assert set(tab["model"]) == {"random", "s_learner"} and tab["profit"].notna().all()
    model_20 = tab[(tab["model"] == "s_learner") & (tab["budget"] == 0.2)].iloc[0]
    random_20 = tab[(tab["model"] == "random") & (tab["budget"] == 0.2)].iloc[0]
    assert model_20["incremental_per_1000"] > random_20["incremental_per_1000"]


# --- schema -------------------------------------------------------------------
def test_schema():
    c = Customer(**{f"f{i}": float(i) for i in range(12)})
    assert c.row() == [float(i) for i in range(12)]
    with pytest.raises(ValidationError):
        Customer(**{f"f{i}": 0.0 for i in range(12)}, treatment=1)  # outcome/treatment fields are not inputs
    r = UpliftResponse(uplift=0.02, p_control=0.04, p_treated=0.06, segment="persuadable", decision="treat", budget=0.2, threshold=0.01,
                       outcome="visit", model="r_learner", version="abc", latency_ms=3.2)
    assert r.decision == "treat"


# --- categorical columns and τ rounds (M2 ablation machinery) ----------------------------
def test_categorical_codes_and_tau_rounds(world):
    from uplift.learners import CategoricalCodes, categorical_columns, fast_settings, fit_learner

    X = world["X"]
    cols = categorical_columns(X, 50)  # the synthetic world's coded columns: f4 (40 values), f5 (7), f9 (5)
    assert cols == (4, 5, 9)
    codes = CategoricalCodes(cols).fit(X)
    Z = codes.transform(X[:10])
    assert Z.shape == X[:10].shape and (Z[:, 5] >= 0).all() and Z[:, 5].max() < 7 and np.array_equal(Z[:, 0], X[:10, 0])
    unseen = X[:1].copy()
    unseen[0, 5] = 99.0
    assert codes.transform(unseen)[0, 5] == -1  # unseen value → missing for LightGBM
    s = fast_settings()
    s.categorical, s.tau_rounds = cols, 30
    m = fit_learner("dr_learner", X, world["t"], world["y"], settings=s)
    assert m.codes is not None and m.tau.num_trees() == 30
    u = m.predict_uplift(world["Xd"])
    assert np.isfinite(u).all() and spearman(world["truth"], u) > 0.3
    p0, p1 = m.predict_outcomes(world["Xd"][:50])
    assert p0.shape == (50,) and (0 <= p0).all() and (p1 <= 1).all()


# --- M3: paired differences, IPW policy value, budget curves, segments ------------------------------------------
def test_paired_and_budget_blocks(world):
    from uplift.metrics import bootstrap_block, budget_block, paired_block

    yd, td, truth = world["yd"], world["td"], world["truth"]
    rng = np.random.default_rng(3)
    scores = {"a": truth, "a_copy": truth.copy(), "rand": rng.random(len(yd))}
    res = paired_block(yd, td, scores, [("a", "a_copy"), ("a", "rand")], budgets=(0.1, 0.2), n_resamples=25)
    assert all(v["point"] == 0 and v["ci_low"] == 0 and v["ci_high"] == 0 and not v["excludes_zero"] for v in res["pairs"]["a-a_copy"].values())
    better = res["pairs"]["a-rand"]
    assert better["qini"]["point"] > 0 and better["qini"]["ci_low"] <= better["qini"]["point"] <= better["qini"]["ci_high"]
    single = bootstrap_block(yd, td, truth, (0.1, 0.2), 25)  # same resamples, same numbers
    assert np.isclose(res["models"]["a"]["qini"]["point"], single["qini"]["point"]) and np.isclose(res["models"]["a"]["qini"]["ci_low"], single["qini"]["ci_low"])
    assert np.isclose(res["models"]["a"]["0.2"]["point"], single["policy"]["0.2"]["point"]) and np.isclose(res["models"]["a"]["0.2"]["ci_high"], single["policy"]["0.2"]["ci_high"])
    e = np.full(len(yd), td.mean())  # a constant propensity makes the IPW estimate the raw one
    blk = budget_block(yd, td, truth, e, (0.1, 0.3), n_resamples=10)
    for b in ("0.1", "0.3"):
        assert np.isclose(blk["raw"][b]["point"], blk["ipw"][b]["point"]) and blk["raw"][b]["ci_low"] <= blk["raw"][b]["point"] <= blk["raw"][b]["ci_high"]
    assert 0 < blk["share_of_total"]["0.3"] <= 1.5 and blk["n_targeted"]["0.1"] == round(0.1 * len(yd))


def test_budget_curves_breakeven_and_segment_report(world, fitted):
    from uplift.learners import fast_settings, propensity_oof
    from uplift.policy import breakeven, budget_curves, segment_report

    yd, td, Xd = world["yd"], world["td"], world["Xd"]
    e = propensity_oof(Xd, td, fast_settings(40))
    assert e.shape == td.shape and (0.02 <= e).all() and (e <= 0.98).all() and abs(e.mean() - td.mean()) < 0.05
    s = fitted["s_learner"].predict_uplift(Xd)
    tab = budget_curves(yd, td, {"s_learner": s}, e, budgets=(0.1, 0.2), n_resamples=10)
    assert set(tab["model"]) == {"random", "s_learner"} and len(tab) == 4
    row = tab[(tab["model"] == "s_learner") & (tab["budget"] == 0.1)].iloc[0]
    assert row["ci_low"] <= row["incremental_per_1000"] <= row["ci_high"] and np.isfinite(row["ipw_per_1000"]) and np.isfinite(row["profit_per_1000_customers"])
    be = breakeven(yd, td, s, e, step=0.05, max_budget=0.5)
    assert be["marginal_threshold_per_1000"] == 50.0 and 0.05 <= be["raw"]["profit_max_budget"] <= 0.5 and "breakeven_budget" in be["ipw"]
    p0, _ = fitted["s_learner"].predict_outcomes(Xd)
    rep = segment_report(yd, td, p0, s, delta=0.005, p0_high=0.13)
    assert abs(sum(rep["shares"].values()) - 1) < 1e-6 and set(rep["observed"]) == {"persuadable", "sure_thing", "lost_cause", "sleeping_dog"}
    assert rep["observed"]["persuadable"]["n"] > 1000 and "ate" in rep["observed"]["persuadable"] and "share" in rep["tau_negative"]
