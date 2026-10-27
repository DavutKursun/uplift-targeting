"""The script chain on a synthetic export: build_splits → ate_report → synthetic_check → train → evaluate (dev; test refused)
→ policy_eval → placebo. Same chain as on the real file, so the first real run is not the first run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ate_report
import build_splits
import evaluate
import paired
import placebo
import plots
import policy_eval
import synthetic_check
import train

from uplift.synthetic import make_campaign, write_raw_csv


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    base = tmp_path_factory.mktemp("ws")
    raw = write_raw_csv(make_campaign(n=40_000, seed=21), base / "raw" / "criteo-uplift-v2.1.csv.gz")
    d = {"raw": raw, "cache": base / "raw" / "criteo.parquet", "splits": base / "splits", "models": base / "models", "out": base / "results"}
    for k in ("splits", "models", "out"):
        d[k].mkdir(parents=True, exist_ok=True)
    return d


def test_build_splits_and_ate(ws):
    assert build_splits.main(["--raw", str(ws["raw"]), "--cache", str(ws["cache"]), "--out", str(ws["splits"]), "--rows", "30000"]) == 0
    s = json.loads((ws["splits"] / "summary.json").read_text())
    assert s["source_rows"] == 40_000 and abs(s["rows"] - 30_000) <= 50 and set(s["splits"]) == {"train", "dev", "test"}
    assert ate_report.main(["--splits-dir", str(ws["splits"]), "--out", str(ws["out"])]) == 0
    a = json.loads((ws["out"] / "ate_train.json").read_text())
    assert a["randomization"]["balanced"] and a["visit"]["ate"] > 0


def test_synthetic_check_script(ws):
    assert synthetic_check.main(["--n", "30000", "--fast", "--learners", "response,s_learner", "--out", str(ws["out"])]) == 0
    r = json.loads((ws["out"] / "synthetic_check.json").read_text())
    names = {row["learner"]: row for row in r["rows"]}
    assert names["s_learner"]["spearman_with_truth"] > names["response"]["spearman_with_truth"]


def test_train_evaluate_policy_placebo(ws):
    common = ["--splits-dir", str(ws["splits"]), "--models-dir", str(ws["models"]), "--out", str(ws["out"])]
    assert train.main(["--learners", "response,s_learner,r_learner", "--fast", *common]) == 0
    res = json.loads((ws["out"] / "train_dev_visit.json").read_text())
    assert {"response", "s_learner", "r_learner"} <= set(res)
    assert evaluate.main(["--split", "test", *common]) == 3
    assert evaluate.main(["--split", "dev", "--resamples", "20", *common]) == 0
    summ = json.loads((ws["out"] / "dev-summary-visit.json").read_text())
    assert {m["model"] for m in summ["metrics"]} == {"response", "s_learner", "r_learner"} and (ws["out"] / "deciles_dev_visit_r_learner.csv").exists()
    assert policy_eval.main(["--split", "test", *common]) == 3
    assert policy_eval.main(["--split", "dev", "--segments-model", "r_learner", "--fast", "--resamples", "10", *common]) == 0
    pol = json.loads((ws["out"] / "policy_dev_visit.json").read_text())
    assert "segments" in pol and abs(sum(pol["segments"]["shares"].values()) - 1) < 1e-6
    assert "breakeven" in pol and "raw_vs_ipw" in pol and len(pol["worked_customers"]["rows"]) >= 2 and (ws["out"] / "policy_dev_visit.csv").exists()
    assert paired.main(["--split", "test", *common]) == 3
    assert paired.main(["--split", "dev", "--pairs", "s_learner-response", "--resamples", "10", *common]) == 0
    pr = json.loads((ws["out"] / "paired_dev_visit.json").read_text())
    assert "s_learner-response" in pr["pairs"] and "qini" in pr["pairs"]["s_learner-response"]
    assert plots.main(["--split", "dev", "--best", "s_learner", *common]) == 0
    assert all((ws["out"] / f).exists() for f in ("qini_curves_dev.png", "deciles_dev_s_learner.png", "budget_curve_dev.png", "profit_curve_dev.png"))
    assert placebo.main(["--learner", "s_learner", "--fast", "--resamples", "20", "--splits-dir", str(ws["splits"]), "--out", str(ws["out"])]) == 0
    pl = json.loads((ws["out"] / "placebo_visit_s_learner.json").read_text())
    assert "qini_placebo" in pl
