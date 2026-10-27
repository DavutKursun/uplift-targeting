"""The API through FastAPI's TestClient (no network, no data splits): a valid score with a decision consistent with the threshold,
schema rejection of extra or malformed fields, the batch limit, the policy endpoint monotone in the budget, health, the Gradio mount."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("lightgbm")
from uplift.config import FEATURES  # noqa: E402
from uplift.schema import SEGMENT_NOTE  # noqa: E402
from uplift.serving import SERVING  # noqa: E402

import app as app_module  # noqa: E402  (loads the deployed bundles from models/)

client = TestClient(app_module.app)
SAMPLE = pd.read_csv(SERVING / "customers_dev_sample.csv")
CUSTOMER = {f: float(SAMPLE.iloc[0][f]) for f in FEATURES}
SEGMENTS = {"persuadable", "sure_thing", "lost_cause", "no_detectable_effect"}


@pytest.mark.parametrize("model", ["s_learner", "r_learner"])
def test_score_valid_customer(model):
    r = client.post(f"/score?model={model}", json=CUSTOMER)
    assert r.status_code == 200, r.text
    b = r.json()
    assert set(b) >= {"uplift", "p_control", "p_treated", "segment", "decision", "budget", "threshold", "outcome", "model", "version", "latency_ms"}
    assert b["decision"] == ("treat" if b["uplift"] >= b["threshold"] else "skip")
    assert b["segment"] in SEGMENTS and (b["segment_note"] == SEGMENT_NOTE) == (b["segment"] == "no_detectable_effect")
    assert 0 <= b["p_control"] <= 1 and 0 <= b["p_treated"] <= 1 and b["outcome"] == "visit" and b["model"] == model
    assert b["budget"] == app_module.scorer.budget and b["threshold"] == app_module.scorer.threshold(model) and b["latency_ms"] > 0 and b["version"]


def test_score_honours_the_budget_query():
    tight, wide = client.post("/score?budget=0.05", json=CUSTOMER).json(), client.post("/score?budget=0.40", json=CUSTOMER).json()
    assert tight["budget"] == 0.05 and wide["budget"] == 0.4 and tight["threshold"] > wide["threshold"]


def test_invalid_and_extra_fields_are_rejected():
    assert client.post("/score", json={**CUSTOMER, "treatment": 1}).status_code == 422
    assert client.post("/score", json={**CUSTOMER, "made_up": 0.0}).status_code == 422
    assert client.post("/score", json={**CUSTOMER, "f3": "high"}).status_code == 422
    assert client.post("/score", json={k: v for k, v in CUSTOMER.items() if k != "f11"}).status_code == 422
    assert client.post("/score?model=response", json=CUSTOMER).status_code == 422
    assert client.post("/score?budget=0.8", json=CUSTOMER).status_code == 422


def test_batch_limit_and_results():
    rows = [{f: float(SAMPLE.iloc[i][f]) for f in FEATURES} for i in range(3)]
    r = client.post("/score/batch?model=s_learner", json={"customers": rows})
    assert r.status_code == 200 and r.json()["n"] == 3 and len(r.json()["results"]) == 3 and r.json()["model"] == "s_learner"
    assert all(x["decision"] == ("treat" if x["uplift"] >= x["threshold"] else "skip") for x in r.json()["results"])
    assert client.post("/score/batch", json={"customers": [CUSTOMER] * 1001}).status_code == 422
    assert client.post("/score/batch", json={"customers": []}).status_code == 422
    assert client.post("/score/batch", json={"customers": rows, "extra": 1}).status_code == 422


def test_policy_monotone_in_budget():
    ps = [client.get(f"/policy?budget={b}").json() for b in (0.05, 0.10, 0.30)]
    assert all(p["model"] == "s_learner" for p in ps)
    assert ps[0]["threshold"] > ps[1]["threshold"] > ps[2]["threshold"]
    assert ps[0]["incremental_per_1000"] > ps[1]["incremental_per_1000"] > ps[2]["incremental_per_1000"] > ps[2]["random_per_1000"]
    assert {"ipw_per_1000", "profit_per_1000_contacts", "breakeven_budget", "profit_max_budget", "economics", "segments_among_targeted"} <= set(ps[1])
    assert ps[1]["economics"]["marginal_threshold_per_1000"] == 50.0 and abs(sum(ps[1]["segments_among_targeted"].values()) - 1) < 1e-6
    assert set(ps[1]["segments_among_targeted"]) == SEGMENTS
    assert client.get("/policy?budget=0.10&model=r_learner").json()["model"] == "r_learner"
    assert client.get("/policy?budget=0.7").status_code == 422 and client.get("/policy?model=response").status_code == 422


def test_health():
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["models"] == ["s_learner", "r_learner"] and h["outcome"] == "visit" and h["version"]
    assert 0.01 <= h["budget"] <= 0.5 and set(h["thresholds_at_budget"]) == {"s_learner", "r_learner"} and h["segments"]["note"] == SEGMENT_NOTE


def test_gradio_ui_is_mounted():
    r = client.get("/ui/")
    assert r.status_code == 200 and "gradio" in r.text.lower()
    cfg = client.get("/ui/config")
    assert cfg.status_code == 200 and "components" in cfg.json()
