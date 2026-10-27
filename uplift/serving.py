"""Serving core: the deployed pair (plus the response model for the UI comparison) loaded once; thresholds, policy numbers and
segment cut-offs from the committed dev grid in models/serving/policy_grid_dev_visit.json. Nothing here reads a data split.

The budget in force comes from the BUDGET environment variable (default 0.10, the headline operating point) and can be
overridden per request. The τ̂ < −δ segment is served as "no_detectable_effect" with SEGMENT_NOTE (CLAUDE.md §0, M5).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from .config import MODELS, ROOT
from .policy import SEGMENTS, segment
from .schema import SEGMENT_NOTE, Customer, UpliftResponse

SERVING = MODELS / "serving"
DEPLOYED = {"s_learner": "s_learner-visit.joblib", "r_learner": "r_learner-visit.joblib"}
COMPARISON = {"response": "response-visit.joblib"}
API_SEGMENT = {"sleeping_dog": "no_detectable_effect"}
DEFAULT_BUDGET = 0.10
MIN_BUDGET, MAX_BUDGET = 0.01, 0.50
OUTCOME = "visit"


def version(root: Path = ROOT, serving: Path = SERVING) -> str:
    """APP_VERSION env var, else models/serving/version.json (written by scripts/export_serving.py), else the git short hash, else 'dev'."""
    if os.environ.get("APP_VERSION"):
        return os.environ["APP_VERSION"]
    f = serving / "version.json"
    if f.exists():
        try:
            v = json.loads(f.read_text(encoding="utf-8")).get("version")
            if v:
                return str(v)
        except (json.JSONDecodeError, AttributeError):
            pass
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=3, check=False).stdout.strip()
        return out or "dev"
    except (OSError, subprocess.SubprocessError):
        return "dev"


def budget_from_env(default: float = DEFAULT_BUDGET) -> float:
    raw = os.environ.get("BUDGET")
    if not raw:
        return default
    try:
        b = float(raw)
    except ValueError:
        return default
    return b if MIN_BUDGET <= b <= MAX_BUDGET else default


def clamp_budget(b: float) -> float:
    return min(max(round(float(b), 2), MIN_BUDGET), MAX_BUDGET)


@dataclass
class Scorer:
    learners: dict
    grid: dict
    budget: float = DEFAULT_BUDGET
    version: str = "dev"
    loaded_at: float = field(default_factory=time.time)

    @classmethod
    def load(cls, models_dir: Path = MODELS, serving_dir: Path = SERVING, budget: float | None = None) -> Scorer:
        learners = {}
        for name, fname in {**DEPLOYED, **COMPARISON}.items():
            p = models_dir / fname
            if p.exists():
                learners[name] = joblib.load(p)["learner"]
        missing = [n for n in DEPLOYED if n not in learners]
        if missing:
            raise FileNotFoundError(f"deployed bundle(s) missing in {models_dir}: {missing}")
        grid = json.loads((serving_dir / "policy_grid_dev_visit.json").read_text(encoding="utf-8"))
        return cls(learners=learners, grid=grid, budget=budget if budget is not None else budget_from_env(), version=version(serving=serving_dir))

    @property
    def models(self) -> list[str]:
        return [n for n in DEPLOYED if n in self.learners]

    @property
    def delta(self) -> float:
        return float(self.grid["segments"]["delta"])

    @property
    def p0_high(self) -> float:
        return float(self.grid["segments"]["p0_high"])

    def row(self, model: str, budget: float | None = None) -> dict:
        """The dev policy-grid row (1% steps) nearest the budget."""
        b = clamp_budget(self.budget if budget is None else budget)
        return min(self.grid["models"][model]["grid"], key=lambda r: abs(r["budget"] - b))

    def threshold(self, model: str, budget: float | None = None) -> float:
        return float(self.row(model, budget)["threshold"])

    def _matrix(self, customers: list[Customer]) -> np.ndarray:
        return np.asarray([c.row() for c in customers], dtype=np.float32)

    def score_many(self, model: str, customers: list[Customer], budget: float | None = None) -> list[UpliftResponse]:
        if model not in DEPLOYED:
            raise ValueError(f"model must be one of {list(DEPLOYED)}")
        t0 = time.perf_counter()
        b = clamp_budget(self.budget if budget is None else budget)
        X = self._matrix(customers)
        learner = self.learners[model]
        p0, p1 = learner.predict_outcomes(X)
        tau = p1 - p0 if model == "s_learner" else learner.predict_uplift(X)  # the S-learner's uplift is exactly f(x,1) − f(x,0)
        thr = self.threshold(model, b)
        seg = segment(p0, tau, self.delta, self.p0_high)
        ms = (time.perf_counter() - t0) * 1000
        out = []
        for i in range(len(customers)):
            name = API_SEGMENT.get(str(seg[i]), str(seg[i]))
            out.append(UpliftResponse(uplift=round(float(tau[i]), 6), p_control=round(float(np.clip(p0[i], 0, 1)), 6), p_treated=round(float(np.clip(p1[i], 0, 1)), 6),
                                      segment=name, segment_note=SEGMENT_NOTE if name == "no_detectable_effect" else None,
                                      decision="treat" if tau[i] >= thr else "skip", budget=b, threshold=round(thr, 7), outcome=OUTCOME, model=model,
                                      version=self.version, latency_ms=round(ms, 2)))
        return out

    def score(self, model: str, customer: Customer, budget: float | None = None) -> UpliftResponse:
        return self.score_many(model, [customer], budget)[0]

    def score_response(self, customers: list[Customer], budget: float | None = None) -> list[dict]:
        """The response model, for the UI comparison only: its ranking score P(visit | treated) and its own decision at the budget."""
        if "response" not in self.learners:
            return []
        t0 = time.perf_counter()
        b = clamp_budget(self.budget if budget is None else budget)
        _, p1 = self.learners["response"].predict_outcomes(self._matrix(customers))
        thr = self.threshold("response", b)
        ms = (time.perf_counter() - t0) * 1000
        return [{"score": float(p1[i]), "decision": "treat" if p1[i] >= thr else "skip", "threshold": thr, "budget": b, "latency_ms": round(ms, 2)} for i in range(len(customers))]

    def policy(self, budget: float | None = None, model: str = "s_learner") -> dict:
        r = self.row(model, budget)
        be = self.grid["models"][model]["breakeven"]
        segs = r.get("segments_among_targeted")
        return {"budget": r["budget"], "model": model, "threshold": r["threshold"], "incremental_per_1000": r["per_1000"], "ipw_per_1000": r["ipw_per_1000"],
                "share_of_total": r["share_of_total"], "profit_per_1000_contacts": r["profit_per_1000_contacts"], "ipw_profit_per_1000_contacts": r["ipw_profit_per_1000_contacts"],
                "profit_per_1000_customers": r["profit_per_1000_customers"], "ipw_profit_per_1000_customers": r["ipw_profit_per_1000_customers"],
                "breakeven_budget": be["raw"]["breakeven_budget"], "profit_max_budget": be["raw"]["profit_max_budget"], "random_per_1000": self.grid["random_per_1000"],
                "economics": self.grid["economics"], "segments_among_targeted": {API_SEGMENT.get(k, k): v for k, v in segs.items()} if segs else None,
                "source": f"dev split ({self.grid['n']:,} customers), budgets on a 1% grid; the test-split results are in the README and DESIGN.md §5",
                "version": self.version}

    def health(self) -> dict:
        return {"status": "ok", "outcome": OUTCOME, "models": self.models, "comparison_models": [n for n in COMPARISON if n in self.learners],
                "budget": self.budget, "thresholds_at_budget": {m: self.threshold(m) for m in self.models},
                "segments": {"names": [API_SEGMENT.get(n, n) for n in SEGMENTS], "delta": self.delta, "p0_high": self.p0_high, "note": SEGMENT_NOTE},
                "version": self.version, "grid_built": self.grid.get("built"), "loaded_at": self.loaded_at}
