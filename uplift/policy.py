"""From an uplift score to a decision: budget → threshold, segments, profit under stated economics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BUDGETS, ECONOMICS, Economics
from .metrics import policy_value

SEGMENTS = ("persuadable", "sure_thing", "lost_cause", "sleeping_dog")


def threshold_for_budget(scores, budget: float) -> float:
    """Score above which the top `budget` fraction sits."""
    return float(np.quantile(np.asarray(scores, dtype=float), 1 - budget))


def decide(scores, threshold: float) -> np.ndarray:
    return np.where(np.asarray(scores, dtype=float) >= threshold, "treat", "skip").astype(object)


def segment(p0, tau, delta: float, p0_high: float) -> np.ndarray:
    """persuadable: τ > δ · sleeping_dog: τ < −δ · otherwise sure_thing if P(Y|control) ≥ p0_high else lost_cause.
    δ and p0_high are stated choices (e.g. δ = a quarter of the ATE, p0_high = twice the control rate)."""
    p0, tau = np.asarray(p0, dtype=float), np.asarray(tau, dtype=float)
    out = np.where(tau > delta, "persuadable", np.where(tau < -delta, "sleeping_dog", np.where(p0 >= p0_high, "sure_thing", "lost_cause")))
    return out.astype(object)


def profit(incremental_responses: float, contacts: int, value_per_response: float, contact_cost: float) -> float:
    return float(incremental_responses * value_per_response - contacts * contact_cost)


def budget_table(y, t, scores_by_model: dict, budgets=BUDGETS, econ: Economics = ECONOMICS, value_per_response: float | None = None) -> pd.DataFrame:
    """One row per (model, budget): observed uplift among targeted, incremental per 1,000 contacts, share of the total
    incremental effect, profit under the stated economics. `random` is added as the reference."""
    value = value_per_response if value_per_response is not None else econ.value_per_visit
    n = len(np.asarray(y))
    rng = np.random.default_rng(0)
    models = {"random": rng.random(n), **scores_by_model}
    rows = []
    for name, s in models.items():
        for b in budgets:
            v = policy_value(y, t, s, b)
            inc = v["incremental_total"] if v["incremental_total"] is not None else np.nan
            rows.append({"model": name, "budget": b, "n_targeted": v["n_targeted"], "incremental_per_1000": v["incremental_per_1000"],
                         "share_of_total": v["share_of_total"], "profit": profit(inc, v["n_targeted"], value, econ.contact_cost) if not np.isnan(inc) else None,
                         "profit_per_1000_contacts": profit(inc, v["n_targeted"], value, econ.contact_cost) / v["n_targeted"] * 1000 if not np.isnan(inc) else None})
    return pd.DataFrame(rows)


def segment_shares(p0, tau, delta: float, p0_high: float) -> dict:
    s = segment(p0, tau, delta, p0_high)
    return {k: round(float(np.mean(s == k)), 4) for k in SEGMENTS}
