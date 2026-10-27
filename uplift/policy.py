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


# --- M3: budget curves with CIs, raw and IPW-weighted; break-even; segment report ----------------------------------
BUDGET_GRID = tuple(round(b, 2) for b in np.arange(0.05, 0.501, 0.05))


def budget_curves(y, t, scores_by_model: dict, e_hat, budgets=BUDGET_GRID, econ: Economics = ECONOMICS, value_per_response: float | None = None,
                  n_resamples: int = 200, seed: int = 7) -> pd.DataFrame:
    """One row per (model, budget): raw and IPW-weighted incremental responses per 1,000 contacts with CIs, the share of
    the total incremental effect, profit per 1,000 contacts and cumulative profit per 1,000 customers of the population,
    under the stated economics (raw and IPW). `random` is added as the reference."""
    from .metrics import budget_block

    value = value_per_response if value_per_response is not None else econ.value_per_visit
    n = len(np.asarray(y))
    rng = np.random.default_rng(0)
    models = {"random": rng.random(n), **scores_by_model}
    rows = []
    for name, s in models.items():
        blk = budget_block(y, t, s, e_hat, budgets, n_resamples, seed)
        for b in budgets:
            raw, ipw, k = blk["raw"][str(b)], blk["ipw"][str(b)], blk["n_targeted"][str(b)]
            rows.append({"model": name, "budget": b, "n_targeted": k,
                         "incremental_per_1000": raw["point"], "ci_low": raw["ci_low"], "ci_high": raw["ci_high"],
                         "ipw_per_1000": ipw["point"], "ipw_ci_low": ipw["ci_low"], "ipw_ci_high": ipw["ci_high"],
                         "share_of_total": blk["share_of_total"][str(b)],
                         "profit_per_1000_contacts": value * raw["point"] - econ.contact_cost * 1000,
                         "ipw_profit_per_1000_contacts": value * ipw["point"] - econ.contact_cost * 1000,
                         "profit_per_1000_customers": (value * raw["point"] - econ.contact_cost * 1000) * k / n,
                         "ipw_profit_per_1000_customers": (value * ipw["point"] - econ.contact_cost * 1000) * k / n})
    return pd.DataFrame(rows)


def breakeven(y, t, score, e_hat, econ: Economics = ECONOMICS, value_per_response: float | None = None, step: float = 0.01, max_budget: float = 0.5) -> dict:
    """On a fine budget grid (point estimates): the profit-maximising budget, the cumulative profit there, and the
    break-even budget (the largest budget at which the cumulative profit is still ≥ 0), raw and IPW-weighted. The
    marginal threshold is contact_cost / value: below that many incremental responses per contact, one more contact loses money."""
    from .metrics import budget_block

    value = value_per_response if value_per_response is not None else econ.value_per_visit
    grid = [round(i * step, 4) for i in range(1, int(round(max_budget / step)) + 1)]
    n = len(np.asarray(y))
    blk = budget_block(y, t, score, e_hat, grid, n_resamples=0)
    out = {"marginal_threshold_per_1000": econ.contact_cost / value * 1000, "grid_step": step}
    for key in ("raw", "ipw"):
        prof = {b: (value * blk[key][str(b)]["point"] - econ.contact_cost * 1000) * blk["n_targeted"][str(b)] / n for b in grid}
        best = max(prof, key=prof.get)
        profitable = [b for b in grid if prof[b] >= 0]
        out[key] = {"profit_max_budget": best, "max_profit_per_1000_customers": round(prof[best], 3),
                    "breakeven_budget": max(profitable) if profitable else None, "per_1000_at_profit_max": round(blk[key][str(best)]["point"], 2)}
    return out


def segment_report(y, t, p0, tau, delta: float, p0_high: float, min_rows: int = 200) -> dict:
    """Shares of the four segments, the observed uplift (treated minus control rate, with CI) inside each, and the same for
    the group with a negative predicted uplift — the sleeping-dog test the randomised data allows."""
    from .data import ate

    y, t, p0, tau = np.asarray(y, dtype=float), np.asarray(t).astype(int), np.asarray(p0, dtype=float), np.asarray(tau, dtype=float)
    seg = segment(p0, tau, delta, p0_high)

    def observed(mask):
        base = {"n": int(mask.sum()), "share": round(float(mask.mean()), 4), "n_control": int((t[mask] == 0).sum()),
                "mean_tau_hat": round(float(tau[mask].mean()), 5) if mask.any() else None, "mean_p0_hat": round(float(p0[mask].mean()), 5) if mask.any() else None}
        if mask.sum() >= min_rows and len(np.unique(t[mask])) == 2:
            base.update(ate(y[mask], t[mask]))
        return base

    return {"delta": delta, "p0_high": p0_high, "shares": segment_shares(p0, tau, delta, p0_high),
            "observed": {k: observed(seg == k) for k in SEGMENTS}, "tau_negative": observed(tau < 0)}
