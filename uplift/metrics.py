"""Evaluating uplift without counterfactuals: curves, Qini, deciles, policy value, bootstrap CIs — and PEHE
where a synthetic truth exists.

Everything here uses only observed outcomes and the randomized treatment flag, so the numbers are unbiased for
any targeting rule. Notation: sort customers by the model's score, descending; for the top fraction k,
N_t(k) / N_c(k) are treated / control counts and Y_t(k) / Y_c(k) the responders among them.

    uplift gain(k) = (Y_t/N_t − Y_c/N_c) · N(k)        incremental responses if the top-k were all treated
    Qini(k)        = Y_t(k) − Y_c(k) · N_t(k)/N_c(k)   treated responders minus the control rate scaled to the treated count
Normalised Qini = (area under Qini − area under the random line) / (same for the perfect ordering of observed
outcomes): 0 = random, 1 = perfect, negative = worse than random.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import BOOTSTRAP_RESAMPLES, SEED


def _order(score: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Descending by score; ties broken at random so a constant score behaves like random targeting."""
    score = np.asarray(score, dtype=float)
    rng = rng or np.random.default_rng(SEED)
    noise = rng.random(len(score))
    return np.lexsort((noise, -score))


def _sorted(y, t, score):
    """(y, t) in the targeting order of `score`."""
    order = _order(score)
    return y[order], t[order]


def _curve_sorted(y_o, t_o, n_points: int = 100) -> pd.DataFrame:
    """Uplift-gain and Qini curves of an already ordered population on an evenly spaced grid of targeted fractions."""
    n = len(y_o)
    cum_t, cum_c = np.cumsum(t_o), np.cumsum(1 - t_o)
    cum_yt, cum_yc = np.cumsum(y_o * t_o), np.cumsum(y_o * (1 - t_o))
    ks = np.unique(np.clip((np.linspace(0, 1, n_points + 1) * n).astype(int), 0, n))
    ks = ks[ks > 0]
    nt, nc, yt, yc = cum_t[ks - 1], cum_c[ks - 1], cum_yt[ks - 1], cum_yc[ks - 1]
    rate_t = np.divide(yt, nt, out=np.zeros(len(ks)), where=nt > 0)
    rate_c = np.divide(yc, nc, out=np.zeros(len(ks)), where=nc > 0)
    qini = np.where(nc > 0, yt - yc * nt / np.maximum(nc, 1), yt.astype(float))
    return pd.DataFrame({"fraction": np.r_[0.0, ks / n], "n": np.r_[0, ks], "uplift_gain": np.r_[0.0, (rate_t - rate_c) * ks], "qini": np.r_[0.0, qini]})


def curves(y, t, score, n_points: int = 100) -> pd.DataFrame:
    """Uplift-gain and Qini curves on an evenly spaced grid of targeted fractions."""
    y, t = np.asarray(y, dtype=float), np.asarray(t).astype(int)
    y_o, t_o = _sorted(y, t, score)
    return _curve_sorted(y_o, t_o, n_points)


def _area(curve: pd.DataFrame, col: str) -> float:
    return float(np.trapezoid(curve[col].to_numpy(), curve["fraction"].to_numpy()))


def perfect_score(y, t) -> np.ndarray:
    """The ordering that maximises Qini on observed outcomes: treated responders, then non-responders, then control responders."""
    y, t = np.asarray(y).astype(int), np.asarray(t).astype(int)
    return (2 * (t * y) + (1 - y)).astype(float)


def _qini_area_sorted(y_o, t_o) -> float:
    """Area between the Qini curve of an ordering and the random line (chord from (0, 0) to (1, total))."""
    c = _curve_sorted(y_o, t_o)
    return _area(c, "qini") - c["qini"].iloc[-1] / 2


def qini_coefficient(y, t, score, normalize: bool = True) -> float:
    y, t = np.asarray(y, dtype=float), np.asarray(t).astype(int)
    model_area = _qini_area_sorted(*_sorted(y, t, score))
    if not normalize:
        return model_area / len(y)
    perfect_area = _qini_area_sorted(*_sorted(y, t, perfect_score(y, t)))
    return float(model_area / perfect_area) if perfect_area > 0 else 0.0


def auuc(y, t, score) -> float:
    """Area under the uplift-gain curve minus the random line, in incremental responses per customer."""
    c = curves(y, t, score)
    n = len(np.asarray(y))
    return float((_area(c, "uplift_gain") - c["uplift_gain"].iloc[-1] / 2) / n)


def uplift_by_bin(y, t, score, predicted=None, bins: int = 10) -> pd.DataFrame:
    """Observed uplift inside equal-count bins of the score (bin 1 = highest score). The RCT makes each bin's
    treated-minus-control rate an unbiased estimate of the uplift the model placed there."""
    y, t, score = np.asarray(y, dtype=float), np.asarray(t).astype(int), np.asarray(score, dtype=float)
    order = _order(score)
    edges = np.linspace(0, len(y), bins + 1).astype(int)
    rows = []
    for b in range(bins):
        idx = order[edges[b] : edges[b + 1]]
        yt, yc = y[idx][t[idx] == 1], y[idx][t[idx] == 0]
        rt, rc = (yt.mean() if len(yt) else np.nan), (yc.mean() if len(yc) else np.nan)
        se = np.sqrt(rt * (1 - rt) / max(len(yt), 1) + rc * (1 - rc) / max(len(yc), 1)) if len(yt) and len(yc) else np.nan
        rows.append({"bin": b + 1, "n": len(idx), "n_treated": len(yt), "n_control": len(yc), "rate_treated": rt, "rate_control": rc,
                     "uplift_observed": rt - rc, "uplift_ci_low": rt - rc - 1.96 * se, "uplift_ci_high": rt - rc + 1.96 * se,
                     "score_mean": float(score[idx].mean()), "uplift_predicted": float(np.asarray(predicted)[idx].mean()) if predicted is not None else None})
    return pd.DataFrame(rows)


def policy_value(y, t, score, budget: float) -> dict:
    """Target the top `budget` fraction by score. Observed uplift among the targeted (treated minus control rate),
    incremental responses per 1,000 contacts, and the share of the whole population's incremental effect captured."""
    y, t = np.asarray(y, dtype=float), np.asarray(t).astype(int)
    n = len(y)
    k = max(int(round(n * budget)), 1)
    idx = _order(score)[:k]
    yt, yc = y[idx][t[idx] == 1], y[idx][t[idx] == 0]
    if len(yt) == 0 or len(yc) == 0:
        return {"budget": budget, "n_targeted": k, "uplift_targeted": None, "incremental_per_1000": None, "share_of_total": None}
    uplift = yt.mean() - yc.mean()
    all_t, all_c = y[t == 1], y[t == 0]
    total_incremental = (all_t.mean() - all_c.mean()) * n
    se = np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
    return {"budget": budget, "n_targeted": int(k), "uplift_targeted": float(uplift), "uplift_se": float(se), "incremental_per_1000": float(uplift * 1000),
            "incremental_total": float(uplift * k), "share_of_total": float(uplift * k / total_incremental) if total_incremental > 0 else None,
            "random_per_1000": float((all_t.mean() - all_c.mean()) * 1000)}


def bootstrap(fn, y, t, score, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = SEED, **kw) -> dict:
    """Stratified (by treatment) bootstrap of a scalar statistic fn(y, t, score, **kw) → point, 2.5% and 97.5%."""
    y, t, score = np.asarray(y), np.asarray(t).astype(int), np.asarray(score)
    rng = np.random.default_rng(seed)
    idx_t, idx_c = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    point = fn(y, t, score, **kw)
    vals = []
    for _ in range(n_resamples):
        idx = np.concatenate([rng.choice(idx_t, len(idx_t)), rng.choice(idx_c, len(idx_c))])
        vals.append(fn(y[idx], t[idx], score[idx], **kw))
    vals = np.asarray(vals, dtype=float)
    return {"point": float(point), "ci_low": float(np.nanpercentile(vals, 2.5)), "ci_high": float(np.nanpercentile(vals, 97.5)), "n_resamples": n_resamples}


def _policy_per_1000(y, t, score, budget):
    v = policy_value(y, t, score, budget)["incremental_per_1000"]
    return np.nan if v is None else v


def _policy_sorted(y_o, t_o, k: int) -> float:
    """Incremental responses per 1,000 contacts among the top k of an ordering; nan when an arm is empty."""
    yt, yc = y_o[:k][t_o[:k] == 1], y_o[:k][t_o[:k] == 0]
    return float((yt.mean() - yc.mean()) * 1000) if len(yt) and len(yc) else np.nan


def bootstrap_block(y, t, score, budgets=(0.1, 0.2, 0.3), n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = SEED) -> dict:
    """Normalised Qini and incremental per 1,000 at every budget from ONE stratified (by treatment) resample loop: each
    resample is sorted once by the score and once by the perfect ordering instead of once per statistic. Same resamples
    and same numbers as calling `bootstrap` per statistic, several times faster."""
    y, t, score = np.asarray(y, dtype=float), np.asarray(t).astype(int), np.asarray(score, dtype=float)
    n = len(y)
    ks = [max(int(round(n * b)), 1) for b in budgets]

    def stats(y_, t_, s_):
        y_o, t_o = _sorted(y_, t_, s_)
        perfect_area = _qini_area_sorted(*_sorted(y_, t_, perfect_score(y_, t_)))
        q = _qini_area_sorted(y_o, t_o) / perfect_area if perfect_area > 0 else 0.0
        return float(q), [_policy_sorted(y_o, t_o, k) for k in ks]

    rng = np.random.default_rng(seed)
    idx_t, idx_c = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    q0, p0 = stats(y, t, score)
    qs, ps = [], []
    for _ in range(n_resamples):
        idx = np.concatenate([rng.choice(idx_t, len(idx_t)), rng.choice(idx_c, len(idx_c))])
        q, p = stats(y[idx], t[idx], score[idx])
        qs.append(q)
        ps.append(p)
    qs, ps = np.asarray(qs, dtype=float), np.asarray(ps, dtype=float)
    out = {"qini": {"point": q0, "ci_low": float(np.nanpercentile(qs, 2.5)), "ci_high": float(np.nanpercentile(qs, 97.5)), "n_resamples": n_resamples},
           "policy": {}}
    for j, b in enumerate(budgets):
        out["policy"][str(b)] = {"point": p0[j], "ci_low": float(np.nanpercentile(ps[:, j], 2.5)), "ci_high": float(np.nanpercentile(ps[:, j], 97.5)),
                                 "n_resamples": n_resamples}
    return out


def evaluate_scores(y, t, score, budgets=(0.1, 0.2, 0.3), n_resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """The standard block for one model on one split: Qini (normalised, with CI), AUUC, policy value at a few budgets with
    CIs and the share of the total incremental effect captured."""
    out = bootstrap_block(y, t, score, budgets, n_resamples)
    out["auuc"] = auuc(y, t, score)
    for b in budgets:
        out["policy"][str(b)]["share_of_total"] = policy_value(y, t, score, b)["share_of_total"]
    return out


def pehe(tau_true, tau_pred) -> float:
    """Root mean squared error of individual effects (needs the truth: synthetic data only)."""
    return float(np.sqrt(np.mean((np.asarray(tau_true, dtype=float) - np.asarray(tau_pred, dtype=float)) ** 2)))


def spearman(a, b) -> float:
    return float(spearmanr(np.asarray(a, dtype=float), np.asarray(b, dtype=float)).statistic)
