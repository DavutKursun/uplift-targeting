"""Uplift learners behind one interface: fit(X, t, y) → predict_uplift(X), predict_outcomes(X) → (p0, p1).

All base models are LightGBM boosters (binary objective for outcome models, L2 regression for τ models).
The propensity e is known for randomized data (the treated share) and passed in; `estimate_propensity=True`
fits it instead (an ablation). Cross-fitting (R- and DR-learners) uses K folds so that no row's pseudo-outcome
is built from a nuisance model that saw that row.

    response            P(Y=1 | X, T=1): the wrong baseline, ranks sure things first
    s_learner           f(X, T) → τ = f(X,1) − f(X,0)
    t_learner           μ1 on treated, μ0 on control → τ = μ1 − μ0
    x_learner           imputed effects per arm, τ = e·τ0 + (1−e)·τ1               (Künzel et al. 2019)
    transformed_outcome regress Z = Y·(T−e)/(e(1−e)) on X                           (Athey & Imbens 2016)
    r_learner           residualize Y and T, weighted regression for τ, cross-fitted (Nie & Wager 2021)
    dr_learner          doubly-robust pseudo-outcomes, cross-fitted, regress on X    (Kennedy 2020)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .config import CROSS_FIT_FOLDS, SEED

CLS_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=0)
REG_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=1000, feature_fraction=0.9, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=0)
ROUNDS = 400


def _train(X, y, params, rounds, weight=None):
    import lightgbm as lgb

    d = lgb.Dataset(np.asarray(X, dtype=np.float32), label=np.asarray(y, dtype=float), weight=None if weight is None else np.asarray(weight, dtype=float),
                    free_raw_data=False)
    return lgb.train(params, d, num_boost_round=rounds)


def _predict(booster, X):
    return np.asarray(booster.predict(np.asarray(X, dtype=np.float32)), dtype=float)


@dataclass
class Settings:
    cls_params: dict = field(default_factory=lambda: dict(CLS_PARAMS))
    reg_params: dict = field(default_factory=lambda: dict(REG_PARAMS))
    rounds: int = ROUNDS
    folds: int = CROSS_FIT_FOLDS
    estimate_propensity: bool = False


class UpliftLearner:
    name = "base"
    uses_propensity = False

    def __init__(self, propensity: float | None = None, settings: Settings | None = None) -> None:
        self.e = propensity
        self.s = settings or Settings()
        self._p0 = None  # auxiliary control-outcome model for segments when the learner has none
        self.fitted = False

    # --- interface ------------------------------------------------------------------
    def fit(self, X, t, y) -> UpliftLearner:
        X, t, y = np.asarray(X, dtype=np.float32), np.asarray(t).astype(int), np.asarray(y).astype(int)
        if self.e is None:
            self.e = float(t.mean())
        self._fit(X, t, y)
        if self._p0 is None and not hasattr(self, "mu0"):
            self._p0 = _train(X[t == 0], y[t == 0], self.s.cls_params, self.s.rounds)
        self.fitted = True
        return self

    def predict_uplift(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict_outcomes(self, X) -> tuple[np.ndarray, np.ndarray]:
        """(P(Y|T=0), P(Y|T=1)) — p1 = p0 + τ clipped to [0, 1] unless the learner has its own μ1."""
        X = np.asarray(X, dtype=np.float32)
        p0 = _predict(self.mu0, X) if hasattr(self, "mu0") else _predict(self._p0, X)
        p1 = np.clip(p0 + self.predict_uplift(X), 0, 1)
        return p0, p1

    def _fit(self, X, t, y) -> None:
        raise NotImplementedError


class ResponseModel(UpliftLearner):
    """Not an uplift model: scores P(Y=1 | X, T=1). Kept as the baseline every uplift claim must beat."""

    name = "response"

    def _fit(self, X, t, y):
        self.mu1 = _train(X[t == 1], y[t == 1], self.s.cls_params, self.s.rounds)

    def predict_uplift(self, X):
        return _predict(self.mu1, X)  # used as a ranking score only

    def predict_outcomes(self, X):
        X = np.asarray(X, dtype=np.float32)
        return _predict(self._p0, X), _predict(self.mu1, X)


class SLearner(UpliftLearner):
    name = "s_learner"

    def _fit(self, X, t, y):
        self.f = _train(np.column_stack([X, t]), y, self.s.cls_params, self.s.rounds)

    def _with(self, X, value):
        return np.column_stack([np.asarray(X, dtype=np.float32), np.full(len(X), value, dtype=np.float32)])

    def predict_uplift(self, X):
        return _predict(self.f, self._with(X, 1)) - _predict(self.f, self._with(X, 0))

    def predict_outcomes(self, X):
        p0, p1 = _predict(self.f, self._with(X, 0)), _predict(self.f, self._with(X, 1))
        return p0, p1


class TLearner(UpliftLearner):
    name = "t_learner"

    def _fit(self, X, t, y):
        self.mu1 = _train(X[t == 1], y[t == 1], self.s.cls_params, self.s.rounds)
        self.mu0 = _train(X[t == 0], y[t == 0], self.s.cls_params, self.s.rounds)

    def predict_uplift(self, X):
        return _predict(self.mu1, X) - _predict(self.mu0, X)

    def predict_outcomes(self, X):
        return _predict(self.mu0, X), _predict(self.mu1, X)


class XLearner(TLearner):
    """Stage 2 imputes each unit's missing potential outcome with the other arm's model, fits τ per arm, and
    weights the two τ models by the propensity: with 85% treated, τ0 (fit on control rows with the well-estimated
    μ1) gets weight e."""

    name = "x_learner"
    uses_propensity = True

    def _fit(self, X, t, y):
        super()._fit(X, t, y)
        d1 = y[t == 1] - _predict(self.mu0, X[t == 1])
        d0 = _predict(self.mu1, X[t == 0]) - y[t == 0]
        self.tau1 = _train(X[t == 1], d1, self.s.reg_params, self.s.rounds)
        self.tau0 = _train(X[t == 0], d0, self.s.reg_params, self.s.rounds)

    def predict_uplift(self, X):
        return self.e * _predict(self.tau0, X) + (1 - self.e) * _predict(self.tau1, X)

    def predict_outcomes(self, X):
        X = np.asarray(X, dtype=np.float32)
        p0 = _predict(self.mu0, X)
        return p0, np.clip(p0 + self.predict_uplift(X), 0, 1)


class TransformedOutcome(UpliftLearner):
    """Z = Y·(T − e) / (e·(1 − e)) has E[Z | X] = τ(X) under randomization; one regression, high variance."""

    name = "transformed_outcome"
    uses_propensity = True

    def _fit(self, X, t, y):
        e = self.e
        z = y * (t - e) / (e * (1 - e))
        self.tau = _train(X, z, self.s.reg_params, self.s.rounds)

    def predict_uplift(self, X):
        return _predict(self.tau, X)


def _cross_fit_outcome(X, t, y, settings: Settings, per_arm: bool):
    """Out-of-fold predictions. per_arm=False → m(x) = E[Y|X]; per_arm=True → (μ0(x), μ1(x)) each from the other folds."""
    skf = StratifiedKFold(n_splits=settings.folds, shuffle=True, random_state=SEED)
    strata = t * 2 + y
    m = np.zeros(len(y))
    mu0, mu1 = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in skf.split(X, strata):
        if per_arm:
            b0 = _train(X[tr][t[tr] == 0], y[tr][t[tr] == 0], settings.cls_params, settings.rounds)
            b1 = _train(X[tr][t[tr] == 1], y[tr][t[tr] == 1], settings.cls_params, settings.rounds)
            mu0[te], mu1[te] = _predict(b0, X[te]), _predict(b1, X[te])
        else:
            b = _train(X[tr], y[tr], settings.cls_params, settings.rounds)
            m[te] = _predict(b, X[te])
    return (mu0, mu1) if per_arm else m


def _propensity(X, t, settings: Settings, e_known: float, estimate: bool) -> np.ndarray:
    if not estimate:
        return np.full(len(t), e_known)
    skf = StratifiedKFold(n_splits=settings.folds, shuffle=True, random_state=SEED)
    out = np.zeros(len(t))
    for tr, te in skf.split(X, t):
        out[te] = _predict(_train(X[tr], t[tr], settings.cls_params, settings.rounds), X[te])
    return np.clip(out, 0.02, 0.98)


class RLearner(UpliftLearner):
    """Residualize: Ỹ = Y − m̂(X), T̃ = T − ê(X); then τ = argmin Σ T̃²·(Ỹ/T̃ − τ(X))², i.e. a regression of Ỹ/T̃ on X
    with weights T̃². Nuisances are cross-fitted."""

    name = "r_learner"
    uses_propensity = True

    def _fit(self, X, t, y):
        m = _cross_fit_outcome(X, t, y, self.s, per_arm=False)
        e = _propensity(X, t, self.s, self.e, self.s.estimate_propensity)
        y_res, t_res = y - m, t - e
        self.tau = _train(X, y_res / t_res, self.s.reg_params, self.s.rounds, weight=t_res**2)

    def predict_uplift(self, X):
        return _predict(self.tau, X)


class DRLearner(UpliftLearner):
    """Pseudo-outcome φ = μ1 − μ0 + T(Y − μ1)/e − (1 − T)(Y − μ0)/(1 − e), nuisances cross-fitted; regress φ on X.
    Unbiased if either the outcome models or the propensity is right."""

    name = "dr_learner"
    uses_propensity = True

    def _fit(self, X, t, y):
        mu0, mu1 = _cross_fit_outcome(X, t, y, self.s, per_arm=True)
        e = _propensity(X, t, self.s, self.e, self.s.estimate_propensity)
        phi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
        self.tau = _train(X, phi, self.s.reg_params, self.s.rounds)
        self.mu0 = _train(X[t == 0], y[t == 0], self.s.cls_params, self.s.rounds)  # full-data μ0 for segments

    def predict_uplift(self, X):
        return _predict(self.tau, X)


LEARNERS: dict[str, type[UpliftLearner]] = {c.name: c for c in (ResponseModel, SLearner, TLearner, XLearner, TransformedOutcome, RLearner, DRLearner)}
UPLIFT_LEARNERS = [n for n in LEARNERS if n != "response"]


def fit_learner(name: str, X, t, y, propensity: float | None = None, settings: Settings | None = None) -> UpliftLearner:
    if name not in LEARNERS:
        raise ValueError(f"unknown learner {name!r}; choose from {list(LEARNERS)}")
    return LEARNERS[name](propensity, settings).fit(X, t, y)


def fast_settings(rounds: int = 120) -> Settings:
    """Small-data settings for tests and smoke runs."""
    cls = {**CLS_PARAMS, "learning_rate": 0.1, "min_data_in_leaf": 50, "num_leaves": 15}
    reg = {**REG_PARAMS, "learning_rate": 0.1, "min_data_in_leaf": 100, "num_leaves": 7}
    return Settings(cls, reg, rounds, folds=3)
