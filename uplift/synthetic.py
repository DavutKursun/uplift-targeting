"""A fake campaign with KNOWN individual treatment effects, so learners can be scored against the truth.

The world has the four textbook segments: persuadables (τ > 0), sure things (high base rate, τ ≈ 0), lost
causes (low base rate, τ ≈ 0) and sleeping dogs (τ < 0, the offer annoys them). A response model that targets
high P(Y | T=1) picks the sure things; an uplift model should pick the persuadables. Treatment is randomized
with the Criteo share (85%) by default. Columns mimic the real file: f0..f11, treatment, visit, conversion,
exposure, plus the hidden truth `tau_true` and `p0_true` that only tests and reports may read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES


def make_campaign(n: int = 100_000, seed: int = 7, treated_share: float = 0.85) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 12))
    # a few "categorical as float" columns like the real file
    X[:, 4] = rng.integers(0, 40, n).astype(float)
    X[:, 5] = rng.integers(0, 7, n).astype(float)
    X[:, 9] = rng.choice([0.0, 1.0, 2.0, 5.0, 10.0], n)
    f0, f1, f2, f3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    # base visit probability without treatment: sure things live at high f1, lost causes at low f1
    logit0 = -3.6 + 1.6 * f1 + 0.3 * f3 - 0.2 * (X[:, 5] == 3)
    p0 = 1 / (1 + np.exp(-logit0))
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    # true effect: persuadables where f0 is high; sure things (high f1) barely move; sleeping dogs where f2 > 1.2
    tau = (0.012 + 0.045 * sigmoid(2.0 * f0)) * (1 - sigmoid(2.5 * (f1 - 0.8))) - 0.04 * (f2 > 1.2)
    p1 = np.clip(p0 + tau, 0.001, 0.999)
    tau = p1 - p0  # the effect after clipping is the truth

    t = (rng.random(n) < treated_share).astype(int)
    p = np.where(t == 1, p1, p0)
    visit = (rng.random(n) < p).astype(int)
    conversion = (visit == 1) & (rng.random(n) < 0.06 + 0.02 * t)
    exposure = (t == 1) & (rng.random(n) < 0.9)
    df = pd.DataFrame(X.astype(np.float32), columns=FEATURES)
    df["treatment"] = t.astype(np.int8)
    df["visit"] = visit.astype(np.int8)
    df["conversion"] = conversion.astype(np.int8)
    df["exposure"] = exposure.astype(np.int8)
    df["tau_true"] = tau.astype(np.float32)
    df["p0_true"] = p0.astype(np.float32)
    return df


def write_raw_csv(df: pd.DataFrame, path, gz: bool = True):
    """The real file's column order, without the hidden truth columns."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = FEATURES + ["treatment", "conversion", "visit", "exposure"]
    df[cols].to_csv(path, index=False, compression="gzip" if gz else None)
    return path
