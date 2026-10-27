from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.data import assign_splits, xty
from uplift.learners import LEARNERS, fast_settings, fit_learner
from uplift.synthetic import make_campaign


@pytest.fixture(scope="session")
def world():
    df = make_campaign(n=90_000, seed=5)
    df["split"] = assign_splits(df)
    tr, dv = df[df["split"] == "train"], df[df["split"] == "dev"]
    X, t, y = xty(tr, "visit")
    Xd, td, yd = xty(dv, "visit")
    return {"df": df, "train": tr, "dev": dv, "X": X, "t": t, "y": y, "Xd": Xd, "td": td, "yd": yd, "truth": dv["tau_true"].to_numpy(), "p0": dv["p0_true"].to_numpy()}


@pytest.fixture(scope="session")
def fitted(world):
    return {name: fit_learner(name, world["X"], world["t"], world["y"], settings=fast_settings()) for name in LEARNERS}
