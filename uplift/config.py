"""Every knob: paths, split shares, seeds, the economics assumptions. No environment variables, no secrets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
SPLITS = DATA / "splits"
MODELS = ROOT / "models"
RESULTS = ROOT / "eval" / "results"

RAW_CANDIDATES = ["criteo-uplift-v2.1.csv", "criteo-research-uplift-v2.1.csv.gz", "criteo-research-uplift-v2.1.csv"]
KAGGLE_DATASET = "arashnic/uplift-modeling"

SEED = 7
FEATURES = [f"f{i}" for i in range(12)]
TREATMENT = "treatment"
OUTCOMES = ("visit", "conversion")
PRIMARY_OUTCOME = "visit"


@dataclass(frozen=True)
class SplitRules:
    train: float = 0.6
    dev: float = 0.2
    test: float = 0.2
    subsample_rows: int = 4_000_000  # default working set; --full uses every row


@dataclass(frozen=True)
class Economics:
    """Stated assumptions for the profit line; the README names them as such."""

    value_per_visit: float = 2.0  # what one incremental visit is worth, in currency units
    value_per_conversion: float = 60.0
    contact_cost: float = 0.10  # cost of one contact (email/SMS/ad impression)


SPLIT_RULES = SplitRules()
ECONOMICS = Economics()
BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
BOOTSTRAP_RESAMPLES = 200
CROSS_FIT_FOLDS = 5
