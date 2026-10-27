"""Two figures per split, from the saved models and the evaluation outputs (never the test split before M4):

  budget_curve_<split>.png     incremental responses per 1,000 contacts vs budget, raw and IPW-weighted panels, four models
  profit_curve_<split>.png     cumulative profit per 1,000 customers vs budget under the stated economics
  qini_curves_<split>.png      Qini curves (incremental visits vs fraction targeted) for every model: the best uplift learner
                               and the response model in colour, the other learners in grey, random as a dashed chord
  deciles_<split>_<best>.png   observed uplift per decile of the best learner's score (bin 1 = highest) with 95% CI bars,
                               the predicted uplift per decile, and the ATE as the reference

Usage: python scripts/plots.py --split dev [--outcome visit] [--best r_learner] [--models-dir models] [--out eval/results]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.config import ECONOMICS, MODELS, PRIMARY_OUTCOME, RESULTS, SPLITS  # noqa: E402
from uplift.data import ate, load_split, xty  # noqa: E402
from uplift.metrics import curves  # noqa: E402

# reference palette (dataviz skill): categorical slots 1–2, chrome in muted inks
BLUE, ORANGE, AQUA, GREY, MUTED, GRID, INK = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7", "#898781", "#e1e0d9", "#0b0b0b"
MODEL_COLOUR = {"s_learner": BLUE, "response": ORANGE, "r_learner": AQUA, "random": MUTED}
LABEL = {"random": "random targeting", "response": "response model", "s_learner": "S-learner", "t_learner": "T-learner", "x_learner": "X-learner",
         "transformed_outcome": "transformed outcome", "r_learner": "R-learner", "dr_learner": "DR-learner"}


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def qini_plot(y, t, scores: dict, best: str, path: Path, split: str, outcome: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150, facecolor="#fcfcfb")
    _style(ax)
    n = len(y)
    per_1000 = 1000 / n  # Qini is in incremental responses; show it per 1,000 customers of the split
    total = None
    for name, s in scores.items():
        c = curves(y, t, s)
        total = c["qini"].iloc[-1] * per_1000
        emphasised = name in (best, "response")
        colour = BLUE if name == best else ORANGE if name == "response" else GREY
        ax.plot(c["fraction"], c["qini"] * per_1000, color=colour, linewidth=1.9 if emphasised else 1.0, zorder=3 if emphasised else 2,
                label=LABEL[name] + (" (best uplift learner)" if name == best else ""))
    ax.plot([0, 1], [0, total], color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), label="random targeting", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of customers targeted, by model score", color=MUTED, fontsize=10)
    ax.set_ylabel(f"Qini: incremental {outcome}s among the treated, per 1,000 customers", color=MUTED, fontsize=10)
    ax.set_title(f"Qini curves on {split} ({n:,} customers): best uplift learner vs response model", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def deciles_plot(tab: pd.DataFrame, base_ate: float, name: str, path: Path, split: str, outcome: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150, facecolor="#fcfcfb")
    _style(ax)
    x = tab["bin"].to_numpy()
    obs, lo, hi = tab["uplift_observed"] * 100, tab["uplift_ci_low"] * 100, tab["uplift_ci_high"] * 100
    ax.axhline(0, color=GREY, linewidth=0.8, zorder=1)
    ax.axhline(base_ate * 100, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(f"ATE {base_ate * 100:.2f} pp", xy=(10.45, base_ate * 100), color=MUTED, fontsize=9, va="bottom", ha="right")
    ax.errorbar(x, obs, yerr=[obs - lo, hi - obs], fmt="o", color=BLUE, ecolor=BLUE, elinewidth=1.2, capsize=3, markersize=6, zorder=3,
                label="observed uplift in the decile (treated − control rate, 95% CI)")
    if tab["uplift_predicted"].notna().all():
        ax.plot(x, tab["uplift_predicted"] * 100, marker="D", markersize=5, markerfacecolor="white", markeredgecolor=ORANGE, color=ORANGE, linewidth=1.0,
                zorder=2, label="mean predicted uplift in the decile")
    ax.set_xticks(x)
    ax.set_xlabel("decile of the model score (1 = highest predicted uplift)", color=MUTED, fontsize=10)
    ax.set_ylabel(f"uplift in {outcome} rate, percentage points", color=MUTED, fontsize=10)
    ax.set_title(f"Observed vs predicted uplift by decile — {LABEL[name]} on {split}", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _two_panel(table: pd.DataFrame, cols, bands, titles, ylabel, suptitle, path: Path, hline: float | None, hline_label: str | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150, facecolor="#fcfcfb", sharey=True)
    for ax, col, (lo, hi), title in zip(axes, cols, bands, titles):
        _style(ax)
        for name, g in table.groupby("model", sort=False):
            g = g.sort_values("budget")
            colour = MODEL_COLOUR.get(name, GREY)
            dashed = {"linestyle": (0, (4, 3))} if name == "random" else {}
            if lo is not None and g[lo].notna().all():
                ax.fill_between(g["budget"], g[lo], g[hi], color=colour, alpha=0.10, linewidth=0)
            ax.plot(g["budget"], g[col], color=colour, linewidth=1.8, marker="o", markersize=3.5, label=LABEL.get(name, name), zorder=3, **dashed)
        if hline is not None:
            ax.axhline(hline, color=MUTED if hline else GREY, linewidth=0.9, linestyle=(0, (1, 2)) if hline else "-", zorder=1)
            if hline_label:
                ax.annotate(hline_label, xy=(0.5, hline), color=MUTED, fontsize=8.5, va="bottom", ha="right")
        ax.set_xlabel("budget: fraction of customers contacted", color=MUTED, fontsize=10)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlim(0.03, 0.52)
    axes[0].set_ylabel(ylabel, color=MUTED, fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(suptitle, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def budget_plot(table: pd.DataFrame, path: Path, split: str, outcome: str, threshold_per_1000: float) -> None:
    _two_panel(table, ("incremental_per_1000", "ipw_per_1000"), (("ci_low", "ci_high"), ("ipw_ci_low", "ipw_ci_high")),
               ("raw: treated − control rate inside the targeted set", "IPW-weighted with a cross-fitted propensity"),
               f"incremental {outcome}s per 1,000 contacts", f"Budget curves on {split}: incremental {outcome}s per 1,000 contacts (bands: 95% CI)", path,
               threshold_per_1000, f"marginal break-even: {threshold_per_1000:.0f} per 1,000 contacts")


def profit_plot(table: pd.DataFrame, path: Path, split: str, outcome: str, value: float, cost: float) -> None:
    _two_panel(table, ("profit_per_1000_customers", "ipw_profit_per_1000_customers"), ((None, None), (None, None)),
               ("raw", "IPW-weighted"), "cumulative profit per 1,000 customers of the population",
               f"Profit on {split} under the stated economics: {value:.2f} per incremental {outcome}, {cost:.2f} per contact (assumptions)", path, 0.0, None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--outcome", default=PRIMARY_OUTCOME, choices=["visit", "conversion"])
    ap.add_argument("--best", default=None, help="learner for the decile plot (default: top uplift learner by Qini in the split summary)")
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    if a.split == "test" and not a.final:
        print("refusing: test needs --final")
        return 3
    summary_path = a.out / f"{a.split}-summary-{a.outcome}.json"
    best = a.best
    if best is None:
        if not summary_path.exists():
            print(f"{summary_path} not found — run scripts/evaluate.py or pass --best")
            return 2
        metrics = json.loads(summary_path.read_text())["metrics"]
        best = next(m["model"] for m in sorted(metrics, key=lambda m: -m["qini"]) if m["model"] != "response")
    df = load_split(a.split, a.splits_dir)
    X, t, y = xty(df, a.outcome)
    scores = {}
    for p in sorted(a.models_dir.glob(f"*-{a.outcome}.joblib")):
        b = joblib.load(p)
        scores[b["name"]] = b["learner"].predict_uplift(X)
    if best not in scores:
        print(f"no bundle for {best} in {a.models_dir}")
        return 2
    a.out.mkdir(parents=True, exist_ok=True)
    qini_plot(y, t, scores, best, a.out / f"qini_curves_{a.split}.png", a.split, a.outcome)
    dec_path = a.out / f"deciles_{a.split}_{a.outcome}_{best}.csv"
    if not dec_path.exists():
        print(f"{dec_path} not found — run scripts/evaluate.py first")
        return 2
    deciles_plot(pd.read_csv(dec_path), ate(y, t)["ate"], best, a.out / f"deciles_{a.split}_{best}.png", a.split, a.outcome)
    written = [a.out / f"qini_curves_{a.split}.png", a.out / f"deciles_{a.split}_{best}.png"]
    policy_path = a.out / f"policy_{a.split}_{a.outcome}.csv"
    if policy_path.exists():
        table = pd.read_csv(policy_path)
        value = ECONOMICS.value_per_visit if a.outcome == "visit" else ECONOMICS.value_per_conversion
        budget_plot(table, a.out / f"budget_curve_{a.split}.png", a.split, a.outcome, ECONOMICS.contact_cost / value * 1000)
        profit_plot(table, a.out / f"profit_curve_{a.split}.png", a.split, a.outcome, value, ECONOMICS.contact_cost)
        written += [a.out / f"budget_curve_{a.split}.png", a.out / f"profit_curve_{a.split}.png"]
    print("wrote " + ", ".join(str(w) for w in written) + f" (best = {best})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
