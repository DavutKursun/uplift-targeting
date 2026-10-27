"""Dev → test side by side (M4): every block of the test run next to its dev counterpart, from the result JSONs, so no
number is transcribed by hand. Writes eval/results/dev_vs_test.md and prints it.

Usage: python scripts/dev_test_table.py [--results eval/results]
Blocks: Qini with CI for every learner (visit, conversion); paired differences; per-1,000 at 5/10/20/30% raw and IPW
for random, response, S, R; profit-maximising and break-even budgets; segment shares and observed uplift; placebo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.config import RESULTS

LABEL = {"random": "random", "response": "response model", "s_learner": "S-learner", "t_learner": "T-learner", "x_learner": "X-learner",
         "transformed_outcome": "transformed outcome", "r_learner": "R-learner", "dr_learner": "DR-learner"}


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _ci(v, d=4):
    return f"{v['point']:.{d}f} [{v['ci_low']:.{d}f}, {v['ci_high']:.{d}f}]"


def _ci2(point, lo, hi, d=1):
    return f"{point:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def qini_block(res: Path, outcome: str, lines: list[str]) -> None:
    dev, test = _load(res / f"dev-summary-{outcome}.json"), _load(res / f"test-summary-{outcome}.json")
    if not dev or not test:
        return
    lines += [f"### Qini and per-1,000 — `{outcome}` (dev {dev['n']:,} rows → test {test['n']:,} rows, {test['n_resamples']} resamples)", "",
              f"ATE: dev {dev['ate']['ate'] * 100:.3f} pp [{dev['ate']['ci_low'] * 100:.3f}, {dev['ate']['ci_high'] * 100:.3f}] → "
              f"test {test['ate']['ate'] * 100:.3f} pp [{test['ate']['ci_low'] * 100:.3f}, {test['ate']['ci_high'] * 100:.3f}]; "
              f"random = {dev['random_per_1000']} → {test['random_per_1000']} per 1,000.", "",
              "| Learner | Dev Qini [CI] | Test Qini [CI] | Dev / 1,000 at 10% | Test at 10% | Dev at 20% | Test at 20% | Test at 30% |", "|---|---|---|---|---|---|---|---|"]
    dm, tm = {m["model"]: m for m in dev["metrics"]}, {m["model"]: m for m in test["metrics"]}
    for name in sorted(tm, key=lambda k: -tm[k]["qini"]):
        d, t = dm.get(name), tm[name]
        dq = f"{d['qini']:.4f} [{d['qini_ci_low']:.4f}, {d['qini_ci_high']:.4f}]" if d else "—"
        lines.append(f"| {LABEL.get(name, name)} | {dq} | {t['qini']:.4f} [{t['qini_ci_low']:.4f}, {t['qini_ci_high']:.4f}] | "
                     f"{d['per_1000_at_10pct']:.1f} {d['per_1000_at_10pct_ci']} | {t['per_1000_at_10pct']:.1f} {t['per_1000_at_10pct_ci']} | "
                     f"{d['per_1000_at_20pct']:.1f} {d['per_1000_at_20pct_ci']} | {t['per_1000_at_20pct']:.1f} {t['per_1000_at_20pct_ci']} | {t['per_1000_at_30pct']:.1f} {t['per_1000_at_30pct_ci']} |")
    lines.append("")


def paired_block(res: Path, outcome: str, lines: list[str]) -> None:
    dev, test = _load(res / f"paired_dev_{outcome}.json"), _load(res / f"paired_test_{outcome}.json")
    if not dev or not test:
        return
    lines += [f"### Paired differences — `{outcome}`", "", "| Difference | Statistic | Dev [CI] | Test [CI] | Test excludes 0 |", "|---|---|---|---|---|"]
    for pair, d in test["pairs"].items():
        a, b = pair.split("-")
        for key, v in d.items():
            stat = "Qini" if key == "qini" else f"per 1,000 at {int(round(float(key) * 100))}%"
            dv = dev["pairs"].get(pair, {}).get(key)
            digits = 4 if key == "qini" else 1
            lines.append(f"| {LABEL[a]} − {LABEL[b]} | {stat} | {_ci(dv, digits) if dv else '—'} | {_ci(v, digits)} | {'yes' if v['excludes_zero'] else 'no'} |")
    lines.append("")


def policy_block(res: Path, lines: list[str]) -> None:
    dev, test = _load(res / "policy_dev_visit.json"), _load(res / "policy_test_visit.json")
    if not dev or not test:
        return
    import pandas as pd

    dt, tt = pd.read_csv(res / "policy_dev_visit.csv"), pd.read_csv(res / "policy_test_visit.csv")
    lines += ["### Budget curves — incremental visits per 1,000 contacts, raw / IPW (dev → test)", "",
              f"Propensity for IPW, cross-fitted on each split: dev AUC {dev['propensity']['auc_oof']} → test AUC {test['propensity']['auc_oof']}.", "",
              "| Model | Budget | Dev raw [CI] | Test raw [CI] | Dev IPW | Test IPW |", "|---|---|---|---|---|---|"]
    for name in ("random", "response", "s_learner", "r_learner"):
        for b in (0.05, 0.1, 0.2, 0.3):
            d = dt[(dt["model"] == name) & (dt["budget"].round(2) == b)].iloc[0]
            t = tt[(tt["model"] == name) & (tt["budget"].round(2) == b)].iloc[0]
            lines.append(f"| {LABEL[name]} | {int(b * 100)}% | {_ci2(d['incremental_per_1000'], d['ci_low'], d['ci_high'])} | {_ci2(t['incremental_per_1000'], t['ci_low'], t['ci_high'])} | "
                         f"{d['ipw_per_1000']:.1f} | {t['ipw_per_1000']:.1f} |")
    lines += ["", "### Profit under the stated economics (dev → test)", "",
              f"Value per visit {test['economics']['value_per_response']}, contact cost {test['economics']['contact_cost']} (assumptions); marginal break-even "
              f"{test['economics']['marginal_threshold_per_1000']:.0f} per 1,000 contacts.", "",
              "| Model | Best budget raw (IPW), dev → test | Max profit per 1,000 customers, dev → test | Break-even budget raw (IPW), dev → test |", "|---|---|---|---|"]
    for name in ("random", "response", "s_learner", "r_learner"):
        d, t = dev["breakeven"][name], test["breakeven"][name]
        be = lambda x: "—" if x is None else f"{x * 100:.0f}%"
        lines.append(f"| {LABEL[name]} | {d['raw']['profit_max_budget'] * 100:.0f}% ({d['ipw']['profit_max_budget'] * 100:.0f}%) → "
                     f"{t['raw']['profit_max_budget'] * 100:.0f}% ({t['ipw']['profit_max_budget'] * 100:.0f}%) | "
                     f"{d['raw']['max_profit_per_1000_customers']:.1f} → {t['raw']['max_profit_per_1000_customers']:.1f} | "
                     f"{be(d['raw']['breakeven_budget'])} ({be(d['ipw']['breakeven_budget'])}) → {be(t['raw']['breakeven_budget'])} ({be(t['ipw']['breakeven_budget'])}) |")
    lines += ["", "### Segments with the S-learner (dev → test; δ and p0_high fixed at the dev values)", "",
              "| Segment | Dev share | Test share | Dev observed uplift [CI] | Test observed uplift [CI] |", "|---|---|---|---|---|"]
    ds, ts = dev["segments"], test["segments"]
    for k in ("persuadable", "sure_thing", "lost_cause", "sleeping_dog"):
        d, t = ds["observed"][k], ts["observed"][k]
        f = lambda o: f"{o['ate'] * 100:+.2f} pp [{o['ci_low'] * 100:.2f}, {o['ci_high'] * 100:.2f}]" if "ate" in o else "too few rows"
        lines.append(f"| {k} | {d['share'] * 100:.1f}% | {t['share'] * 100:.1f}% | {f(d)} | {f(t)} |")
    d, t = ds["tau_negative"], ts["tau_negative"]
    lines.append(f"| τ̂ < 0 (any size) | {d['share'] * 100:.1f}% | {t['share'] * 100:.1f}% | {d['ate'] * 100:+.2f} pp [{d['ci_low'] * 100:.2f}, {d['ci_high'] * 100:.2f}] | "
                 f"{t['ate'] * 100:+.2f} pp [{t['ci_low'] * 100:.2f}, {t['ci_high'] * 100:.2f}] |")
    lines.append("")


def placebo_block(res: Path, lines: list[str]) -> None:
    rows = []
    for split in ("dev", "test"):
        for learner in ("s_learner", "r_learner"):
            d = _load(res / f"placebo_{split}_visit_{learner}.json")
            if d:
                q = d["qini_placebo"]
                rows.append(f"| {LABEL[learner]} | {split} | {q['point']:+.4f} [{q['ci_low']:+.4f}, {q['ci_high']:+.4f}] | {'yes' if d['contains_zero'] else 'NO'} |")
    if rows:
        lines += ["### Placebo (treatment shuffled in train, deployed settings)", "", "| Learner | Scored on | Placebo Qini [CI] | Contains 0 |", "|---|---|---|---|", *rows, ""]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    test = _load(a.results / "test-summary-visit.json")
    lines = [f"# Dev → test, side by side (test run {test['run_date'] if test else '—'})", ""]
    for outcome in ("visit", "conversion"):
        qini_block(a.results, outcome, lines)
        paired_block(a.results, outcome, lines)
    policy_block(a.results, lines)
    placebo_block(a.results, lines)
    (a.results / "dev_vs_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
