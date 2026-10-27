"""One table for the M2 ablations: base vs estimated propensity vs categorical columns, for the learners that ran in every
variant — normalised Qini and incremental visits per 1,000 contacts at a 20% budget, each with its bootstrap CI — and the
decision, by a rule fixed before the numbers were seen: keep the variant whose Qini CI lower bound is higher; ties go to
the simpler setting (base < estimated propensity < categorical).

Usage: python scripts/ablation_table.py [--split dev] [--outcome visit] [--results eval/results]
Reads <results>/<split>-summary-<outcome>.json (base) and <results>/ablations/<variant>/<split>-summary-<outcome>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.config import PRIMARY_OUTCOME, RESULTS

VARIANTS = ("base", "propensity", "categorical")  # in order of simplicity


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--outcome", default=PRIMARY_OUTCOME)
    ap.add_argument("--results", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    summaries = {}
    for v in VARIANTS:
        p = (a.results if v == "base" else a.results / "ablations" / v) / f"{a.split}-summary-{a.outcome}.json"
        if p.exists():
            summaries[v] = {m["model"]: m for m in json.loads(p.read_text())["metrics"]}
    if "base" not in summaries or len(summaries) < 2:
        print("need the base summary and at least one ablation summary — run train.py / evaluate.py into eval/results/ablations/<variant>/")
        return 2
    learners = sorted(set.intersection(*(set(s) for s in summaries.values())), key=lambda n: -summaries["base"][n]["qini"])
    rows, decisions = [], {}
    for name in learners:
        for v in VARIANTS:
            if v not in summaries:
                continue
            m = summaries[v][name]
            rows.append({"learner": name, "variant": v, "qini": m["qini"], "qini_ci_low": m["qini_ci_low"], "qini_ci_high": m["qini_ci_high"],
                         "per_1000_at_20pct": m["per_1000_at_20pct"], "per_1000_at_20pct_ci": m["per_1000_at_20pct_ci"]})
        cands = [r for r in rows if r["learner"] == name]
        best_low = max(r["qini_ci_low"] for r in cands)
        keep = next(r["variant"] for r in cands if r["qini_ci_low"] == best_low)  # first in simplicity order among ties
        decisions[name] = {"keep": keep, "qini_ci_low": {r["variant"]: r["qini_ci_low"] for r in cands}}
    table = pd.DataFrame(rows)
    out = {"split": a.split, "outcome": a.outcome, "rule": "keep the variant with the higher Qini CI lower bound; ties → the simpler setting",
           "rows": rows, "decisions": decisions}
    (a.results / f"ablations_{a.split}_{a.outcome}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = [f"# Ablations on {a.split} — outcome `{a.outcome}`", "", "Rule, fixed before the numbers: keep the variant whose Qini CI lower bound is higher; "
          "ties → the simpler setting.", "", table.to_markdown(index=False), ""]
    md += [f"- **{n}**: keep `{d['keep']}` (CI lower bounds: " + ", ".join(f"{v} {lo:.4f}" for v, lo in d["qini_ci_low"].items()) + ")" for n, d in decisions.items()]
    (a.results / f"ablations_{a.split}_{a.outcome}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(decisions, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
