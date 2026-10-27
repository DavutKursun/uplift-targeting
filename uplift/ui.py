"""The demo UI (Gradio, mounted at /ui): Customer · Budget · Evidence · About.

Reads only committed files: the 2,000-customer dev sample and the dev policy grid under models/serving/, and the test-split
plots and tables under eval/results/ (labelled as test, used once). Nothing typed here is stored.
"""

from __future__ import annotations

import json

import gradio as gr
import pandas as pd
from PIL import Image
from pydantic import ValidationError

from .config import FEATURES, RESULTS
from .schema import SEGMENT_NOTE, Customer
from .serving import SERVING, Scorer

TEST_LABEL = "test split, used once on 2026-10-28"
ABOUT = """
## About this demo
**Data.** Criteo Uplift Prediction Dataset v2.1: 13.98M customers of an online advertiser, a randomized treatment (85% treated),
outcomes `visit` (4.7%) and `conversion` (0.29%), twelve anonymous features. Released by Criteo AI Lab under **CC BY-NC-SA 4.0**
(non-commercial, attribution, share-alike); cite Diemert et al., *A Large Scale Benchmark for Uplift Modeling*, AdKDD 2018.

**What the models do.** The S-learner estimates each customer's uplift, the change in visit probability the ad causes, as
f(x, treated) − f(x, control) from one LightGBM model; the R-learner is the cross-check. The response model, kept in every
comparison, ranks by P(visit | treated): who visits, not whom the ad moves. Thresholds come from the dev split (800k customers)
at the chosen budget; the numbers quoted as *test* come from a split of 800k customers opened once, on 2026-10-28.

**The experiment is not perfect.** Treatment is weakly predictable from the features (AUC 0.51, max |SMD| 0.05) and every
covariate adjustment lowers the average effect from 1.03 to about 0.8 percentage points. Raw treated-minus-control numbers are
primary, as in the dataset paper; IPW-weighted numbers (cross-fitted propensity) are shown next to them and run about a quarter
lower at broad budgets without changing any ranking.

**Segments.** persuadable (uplift > δ), sure_thing (little uplift, high base rate), lost_cause (little uplift, low base rate),
and **no_detectable_effect** for a predicted uplift below −δ: the textbook "sleeping dog", renamed because held-out data did
not confirm any harm — the flagged customers showed +10 pp of observed uplift on test. δ = a quarter of the average effect,
p0_high = twice the control visit rate, both fixed on dev.

**Features are anonymous**, so the service can say *that* a customer is persuadable, not why.

**This is a historical experiment, not a live marketing system; nothing is stored.** The server logs the method, path,
status and latency of each call, never a feature value.

Repository: https://github.com/DavutKursun/uplift-targeting · design notes `docs/DESIGN.md` · model card `docs/model_card.md`
"""


def _img(name: str):
    p = RESULTS / name
    return Image.open(p) if p.exists() else None


def _customers() -> pd.DataFrame:
    p = SERVING / "customers_dev_sample.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["id", *FEATURES])


def _paired_table() -> pd.DataFrame:
    p = RESULTS / "paired_test_visit.json"
    if not p.exists():
        return pd.DataFrame()
    rows = json.loads(p.read_text(encoding="utf-8"))["table"]
    return pd.DataFrame([{"difference": r["pair"].replace("-", " − "), "statistic": r["statistic"].replace("per_1000_at_", "per 1,000 at ").replace("pct", "%"),
                          "point": r["difference"], "95% CI": f"[{r['ci_low']}, {r['ci_high']}]", "excludes 0": "yes" if r["excludes_zero"] else "no"} for r in rows])


def _placebo_md() -> str:
    parts = ["### Placebo: treatment labels shuffled before training"]
    null = RESULTS / "placebo_null_dev_visit_s_learner.json"
    if null.exists():
        d = json.loads(null.read_text(encoding="utf-8"))
        vals = ", ".join(f"{s['qini_placebo']['point']:+.3f}" for s in d["shuffles"])
        sp = d["spread"]
        verdict = ""
        if "compare_qini" in d:
            verdict = f" The deployed model's test Qini, **{d['compare_qini']:.4f}**, lies {'outside' if d['outside_spread'] else 'INSIDE'} that spread."
        parts.append(f"{d['n_shuffles']} independent shuffles fitted with the deployed settings and scored on dev give Qini {vals}: the placebo null spans "
                     f"**{sp['min']:+.3f} to {sp['max']:+.3f}** (SD {sp['sd']:.3f}).{verdict}")
    for split, label in (("dev", "dev"), ("test", "test, one shuffle")):
        p = RESULTS / f"placebo_{split}_visit_s_learner.json"
        if p.exists():
            q = json.loads(p.read_text(encoding="utf-8"))["qini_placebo"]
            parts.append(f"Single shuffle (seed 7) scored on {label}: Qini {q['point']:+.4f} [{q['ci_low']:+.4f}, {q['ci_high']:+.4f}] — reported, not used as the criterion.")
    return "\n\n".join(parts)


def _dev_vs_test_md() -> str:
    p = RESULTS / "dev_vs_test.md"
    return p.read_text(encoding="utf-8") if p.exists() else "*dev → test table not available in this build*"


def _segments_df(segs: dict | None) -> pd.DataFrame:
    if not segs:
        return pd.DataFrame({"segment": [], "share among the targeted": []})
    return pd.DataFrame({"segment": list(segs), "share among the targeted": [f"{v * 100:.1f}%" for v in segs.values()]})


def build(scorer: Scorer) -> gr.Blocks:
    customers = _customers()
    ids = [str(i) for i in customers["id"].tolist()]
    first = customers.iloc[0] if len(customers) else None

    def fill(cid: str):
        row = customers.loc[customers["id"] == int(cid)].iloc[0]
        return [float(row[f]) for f in FEATURES]

    def score_both(budget_pct: float, *values):
        try:
            c = Customer(**{f: float(v) for f, v in zip(FEATURES, values, strict=True)})
        except (ValidationError, ValueError, TypeError) as e:
            msg = f"**Invalid customer:** {str(e).splitlines()[0][:200]}"
            return msg, msg
        b = float(budget_pct) / 100
        s = scorer.score("s_learner", c, b)
        cross = scorer.score("r_learner", c, b)
        resp = scorer.score_response([c], b)
        if resp:
            r = resp[0]
            resp_md = (f"### Response model\n**P(visit | treated) = {r['score']:.3f}** → **{r['decision'].upper()}** at a {budget_pct:.0f}% budget "
                       f"(its own threshold: P(visit) ≥ {r['threshold']:.3f})\n\nRanks by who would visit, not by what the ad changes: no uplift, no segment.")
        else:
            resp_md = "*response model not loaded*"
        seg_line = f"Segment **{s.segment}**" + (f" — *{SEGMENT_NOTE}*" if s.segment_note else "")
        s_md = (f"### S-learner (deployed)\n**uplift = {s.uplift * 100:+.2f} pp** — P(visit | control) {s.p_control:.3f} → P(visit | treated) {s.p_treated:.3f} "
                f"→ **{s.decision.upper()}** at a {budget_pct:.0f}% budget (threshold {s.threshold * 100:.2f} pp)\n\n{seg_line}\n\n"
                f"Cross-check R-learner: uplift {cross.uplift * 100:+.2f} pp → {cross.decision}")
        return resp_md, s_md

    def policy_view(budget_pct: float):
        b = float(budget_pct) / 100
        p, r, q = scorer.policy(b, "s_learner"), scorer.policy(b, "r_learner"), scorer.policy(b, "response")
        econ = p["economics"]
        money = "makes money" if p["profit_per_1000_contacts"] > 0 else "loses money"
        md = (f"### At a {budget_pct:.0f}% budget — dev split, {scorer.grid['n']:,} customers; random targeting = {p['random_per_1000']:.1f} per 1,000\n\n"
              "| | S-learner (deployed) | R-learner (cross-check) | response model |\n|---|---|---|---|\n"
              f"| threshold on the score | uplift ≥ {p['threshold'] * 100:.2f} pp | uplift ≥ {r['threshold'] * 100:.2f} pp | P(visit) ≥ {q['threshold']:.3f} |\n"
              f"| incremental visits per 1,000 contacts, raw | **{p['incremental_per_1000']:.1f}** | {r['incremental_per_1000']:.1f} | {q['incremental_per_1000']:.1f} |\n"
              f"| IPW-weighted (imbalance correction) | {p['ipw_per_1000']:.1f} | {r['ipw_per_1000']:.1f} | {q['ipw_per_1000']:.1f} |\n"
              f"| share of the total incremental effect | {p['share_of_total'] * 100:.0f}% | {r['share_of_total'] * 100:.0f}% | {q['share_of_total'] * 100:.0f}% |\n"
              f"| profit per 1,000 contacts | **{p['profit_per_1000_contacts']:+.1f}** | {r['profit_per_1000_contacts']:+.1f} | {q['profit_per_1000_contacts']:+.1f} |\n"
              f"| cumulative profit per 1,000 customers | {p['profit_per_1000_customers']:+.2f} | {r['profit_per_1000_customers']:+.2f} | {q['profit_per_1000_customers']:+.2f} |\n"
              f"| break-even budget / profit-maximising budget | {p['breakeven_budget']:.0%} / {p['profit_max_budget']:.0%} | {r['breakeven_budget']:.0%} / {r['profit_max_budget']:.0%} | "
              f"{q['breakeven_budget']:.0%} / {q['profit_max_budget']:.0%} |\n\n"
              f"Economics are **assumptions**: {econ['value_per_visit']:.2f} per incremental visit, {econ['contact_cost']:.2f} per contact, so one more contact pays only above "
              f"{econ['marginal_threshold_per_1000']:.0f} incremental visits per 1,000. At {budget_pct:.0f}% the deployed policy **{money}**.")
        return md, _segments_df(p["segments_among_targeted"])

    with gr.Blocks(title="uplift-targeting") as demo:
        gr.Markdown(f"# uplift-targeting — who should get the campaign?\nEstimated uplift, segment and treat / skip under a budget, from a randomized experiment. "
                    f"Budget in force **{scorer.budget:.0%}** (S-learner threshold {scorer.threshold('s_learner') * 100:.2f} pp of uplift). Version {scorer.version}.")
        with gr.Tabs():
            with gr.Tab("Customer"):
                gr.Markdown("Pick one of 2,000 dev customers (anonymous features, ids only) or edit the twelve fields, then score with both models.")
                with gr.Row():
                    cid = gr.Dropdown(choices=ids, value=ids[0] if ids else None, label="Dev customer id")
                    budget_c = gr.Slider(5, 50, value=round(scorer.budget * 100), step=1, label="Budget (% of customers contacted)")
                fields = []
                for start in range(0, 12, 4):
                    with gr.Row():
                        for j in range(start, start + 4):
                            fields.append(gr.Number(value=float(first[FEATURES[j]]) if first is not None else 0.0, label=FEATURES[j]))
                btn = gr.Button("Score with both models", variant="primary")
                with gr.Row():
                    out_resp = gr.Markdown("*Response model: press Score*")
                    out_s = gr.Markdown("*S-learner: press Score*")
                gr.Markdown("**Why they can disagree:** the response model ranks by who would visit, the S-learner by what the ad changes. On Criteo the effect is "
                            "roughly proportional to the base rate, so they agree for most customers and part ways for a customer who visits anyway (sure thing) or "
                            "one with a low base rate and a real change — the tight-budget region where uplift targeting found 14 more visits per 1,000 contacts on test.")
                btn.click(score_both, inputs=[budget_c, *fields], outputs=[out_resp, out_s])
                cid.change(fill, inputs=[cid], outputs=fields).then(score_both, inputs=[budget_c, *fields], outputs=[out_resp, out_s])
                budget_c.change(score_both, inputs=[budget_c, *fields], outputs=[out_resp, out_s])
            with gr.Tab("Budget"):
                budget_b = gr.Slider(5, 50, value=round(scorer.budget * 100), step=1, label="Budget (% of customers contacted)")
                md0, df0 = policy_view(round(scorer.budget * 100))
                out_policy = gr.Markdown(md0)
                out_segments = gr.Dataframe(value=df0, label="Four-segment breakdown among the targeted (S-learner, dev)", interactive=False)
                budget_b.change(policy_view, inputs=[budget_b], outputs=[out_policy, out_segments])
                gr.Markdown(f"**Budget and profit curves — {TEST_LABEL}.** Raw and IPW-weighted panels; the uplift pair against the response model and random.")
                with gr.Row():
                    gr.Image(value=_img("budget_curve_test.png"), label=f"Incremental visits per 1,000 contacts vs budget ({TEST_LABEL})", interactive=False)
                    gr.Image(value=_img("profit_curve_test.png"), label=f"Cumulative profit vs budget ({TEST_LABEL})", interactive=False)
            with gr.Tab("Evidence"):
                gr.Markdown(f"## What the holdout shows — {TEST_LABEL}\nQini curves and the deployed learner's decile check, the paired differences against the "
                            "response model (same bootstrap resamples), the placebo null, and every dev number next to its test counterpart.")
                with gr.Row():
                    gr.Image(value=_img("qini_curves_test.png"), label=f"Qini curves ({TEST_LABEL})", interactive=False)
                    gr.Image(value=_img("deciles_test_s_learner.png"), label=f"Observed vs predicted uplift by decile, S-learner ({TEST_LABEL})", interactive=False)
                gr.Dataframe(value=_paired_table(), label="Paired differences on test (200 shared resamples)", interactive=False, wrap=True)
                gr.Markdown(_placebo_md())
                with gr.Accordion("Dev → test, every block", open=False):
                    gr.Markdown(_dev_vs_test_md())
            with gr.Tab("About"):
                gr.Markdown(ABOUT)
    return demo
