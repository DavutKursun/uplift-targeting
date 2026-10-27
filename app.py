"""uplift-targeting API + UI: FastAPI on port 7860 with Gradio mounted at /ui.

POST /score?model=s_learner|r_learner[&budget=]   Customer (12 features) → {uplift, p_control, p_treated, segment, segment_note,
                                                  decision, budget, threshold, outcome, model, version, latency_ms}
POST /score/batch                                 {"customers": [...]} (≤ 1,000) → the same per row
GET  /policy?budget=0.10[&model=]                 threshold (dev score quantile), expected incremental visits per 1,000 contacts
                                                  (dev, raw and IPW-weighted), profit under the stated economics, break-even budget
GET  /health
UI   /ui                                          Customer · Budget · Evidence · About

Run: uvicorn app:app --port 7860      Budget in force: BUDGET=0.10 (default, the headline operating point)
Requests are logged without any feature value. Nothing is stored.
"""

from __future__ import annotations

import logging
import os
import time

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no telemetry from the UI

from fastapi import FastAPI, Query, Request
from fastapi.responses import RedirectResponse

from uplift.schema import BatchRequest, BatchResponse, Customer, ModelName, PolicyResponse, UpliftResponse
from uplift.serving import MAX_BUDGET, MIN_BUDGET, Scorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("uplift.app")

scorer = Scorer.load()
app = FastAPI(title="uplift-targeting", version=scorer.version,
              description="Who should get the campaign: estimated uplift, segment and treat / skip under a budget, from randomized-experiment data.")


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    if not request.url.path.startswith("/ui"):
        log.info("%s %s → %s in %.1f ms", request.method, request.url.path, response.status_code, (time.perf_counter() - t0) * 1000)
    return response


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui")


@app.get("/health")
def health():
    return scorer.health()


@app.get("/policy", response_model=PolicyResponse)
def policy(budget: float | None = Query(default=None, ge=MIN_BUDGET, le=MAX_BUDGET), model: ModelName = "s_learner"):
    return scorer.policy(budget, model)


@app.post("/score", response_model=UpliftResponse)
def score(customer: Customer, model: ModelName = "s_learner", budget: float | None = Query(default=None, ge=MIN_BUDGET, le=MAX_BUDGET)):
    r = scorer.score(model, customer, budget)
    log.info("score model=%s budget=%.2f decision=%s segment=%s latency_ms=%.1f", model, r.budget, r.decision, r.segment, r.latency_ms)
    return r


@app.post("/score/batch", response_model=BatchResponse)
def score_batch(batch: BatchRequest, model: ModelName = "s_learner", budget: float | None = Query(default=None, ge=MIN_BUDGET, le=MAX_BUDGET)):
    t0 = time.perf_counter()
    results = scorer.score_many(model, batch.customers, budget)
    ms = (time.perf_counter() - t0) * 1000
    log.info("score/batch model=%s budget=%.2f n=%d latency_ms=%.1f", model, results[0].budget, len(results), ms)
    return BatchResponse(n=len(results), results=results, model=model, budget=results[0].budget, threshold=results[0].threshold, version=scorer.version, latency_ms=round(ms, 1))


try:
    import gradio as gr

    from uplift.ui import build

    app = gr.mount_gradio_app(app, build(scorer), path="/ui")
except ImportError:  # the API works without the UI
    log.warning("gradio not installed; /ui is disabled")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=7860)
