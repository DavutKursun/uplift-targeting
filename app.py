"""FastAPI + Gradio serving — built in Milestone 5. Planned routes:

POST /score            Customer (f0..f11) → {uplift, p_control, p_treated, segment, decision, budget, threshold, ...}
POST /score/batch      up to 1,000 customers
GET  /policy?budget=   threshold, expected incremental visits per 1,000 contacts, profit under the stated economics
GET  /health
UI   /ui               Customer · Budget · Evidence · About
"""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("app.py is built in Milestone 5; run the scripts in scripts/ until then")
