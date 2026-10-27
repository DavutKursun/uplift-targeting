"""API schema: a customer is the 12 anonymous Criteo features; the response is an uplift, a segment and a decision.

Segment names in the API: persuadable, sure_thing, lost_cause, no_detectable_effect. The last is the textbook "sleeping dog"
(predicted uplift below −δ), renamed because held-out data did not confirm any harm (DESIGN.md §4.3 and §5.5); every response
that carries it also carries SEGMENT_NOTE.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Segment = Literal["persuadable", "sure_thing", "lost_cause", "no_detectable_effect"]
ModelName = Literal["s_learner", "r_learner"]
SEGMENT_NOTE = "negative predicted uplift was not confirmed on held-out data; do not treat as harm"


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    f0: float
    f1: float
    f2: float
    f3: float
    f4: float
    f5: float
    f6: float
    f7: float
    f8: float
    f9: float
    f10: float
    f11: float

    def row(self) -> list[float]:
        return [getattr(self, f"f{i}") for i in range(12)]


class UpliftResponse(BaseModel):
    uplift: float = Field(description="estimated change in visit probability caused by the treatment")
    p_control: float = Field(ge=0, le=1)
    p_treated: float = Field(ge=0, le=1)
    segment: Segment
    segment_note: str | None = None
    decision: Literal["treat", "skip"]
    budget: float = Field(ge=0, le=1)
    threshold: float
    outcome: str
    model: str
    version: str
    latency_ms: float


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customers: list[Customer] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    n: int
    results: list[UpliftResponse]
    model: str
    budget: float
    threshold: float
    version: str
    latency_ms: float


class PolicyResponse(BaseModel):
    budget: float
    model: str
    threshold: float
    incremental_per_1000: float
    ipw_per_1000: float
    share_of_total: float | None
    profit_per_1000_contacts: float
    ipw_profit_per_1000_contacts: float
    profit_per_1000_customers: float
    ipw_profit_per_1000_customers: float
    breakeven_budget: float | None
    profit_max_budget: float
    random_per_1000: float
    economics: dict
    segments_among_targeted: dict | None
    source: str
    version: str
