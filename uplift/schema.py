"""API schema: a customer is the 12 anonymous Criteo features; the response is an uplift, a segment and a decision."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Segment = Literal["persuadable", "sure_thing", "lost_cause", "sleeping_dog"]


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
    uplift: float = Field(description="estimated change in outcome probability caused by the treatment")
    p_control: float = Field(ge=0, le=1)
    p_treated: float = Field(ge=0, le=1)
    segment: Segment
    decision: Literal["treat", "skip"]
    budget: float = Field(ge=0, le=1)
    threshold: float
    outcome: str
    model: str
    version: str
    latency_ms: float
