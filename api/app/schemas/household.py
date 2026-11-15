from datetime import datetime
from typing import Any

from pydantic import BaseModel


class HouseholdHistoryResponse(BaseModel):
    household_id: int
    household_code: str
    issued_at: datetime
    start_at: datetime
    end_at: datetime
    series: list[dict[str, Any]]
    context: dict[str, Any]
    model_version_id: int


class ForecastResponse(BaseModel):
    household_id: int
    issued_at: datetime
    horizon_start: datetime
    horizon_end: datetime
    forecast: list[dict[str, Any]]
    context: dict[str, Any]
    model_version_id: int
    run_id: int | None


class RecommendationResponse(BaseModel):
    household_id: int
    code: str
    severity: str
    evidence: dict[str, Any]
    message: str
    valid_until: datetime
    model_version_id: int


class HouseholdContextResponse(BaseModel):
    household_id: int
    model_version_id: int
    context: dict[str, Any]

