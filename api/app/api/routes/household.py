from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import require_household_or_admin
from app.db.session import get_db
from app.schemas.household import (
    ForecastResponse,
    HouseholdContextResponse,
    HouseholdHistoryResponse,
    RecommendationResponse,
)
from app.services.household import household_id_for_user, get_latest_forecast, get_latest_history, get_latest_recommendations

router = APIRouter(prefix="/household", tags=["household"])


@router.get("/history", response_model=HouseholdHistoryResponse)
def history(
    household_id: int | None = Query(default=None),
    user=Depends(require_household_or_admin),
    db: Session = Depends(get_db),
) -> HouseholdHistoryResponse:
    resolved_id = household_id_for_user(user, household_id)
    snapshot = get_latest_history(db, resolved_id)
    return HouseholdHistoryResponse(
        household_id=snapshot.household_id,
        household_code=snapshot.household.household_code,
        issued_at=snapshot.issued_at,
        start_at=snapshot.start_at,
        end_at=snapshot.end_at,
        series=snapshot.history,
        context=snapshot.context,
        model_version_id=snapshot.model_version_id,
    )


@router.get("/forecast", response_model=ForecastResponse)
def forecast(
    household_id: int | None = Query(default=None),
    user=Depends(require_household_or_admin),
    db: Session = Depends(get_db),
) -> ForecastResponse:
    resolved_id = household_id_for_user(user, household_id)
    snapshot = get_latest_forecast(db, resolved_id)
    return ForecastResponse(
        household_id=snapshot.household_id,
        issued_at=snapshot.issued_at,
        horizon_start=snapshot.horizon_start,
        horizon_end=snapshot.horizon_end,
        forecast=snapshot.forecast,
        context=snapshot.context,
        model_version_id=snapshot.model_version_id,
        run_id=snapshot.run_id,
    )


@router.get("/context", response_model=HouseholdContextResponse)
def context(
    household_id: int | None = Query(default=None),
    user=Depends(require_household_or_admin),
    db: Session = Depends(get_db),
) -> HouseholdContextResponse:
    resolved_id = household_id_for_user(user, household_id)
    snapshot = get_latest_forecast(db, resolved_id)
    return HouseholdContextResponse(
        household_id=snapshot.household_id,
        model_version_id=snapshot.model_version_id,
        context=snapshot.context,
    )


@router.get("/recommendations", response_model=list[RecommendationResponse])
def recommendations(
    household_id: int | None = Query(default=None),
    user=Depends(require_household_or_admin),
    db: Session = Depends(get_db),
) -> list[RecommendationResponse]:
    resolved_id = household_id_for_user(user, household_id)
    items = get_latest_recommendations(db, resolved_id)
    return [
        RecommendationResponse(
            household_id=item.household_id,
            code=item.code,
            severity=item.severity,
            evidence=item.evidence,
            message=item.message,
            valid_until=item.valid_until,
            model_version_id=item.model_version_id,
        )
        for item in items
    ]

