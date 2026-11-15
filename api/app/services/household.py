from fastapi import HTTPException, status
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.models import ForecastSnapshot, HouseholdSummarySnapshot, RecommendationSnapshot, User, UserRole


def household_id_for_user(user: User, requested_household_id: int | None = None) -> int:
    if user.role == UserRole.ADMIN:
        if requested_household_id is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="household_id is required for admin requests")
        return requested_household_id
    if user.household_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Household mapping not found")
    return user.household_id


def get_latest_history(db: Session, household_id: int) -> HouseholdSummarySnapshot:
    snapshot = db.scalar(
        select(HouseholdSummarySnapshot)
        .where(HouseholdSummarySnapshot.household_id == household_id)
        .order_by(desc(HouseholdSummarySnapshot.issued_at))
    )
    if snapshot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History snapshot not found")
    return snapshot


def get_latest_forecast(db: Session, household_id: int) -> ForecastSnapshot:
    snapshot = db.scalar(
        select(ForecastSnapshot).where(ForecastSnapshot.household_id == household_id).order_by(desc(ForecastSnapshot.issued_at))
    )
    if snapshot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Forecast snapshot not found")
    return snapshot


def get_latest_recommendations(db: Session, household_id: int) -> list[RecommendationSnapshot]:
    recommendations = list(
        db.scalars(
            select(RecommendationSnapshot)
            .where(RecommendationSnapshot.household_id == household_id)
            .order_by(desc(RecommendationSnapshot.created_at))
        )
    )
    if not recommendations:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recommendations not found")
    return recommendations

