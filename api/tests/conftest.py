from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.core.security import hash_password
from app.main import app
from app.models import (
    Base,
    DatasetStatus,
    DatasetVersion,
    ForecastSnapshot,
    Household,
    HouseholdSummarySnapshot,
    Job,
    JobStatus,
    JobType,
    ModelState,
    ModelVersion,
    RecommendationSnapshot,
    RunStatus,
    TrainingRun,
    User,
    UserRole,
)


@pytest.fixture()
def session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    db = TestingSessionLocal()
    household = Household(household_code="HOUSEHOLD-0001", display_name="Fixture Household 1", region_code="DUBLIN", metadata_json={})
    other_household = Household(household_code="HOUSEHOLD-0002", display_name="Fixture Household 2", region_code="CORK", metadata_json={})
    db.add_all([household, other_household])
    db.flush()
    admin = User(
        email="admin@demo.local",
        full_name="Admin User",
        role=UserRole.ADMIN,
        password_hash=hash_password("Admin123!"),
        is_active=True,
        activated_at=datetime.now(UTC),
    )
    household_user = User(
        email="household-0001@demo.local",
        full_name="Household User",
        role=UserRole.HOUSEHOLD,
        password_hash=hash_password("Household123!"),
        household_id=household.id,
        is_active=True,
        activated_at=datetime.now(UTC),
    )
    invited_household_user = User(
        email="household-0002@demo.local",
        full_name="Invited Household User",
        role=UserRole.HOUSEHOLD,
        password_hash=None,
        household_id=other_household.id,
        is_active=False,
        activation_token="activate-household-0002",
    )
    dataset = DatasetVersion(
        name="fixture-001",
        source_type="fixture",
        profile="fixture",
        status=DatasetStatus.READY,
        import_manifest={"artifact_root": "/tmp/artifacts", "weather_ready": True},
        household_count=2,
    )
    run = TrainingRun(
        dataset_version=dataset,
        status=RunStatus.SUCCEEDED,
        config={"dataset_version_id": 1},
        metrics_summary={"fl_metrics": {"mae": 0.12}},
        privacy_summary={"epsilon": 3.2},
    )
    model = ModelVersion(
        dataset_version=dataset,
        run=run,
        version_label="model-run-1",
        state=ModelState.APPROVED,
        metrics={"fl_metrics": {"mae": 0.12}},
        privacy_budget={"epsilon": 3.2},
        artifact_manifest={"model_path": "/tmp/model.pt"},
        approved_at=datetime.now(UTC),
        published_at=datetime.now(UTC),
    )
    db.add_all([admin, household_user, invited_household_user, dataset, run, model])
    db.flush()
    now = datetime.now(UTC)
    db.add(
        HouseholdSummarySnapshot(
            household_id=household.id,
            model_version_id=model.id,
            source_run_id=run.id,
            issued_at=now,
            start_at=now - timedelta(days=30),
            end_at=now,
            history=[{"timestamp": now.isoformat(), "consumption_kwh": 0.8}],
            context={"weather_source": "synthetic"},
        )
    )
    db.add(
        ForecastSnapshot(
            household_id=household.id,
            model_version_id=model.id,
            run_id=run.id,
            issued_at=now,
            horizon_start=now,
            horizon_end=now + timedelta(hours=23),
            forecast=[{"timestamp": now.isoformat(), "prediction_kwh": 0.95, "temperature_2m": 12.0, "apparent_temperature": 10.5, "precipitation": 0.1}],
            context={"weather_source": "synthetic"},
        )
    )
    db.add(
        RecommendationSnapshot(
            household_id=household.id,
            model_version_id=model.id,
            run_id=run.id,
            code="shift_peak_load",
            severity="medium",
            evidence={"peak_hour": now.isoformat()},
            message="Shift discretionary loads out of the evening peak.",
            valid_until=now + timedelta(days=1),
        )
    )
    db.add(Job(job_type=JobType.DATASET_IMPORT, status=JobStatus.SUCCEEDED, payload={}, result={}, created_at=now, updated_at=now))
    db.commit()

    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield db
    finally:
        app.dependency_overrides.clear()
        db.close()
        Base.metadata.drop_all(engine)


@pytest.fixture()
def client(session: Session) -> Generator[TestClient, None, None]:
    _ = session
    with TestClient(app) as test_client:
      yield test_client
