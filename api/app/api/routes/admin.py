from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.db.session import get_db
from app.models import DatasetVersion, Household, ModelVersion, RunStatus, TrainingRoundMetric, TrainingRun, User
from app.schemas.admin import (
    DatasetImportRequest,
    HouseholdAdminResponse,
    ModelActionRequest,
    PublishModelRequest,
    TrainingRunRequest,
    WeatherSyncRequest,
)
from app.schemas.common import (
    DatasetVersionResponse,
    JobResponse,
    ModelVersionResponse,
    RoundMetricResponse,
    TrainingRunResponse,
)
from app.services.admin import (
    cancel_training_run,
    mark_model_approved,
    mark_model_validated,
    queue_dataset_import,
    queue_model_publication,
    queue_weather_sync,
    create_training_run,
)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/datasets", response_model=list[DatasetVersionResponse])
def list_datasets(_: User = Depends(require_admin), db: Session = Depends(get_db)) -> list[DatasetVersionResponse]:
    datasets = list(db.scalars(select(DatasetVersion).order_by(desc(DatasetVersion.created_at))))
    return [DatasetVersionResponse.model_validate(dataset) for dataset in datasets]


@router.post("/datasets/import", response_model=JobResponse)
def import_dataset(
    request: DatasetImportRequest,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> JobResponse:
    job = queue_dataset_import(db, request, user)
    return JobResponse.model_validate(job)


@router.post("/datasets/weather-sync", response_model=JobResponse)
def sync_weather(
    request: WeatherSyncRequest,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> JobResponse:
    job = queue_weather_sync(db, request, user)
    return JobResponse.model_validate(job)


@router.get("/runs", response_model=list[TrainingRunResponse])
def list_runs(_: User = Depends(require_admin), db: Session = Depends(get_db)) -> list[TrainingRunResponse]:
    runs = list(db.scalars(select(TrainingRun).order_by(desc(TrainingRun.created_at))))
    return [TrainingRunResponse.model_validate(run) for run in runs]


@router.post("/runs", response_model=TrainingRunResponse)
def create_run(
    request: TrainingRunRequest,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> TrainingRunResponse:
    run = create_training_run(db, request.config, user)
    return TrainingRunResponse.model_validate(run)


@router.get("/runs/{run_id}", response_model=TrainingRunResponse)
def get_run(run_id: int, _: User = Depends(require_admin), db: Session = Depends(get_db)) -> TrainingRunResponse:
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return TrainingRunResponse.model_validate(run)


@router.get("/runs/{run_id}/rounds", response_model=list[RoundMetricResponse])
def get_run_rounds(run_id: int, _: User = Depends(require_admin), db: Session = Depends(get_db)) -> list[RoundMetricResponse]:
    rows = list(db.scalars(select(TrainingRoundMetric).where(TrainingRoundMetric.run_id == run_id).order_by(TrainingRoundMetric.round_number)))
    return [RoundMetricResponse.model_validate(row) for row in rows]


@router.post("/runs/{run_id}/cancel", response_model=TrainingRunResponse)
def cancel_run(run_id: int, user: User = Depends(require_admin), db: Session = Depends(get_db)) -> TrainingRunResponse:
    run = cancel_training_run(db, run_id, user)
    return TrainingRunResponse.model_validate(run)


@router.get("/models", response_model=list[ModelVersionResponse])
def list_models(_: User = Depends(require_admin), db: Session = Depends(get_db)) -> list[ModelVersionResponse]:
    models = list(db.scalars(select(ModelVersion).order_by(desc(ModelVersion.created_at))))
    return [ModelVersionResponse.model_validate(model) for model in models]


@router.post("/models/{model_id}/validate", response_model=ModelVersionResponse)
def validate_model(
    model_id: int,
    request: ModelActionRequest,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> ModelVersionResponse:
    model = mark_model_validated(db, model_id, request.notes, user)
    return ModelVersionResponse.model_validate(model)


@router.post("/models/{model_id}/approve", response_model=ModelVersionResponse)
def approve_model(
    model_id: int,
    request: ModelActionRequest,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> ModelVersionResponse:
    model = mark_model_approved(db, model_id, request.notes, user)
    return ModelVersionResponse.model_validate(model)


@router.post("/models/publish", response_model=JobResponse)
def publish_model(
    request: PublishModelRequest,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> JobResponse:
    job = queue_model_publication(db, request.model_version_id, user)
    return JobResponse.model_validate(job)


@router.get("/households", response_model=list[HouseholdAdminResponse])
def list_households(_: User = Depends(require_admin), db: Session = Depends(get_db)) -> list[HouseholdAdminResponse]:
    households = list(db.scalars(select(Household).order_by(Household.household_code)))
    responses: list[HouseholdAdminResponse] = []
    for household in households:
        user = db.scalar(select(User).where(User.household_id == household.id))
        responses.append(
            HouseholdAdminResponse(
                id=household.id,
                household_code=household.household_code,
                display_name=household.display_name,
                region_code=household.region_code,
                metadata_json=household.metadata_json,
                user_email=user.email if user else None,
                activation_token=user.activation_token if user else None,
            )
        )
    return responses
