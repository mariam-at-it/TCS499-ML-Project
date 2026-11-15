import secrets
from datetime import UTC, datetime

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    AuditLog,
    DatasetVersion,
    Job,
    JobStatus,
    JobType,
    DatasetStatus,
    ModelState,
    ModelVersion,
    RunStatus,
    TrainingRun,
    User,
    UserRole,
)
from app.schemas.admin import DatasetImportRequest, TrainingRunConfig, WeatherSyncRequest


def create_job(db: Session, job_type: JobType, payload: dict) -> Job:
    job = Job(job_type=job_type, status=JobStatus.PENDING, payload=payload, result={})
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def queue_dataset_import(db: Session, request: DatasetImportRequest, actor: User) -> Job:
    payload = request.model_dump()
    payload["requested_by"] = actor.id
    return create_job(db, JobType.DATASET_IMPORT, payload)


def queue_weather_sync(db: Session, request: WeatherSyncRequest, actor: User) -> Job:
    payload = request.model_dump()
    payload["requested_by"] = actor.id
    return create_job(db, JobType.WEATHER_SYNC, payload)


def create_training_run(db: Session, config: TrainingRunConfig, actor: User) -> TrainingRun:
    dataset_version = db.get(DatasetVersion, config.dataset_version_id)
    if dataset_version is None or dataset_version.status != DatasetStatus.READY:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Dataset version is not ready")

    run = TrainingRun(
        dataset_version_id=config.dataset_version_id,
        status=RunStatus.PENDING,
        config=config.model_dump(),
        metrics_summary={},
        privacy_summary={},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    create_job(
        db,
        JobType.TRAINING_RUN,
        {"run_id": run.id, "requested_by": actor.id, "dataset_version_id": config.dataset_version_id},
    )
    return run


def cancel_training_run(db: Session, run_id: int, actor: User | None = None) -> TrainingRun:
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    now = datetime.now(UTC)
    run.status = RunStatus.CANCELLED
    run.error_message = "Cancelled by admin"
    if run.finished_at is None:
        run.finished_at = now
    db.add(run)

    active_jobs = list(
        db.scalars(
            select(Job).where(
                Job.job_type == JobType.TRAINING_RUN,
                Job.status.in_([JobStatus.PENDING, JobStatus.RUNNING]),
            )
        )
    )
    for job in active_jobs:
        if (job.payload or {}).get("run_id") != run_id:
            continue
        job.status = JobStatus.CANCELLED
        job.error_message = "Cancelled by admin"
        job.finished_at = now
        db.add(job)

    if actor is not None:
        db.add(
            AuditLog(
                actor_user_id=actor.id,
                action="training_run.cancelled",
                entity_type="training_run",
                entity_id=str(run.id),
                details={},
            )
        )

    db.commit()
    db.refresh(run)
    return run


def mark_model_validated(db: Session, model_id: int, notes: str | None, actor: User) -> ModelVersion:
    model = db.get(ModelVersion, model_id)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    model.state = ModelState.VALIDATED
    model.validation_notes = notes
    model.validated_at = datetime.now(UTC)
    db.add(model)
    db.add(AuditLog(actor_user_id=actor.id, action="model.validated", entity_type="model_version", entity_id=str(model.id), details={"notes": notes or ""}))
    db.commit()
    db.refresh(model)
    return model


def mark_model_approved(db: Session, model_id: int, notes: str | None, actor: User) -> ModelVersion:
    model = db.get(ModelVersion, model_id)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    model.state = ModelState.APPROVED
    model.validation_notes = notes
    model.approved_at = datetime.now(UTC)
    db.add(model)
    db.add(AuditLog(actor_user_id=actor.id, action="model.approved", entity_type="model_version", entity_id=str(model.id), details={"notes": notes or ""}))
    db.commit()
    db.refresh(model)
    return model


def queue_model_publication(db: Session, model_id: int, actor: User) -> Job:
    model = db.get(ModelVersion, model_id)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    if model.state != ModelState.APPROVED:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model must be approved before publication")
    return create_job(db, JobType.PUBLISH_MODEL, {"model_version_id": model_id, "requested_by": actor.id})


def ensure_fixture_admin(db: Session) -> None:
    existing = db.scalar(select(User).where(User.email == "admin@demo.local"))
    if existing is not None:
        return
    user = User(
        email="admin@demo.local",
        full_name="Fixture Admin",
        role=UserRole.ADMIN,
        password_hash="$argon2id$v=19$m=65536,t=3,p=4$rf0D7qsX8OkQV4KfYttYiw$ltTcWHSMZKhXTOi6vj9kP8hjNtj7EeNliw8Z4Quuxmg",
        is_active=True,
        activated_at=datetime.now(UTC),
    )
    db.add(user)
    db.commit()


def create_activation_token() -> str:
    return secrets.token_urlsafe(24)
