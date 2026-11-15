from datetime import datetime
from typing import Any

from pydantic import BaseModel

from app.models import DatasetStatus, JobStatus, JobType, ModelState, RunStatus


class HealthResponse(BaseModel):
    status: str
    profile: str


class JobResponse(BaseModel):
    id: int
    job_type: JobType
    status: JobStatus
    payload: dict[str, Any]
    result: dict[str, Any]
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    model_config = {"from_attributes": True}


class DatasetVersionResponse(BaseModel):
    id: int
    name: str
    source_type: str
    profile: str
    status: DatasetStatus
    import_manifest: dict[str, Any]
    household_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TrainingRunResponse(BaseModel):
    id: int
    dataset_version_id: int
    status: RunStatus
    config: dict[str, Any]
    metrics_summary: dict[str, Any]
    privacy_summary: dict[str, Any]
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    model_config = {"from_attributes": True}


class RoundMetricResponse(BaseModel):
    round_number: int
    metrics: dict[str, Any]
    bytes_transferred: int
    duration_seconds: float
    participant_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelVersionResponse(BaseModel):
    id: int
    dataset_version_id: int
    run_id: int | None
    version_label: str
    state: ModelState
    metrics: dict[str, Any]
    privacy_budget: dict[str, Any]
    artifact_manifest: dict[str, Any]
    validation_notes: str | None
    created_at: datetime
    validated_at: datetime | None
    approved_at: datetime | None
    published_at: datetime | None

    model_config = {"from_attributes": True}

