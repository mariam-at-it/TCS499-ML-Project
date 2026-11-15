from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy import Boolean, DateTime, Enum as SqlEnum, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class UserRole(str, Enum):
    ADMIN = "admin"
    HOUSEHOLD = "household"


class JobType(str, Enum):
    DATASET_IMPORT = "dataset_import"
    WEATHER_SYNC = "weather_sync"
    TRAINING_RUN = "training_run"
    PUBLISH_MODEL = "publish_model"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DatasetStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelState(str, Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    APPROVED = "approved"


class Household(Base):
    __tablename__ = "households"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    household_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(255))
    region_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    users: Mapped[list["User"]] = relationship(back_populates="household")
    summary_snapshots: Mapped[list["HouseholdSummarySnapshot"]] = relationship(back_populates="household")
    forecast_snapshots: Mapped[list["ForecastSnapshot"]] = relationship(back_populates="household")
    recommendation_snapshots: Mapped[list["RecommendationSnapshot"]] = relationship(back_populates="household")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(SqlEnum(UserRole, name="user_role"))
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    activation_token: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    household_id: Mapped[int | None] = mapped_column(ForeignKey("households.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    household: Mapped[Household | None] = relationship(back_populates="users")
    audit_logs: Mapped[list["AuditLog"]] = relationship(back_populates="actor_user")
    sessions: Mapped[list["UserSession"]] = relationship(back_populates="user")


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    source_type: Mapped[str] = mapped_column(String(64))
    profile: Mapped[str] = mapped_column(String(64))
    status: Mapped[DatasetStatus] = mapped_column(SqlEnum(DatasetStatus, name="dataset_status"))
    import_manifest: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    household_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    training_runs: Mapped[list["TrainingRun"]] = relationship(back_populates="dataset_version")
    model_versions: Mapped[list["ModelVersion"]] = relationship(back_populates="dataset_version")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_type: Mapped[JobType] = mapped_column(SqlEnum(JobType, name="job_type"))
    status: Mapped[JobStatus] = mapped_column(SqlEnum(JobStatus, name="job_status"))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"))
    status: Mapped[RunStatus] = mapped_column(SqlEnum(RunStatus, name="run_status"))
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    privacy_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    dataset_version: Mapped[DatasetVersion] = relationship(back_populates="training_runs")
    round_metrics: Mapped[list["TrainingRoundMetric"]] = relationship(back_populates="run")
    model_versions: Mapped[list["ModelVersion"]] = relationship(back_populates="run")


class TrainingRoundMetric(Base):
    __tablename__ = "training_round_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"))
    round_number: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    bytes_transferred: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    participant_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[TrainingRun] = relationship(back_populates="round_metrics")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"))
    run_id: Mapped[int | None] = mapped_column(ForeignKey("training_runs.id"), nullable=True)
    version_label: Mapped[str] = mapped_column(String(128), unique=True)
    state: Mapped[ModelState] = mapped_column(SqlEnum(ModelState, name="model_state"))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    privacy_budget: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    artifact_manifest: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    validation_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    validated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    dataset_version: Mapped[DatasetVersion] = relationship(back_populates="model_versions")
    run: Mapped[TrainingRun | None] = relationship(back_populates="model_versions")
    summary_snapshots: Mapped[list["HouseholdSummarySnapshot"]] = relationship(back_populates="model_version")
    forecast_snapshots: Mapped[list["ForecastSnapshot"]] = relationship(back_populates="model_version")
    recommendation_snapshots: Mapped[list["RecommendationSnapshot"]] = relationship(back_populates="model_version")


class HouseholdSummarySnapshot(Base):
    __tablename__ = "household_summary_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    household_id: Mapped[int] = mapped_column(ForeignKey("households.id"))
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    source_run_id: Mapped[int | None] = mapped_column(ForeignKey("training_runs.id"), nullable=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    start_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    history: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    context: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    household: Mapped[Household] = relationship(back_populates="summary_snapshots")
    model_version: Mapped[ModelVersion] = relationship(back_populates="summary_snapshots")


class ForecastSnapshot(Base):
    __tablename__ = "forecast_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    household_id: Mapped[int] = mapped_column(ForeignKey("households.id"))
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    run_id: Mapped[int | None] = mapped_column(ForeignKey("training_runs.id"), nullable=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    horizon_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    horizon_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    forecast: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    context: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    household: Mapped[Household] = relationship(back_populates="forecast_snapshots")
    model_version: Mapped[ModelVersion] = relationship(back_populates="forecast_snapshots")


class RecommendationSnapshot(Base):
    __tablename__ = "recommendation_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    household_id: Mapped[int] = mapped_column(ForeignKey("households.id"))
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    run_id: Mapped[int | None] = mapped_column(ForeignKey("training_runs.id"), nullable=True)
    code: Mapped[str] = mapped_column(String(64))
    severity: Mapped[str] = mapped_column(String(32))
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    message: Mapped[str] = mapped_column(Text)
    valid_until: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    household: Mapped[Household] = relationship(back_populates="recommendation_snapshots")
    model_version: Mapped[ModelVersion] = relationship(back_populates="recommendation_snapshots")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    actor_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(128))
    entity_type: Mapped[str] = mapped_column(String(128))
    entity_id: Mapped[str] = mapped_column(String(128))
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    actor_user: Mapped[User | None] = relationship(back_populates="audit_logs")


class UserSession(Base):
    __tablename__ = "user_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    refresh_jti: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    token_family: Mapped[str] = mapped_column(String(64), index=True)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    replaced_by_jti: Mapped[str | None] = mapped_column(String(64), nullable=True)

    user: Mapped[User] = relationship(back_populates="sessions")
