from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum as SqlEnum,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)

metadata = MetaData()

user_role_enum = SqlEnum("ADMIN", "HOUSEHOLD", name="user_role", create_constraint=False)
job_type_enum = SqlEnum("DATASET_IMPORT", "WEATHER_SYNC", "TRAINING_RUN", "PUBLISH_MODEL", name="job_type", create_constraint=False)
job_status_enum = SqlEnum("PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED", name="job_status", create_constraint=False)
dataset_status_enum = SqlEnum("PENDING", "READY", "FAILED", name="dataset_status", create_constraint=False)
run_status_enum = SqlEnum("PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED", name="run_status", create_constraint=False)
model_state_enum = SqlEnum("DRAFT", "VALIDATED", "APPROVED", name="model_state", create_constraint=False)

households = Table(
    "households",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("household_code", String(64), nullable=False),
    Column("display_name", String(255), nullable=False),
    Column("region_code", String(64)),
    Column("latitude", Float),
    Column("longitude", Float),
    Column("metadata_json", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("email", String(255), nullable=False),
    Column("full_name", String(255), nullable=False),
    Column("role", user_role_enum, nullable=False),
    Column("password_hash", String(255)),
    Column("activation_token", String(255)),
    Column("is_active", Boolean, nullable=False),
    Column("household_id", ForeignKey("households.id")),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("activated_at", DateTime(timezone=True)),
)

user_sessions = Table(
    "user_sessions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", ForeignKey("users.id"), nullable=False),
    Column("refresh_jti", String(64), nullable=False),
    Column("token_family", String(64), nullable=False),
    Column("ip_address", String(64)),
    Column("user_agent", String(512)),
    Column("issued_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("last_used_at", DateTime(timezone=True)),
    Column("revoked_at", DateTime(timezone=True)),
    Column("replaced_by_jti", String(64)),
)

dataset_versions = Table(
    "dataset_versions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(255), nullable=False),
    Column("source_type", String(64), nullable=False),
    Column("profile", String(64), nullable=False),
    Column("status", dataset_status_enum, nullable=False),
    Column("import_manifest", JSON, nullable=False),
    Column("household_count", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

jobs = Table(
    "jobs",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("job_type", job_type_enum, nullable=False),
    Column("status", job_status_enum, nullable=False),
    Column("payload", JSON, nullable=False),
    Column("result", JSON, nullable=False),
    Column("error_message", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
)

training_runs = Table(
    "training_runs",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("dataset_version_id", ForeignKey("dataset_versions.id"), nullable=False),
    Column("status", run_status_enum, nullable=False),
    Column("config", JSON, nullable=False),
    Column("metrics_summary", JSON, nullable=False),
    Column("privacy_summary", JSON, nullable=False),
    Column("error_message", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
)

training_round_metrics = Table(
    "training_round_metrics",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("run_id", ForeignKey("training_runs.id"), nullable=False),
    Column("round_number", Integer, nullable=False),
    Column("metrics", JSON, nullable=False),
    Column("bytes_transferred", Integer, nullable=False),
    Column("duration_seconds", Float, nullable=False),
    Column("participant_count", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

model_versions = Table(
    "model_versions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("dataset_version_id", ForeignKey("dataset_versions.id"), nullable=False),
    Column("run_id", ForeignKey("training_runs.id")),
    Column("version_label", String(128), nullable=False),
    Column("state", model_state_enum, nullable=False),
    Column("metrics", JSON, nullable=False),
    Column("privacy_budget", JSON, nullable=False),
    Column("artifact_manifest", JSON, nullable=False),
    Column("validation_notes", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("validated_at", DateTime(timezone=True)),
    Column("approved_at", DateTime(timezone=True)),
    Column("published_at", DateTime(timezone=True)),
)

household_summary_snapshots = Table(
    "household_summary_snapshots",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("household_id", ForeignKey("households.id"), nullable=False),
    Column("model_version_id", ForeignKey("model_versions.id"), nullable=False),
    Column("source_run_id", ForeignKey("training_runs.id")),
    Column("issued_at", DateTime(timezone=True), nullable=False),
    Column("start_at", DateTime(timezone=True), nullable=False),
    Column("end_at", DateTime(timezone=True), nullable=False),
    Column("history", JSON, nullable=False),
    Column("context", JSON, nullable=False),
)

forecast_snapshots = Table(
    "forecast_snapshots",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("household_id", ForeignKey("households.id"), nullable=False),
    Column("model_version_id", ForeignKey("model_versions.id"), nullable=False),
    Column("run_id", ForeignKey("training_runs.id")),
    Column("issued_at", DateTime(timezone=True), nullable=False),
    Column("horizon_start", DateTime(timezone=True), nullable=False),
    Column("horizon_end", DateTime(timezone=True), nullable=False),
    Column("forecast", JSON, nullable=False),
    Column("context", JSON, nullable=False),
)

recommendation_snapshots = Table(
    "recommendation_snapshots",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("household_id", ForeignKey("households.id"), nullable=False),
    Column("model_version_id", ForeignKey("model_versions.id"), nullable=False),
    Column("run_id", ForeignKey("training_runs.id")),
    Column("code", String(64), nullable=False),
    Column("severity", String(32), nullable=False),
    Column("evidence", JSON, nullable=False),
    Column("message", Text, nullable=False),
    Column("valid_until", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

audit_logs = Table(
    "audit_logs",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("actor_user_id", ForeignKey("users.id")),
    Column("action", String(128), nullable=False),
    Column("entity_type", String(128), nullable=False),
    Column("entity_id", String(128), nullable=False),
    Column("details", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
