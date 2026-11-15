"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-03-15 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    user_role = postgresql.ENUM("ADMIN", "HOUSEHOLD", name="user_role", create_type=False)
    job_type = postgresql.ENUM("DATASET_IMPORT", "WEATHER_SYNC", "TRAINING_RUN", "PUBLISH_MODEL", name="job_type", create_type=False)
    job_status = postgresql.ENUM("PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED", name="job_status", create_type=False)
    dataset_status = postgresql.ENUM("PENDING", "READY", "FAILED", name="dataset_status", create_type=False)
    run_status = postgresql.ENUM("PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED", name="run_status", create_type=False)
    model_state = postgresql.ENUM("DRAFT", "VALIDATED", "APPROVED", name="model_state", create_type=False)

    bind = op.get_bind()
    user_role.create(bind, checkfirst=True)
    job_type.create(bind, checkfirst=True)
    job_status.create(bind, checkfirst=True)
    dataset_status.create(bind, checkfirst=True)
    run_status.create(bind, checkfirst=True)
    model_state.create(bind, checkfirst=True)

    op.create_table(
        "households",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("household_code", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("region_code", sa.String(length=64), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.UniqueConstraint("household_code"),
    )
    op.create_index("ix_households_household_code", "households", ["household_code"], unique=True)

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=False),
        sa.Column("role", user_role, nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=True),
        sa.Column("activation_token", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("household_id", sa.Integer(), sa.ForeignKey("households.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("email"),
        sa.UniqueConstraint("activation_token"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "dataset_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("profile", sa.String(length=64), nullable=False),
        sa.Column("status", dataset_status, nullable=False),
        sa.Column("import_manifest", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("household_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("job_type", job_type, nullable=False),
        sa.Column("status", job_status, nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("result", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "training_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset_version_id", sa.Integer(), sa.ForeignKey("dataset_versions.id"), nullable=False),
        sa.Column("status", run_status, nullable=False),
        sa.Column("config", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("metrics_summary", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("privacy_summary", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "training_round_metrics",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("training_runs.id"), nullable=False),
        sa.Column("round_number", sa.Integer(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("bytes_transferred", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("duration_seconds", sa.Float(), nullable=False, server_default="0"),
        sa.Column("participant_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset_version_id", sa.Integer(), sa.ForeignKey("dataset_versions.id"), nullable=False),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("training_runs.id"), nullable=True),
        sa.Column("version_label", sa.String(length=128), nullable=False),
        sa.Column("state", model_state, nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("privacy_budget", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("artifact_manifest", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("validation_notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("validated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("version_label"),
    )

    op.create_table(
        "household_summary_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("household_id", sa.Integer(), sa.ForeignKey("households.id"), nullable=False),
        sa.Column("model_version_id", sa.Integer(), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("source_run_id", sa.Integer(), sa.ForeignKey("training_runs.id"), nullable=True),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("start_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("history", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("context", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )

    op.create_table(
        "forecast_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("household_id", sa.Integer(), sa.ForeignKey("households.id"), nullable=False),
        sa.Column("model_version_id", sa.Integer(), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("training_runs.id"), nullable=True),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("horizon_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("horizon_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("forecast", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("context", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )

    op.create_table(
        "recommendation_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("household_id", sa.Integer(), sa.ForeignKey("households.id"), nullable=False),
        sa.Column("model_version_id", sa.Integer(), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("training_runs.id"), nullable=True),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("actor_user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("action", sa.String(length=128), nullable=False),
        sa.Column("entity_type", sa.String(length=128), nullable=False),
        sa.Column("entity_id", sa.String(length=128), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("recommendation_snapshots")
    op.drop_table("forecast_snapshots")
    op.drop_table("household_summary_snapshots")
    op.drop_table("model_versions")
    op.drop_table("training_round_metrics")
    op.drop_table("training_runs")
    op.drop_table("jobs")
    op.drop_table("dataset_versions")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
    op.drop_index("ix_households_household_code", table_name="households")
    op.drop_table("households")

    bind = op.get_bind()
    sa.Enum(name="model_state").drop(bind, checkfirst=True)
    sa.Enum(name="run_status").drop(bind, checkfirst=True)
    sa.Enum(name="dataset_status").drop(bind, checkfirst=True)
    sa.Enum(name="job_status").drop(bind, checkfirst=True)
    sa.Enum(name="job_type").drop(bind, checkfirst=True)
    sa.Enum(name="user_role").drop(bind, checkfirst=True)
