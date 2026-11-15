from __future__ import annotations

import secrets
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Iterator

from pwdlib import PasswordHash
from sqlalchemy import and_, create_engine, delete, desc, insert, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from trainer.core.settings import get_settings
from trainer.db import schema

password_hasher = PasswordHash.recommended()


def utcnow() -> datetime:
    return datetime.now(UTC)


class Store:
    def __init__(self) -> None:
        settings = get_settings()
        self.engine: Engine = create_engine(settings.trainer_database_url, future=True, pool_pre_ping=True)
        schema.metadata.create_all(self.engine)
        self._sessionmaker = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._sessionmaker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def fetch_next_job(self) -> dict[str, Any] | None:
        with self.session() as session:
            pending_jobs = session.execute(
                select(schema.jobs).where(schema.jobs.c.status == "PENDING").order_by(schema.jobs.c.created_at)
            ).mappings().all()
            for row in pending_jobs:
                if row["job_type"] == "TRAINING_RUN":
                    payload = row["payload"] or {}
                    run_id = payload.get("run_id")
                    run_status = session.execute(
                        select(schema.training_runs.c.status).where(schema.training_runs.c.id == run_id)
                    ).scalar_one_or_none()
                    if run_status is None or run_status == "CANCELLED":
                        session.execute(
                            update(schema.jobs)
                            .where(schema.jobs.c.id == row["id"])
                            .values(
                                status="CANCELLED",
                                error_message="Run cancelled before execution",
                                updated_at=utcnow(),
                                finished_at=utcnow(),
                            )
                        )
                        continue
                session.execute(
                    update(schema.jobs)
                    .where(schema.jobs.c.id == row["id"])
                    .values(status="RUNNING", started_at=utcnow(), updated_at=utcnow())
                )
                return dict(row)
            return None

    def complete_job(self, job_id: int, result: dict[str, Any]) -> None:
        with self.session() as session:
            session.execute(
                update(schema.jobs)
                .where(schema.jobs.c.id == job_id, schema.jobs.c.status != "CANCELLED")
                .values(status="SUCCEEDED", result=result, updated_at=utcnow(), finished_at=utcnow(), error_message=None)
            )

    def fail_job(self, job_id: int, error: str) -> None:
        with self.session() as session:
            session.execute(
                update(schema.jobs)
                .where(schema.jobs.c.id == job_id, schema.jobs.c.status != "CANCELLED")
                .values(status="FAILED", error_message=error, updated_at=utcnow(), finished_at=utcnow())
            )

    def cancel_job(self, job_id: int, error: str | None = None) -> None:
        with self.session() as session:
            values: dict[str, Any] = {"status": "CANCELLED", "updated_at": utcnow(), "finished_at": utcnow()}
            if error is not None:
                values["error_message"] = error
            session.execute(update(schema.jobs).where(schema.jobs.c.id == job_id).values(**values))

    def cancel_jobs_for_run(self, run_id: int, error: str | None = None) -> int:
        active_statuses = {"PENDING", "RUNNING"}
        updated = 0
        with self.session() as session:
            rows = session.execute(
                select(schema.jobs).where(
                    schema.jobs.c.job_type == "TRAINING_RUN",
                    schema.jobs.c.status.in_(sorted(active_statuses)),
                )
            ).mappings()
            for row in rows:
                payload = row["payload"] or {}
                if payload.get("run_id") != run_id:
                    continue
                values: dict[str, Any] = {"status": "CANCELLED", "updated_at": utcnow(), "finished_at": utcnow()}
                if error is not None:
                    values["error_message"] = error
                session.execute(update(schema.jobs).where(schema.jobs.c.id == row["id"]).values(**values))
                updated += 1
        return updated

    def recover_stale_execution_state(self) -> None:
        with self.session() as session:
            running_jobs = list(
                session.execute(select(schema.jobs).where(schema.jobs.c.status == "RUNNING")).mappings()
            )
            for job in running_jobs:
                payload = job["payload"] or {}
                run_id = payload.get("run_id")
                if job["job_type"] == "TRAINING_RUN" and isinstance(run_id, int):
                    run_status = session.execute(
                        select(schema.training_runs.c.status).where(schema.training_runs.c.id == run_id)
                    ).scalar_one_or_none()
                    if run_status == "CANCELLED":
                        session.execute(
                            update(schema.jobs)
                            .where(schema.jobs.c.id == job["id"])
                            .values(
                                status="CANCELLED",
                                error_message="Recovered cancelled run after trainer restart",
                                updated_at=utcnow(),
                                finished_at=utcnow(),
                            )
                        )
                        continue
                session.execute(
                    update(schema.jobs)
                    .where(schema.jobs.c.id == job["id"])
                    .values(
                        status="PENDING",
                        error_message="Recovered pending job after trainer restart",
                        started_at=None,
                        updated_at=utcnow(),
                    )
                )

            active_run_ids = {
                int((row["payload"] or {}).get("run_id"))
                for row in session.execute(
                    select(schema.jobs).where(schema.jobs.c.job_type == "TRAINING_RUN", schema.jobs.c.status.in_(["PENDING", "RUNNING"]))
                ).mappings()
                if isinstance((row["payload"] or {}).get("run_id"), int)
            }
            running_runs = list(
                session.execute(select(schema.training_runs).where(schema.training_runs.c.status == "RUNNING")).mappings()
            )
            for run in running_runs:
                if int(run["id"]) in active_run_ids:
                    continue
                session.execute(
                    update(schema.training_runs)
                    .where(schema.training_runs.c.id == run["id"])
                    .values(
                        status="FAILED",
                        error_message="Trainer restarted before the run completed",
                        updated_at=utcnow(),
                        finished_at=utcnow(),
                    )
                )

    def get_dataset_version(self, dataset_version_id: int) -> dict[str, Any] | None:
        with self.session() as session:
            row = session.execute(
                select(schema.dataset_versions).where(schema.dataset_versions.c.id == dataset_version_id)
            ).mappings().first()
            return dict(row) if row else None

    def create_dataset_version(self, *, name: str, source_type: str, profile: str, manifest: dict[str, Any], household_count: int) -> int:
        with self.session() as session:
            row = session.execute(
                insert(schema.dataset_versions)
                .values(
                    name=name,
                    source_type=source_type,
                    profile=profile,
                    status="READY",
                    import_manifest=manifest,
                    household_count=household_count,
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )
                .returning(schema.dataset_versions.c.id)
            ).first()
            return int(row[0])

    def update_dataset_manifest(self, dataset_version_id: int, manifest: dict[str, Any]) -> None:
        with self.session() as session:
            session.execute(
                update(schema.dataset_versions)
                .where(schema.dataset_versions.c.id == dataset_version_id)
                .values(import_manifest=manifest, updated_at=utcnow())
            )

    def list_households(self, household_ids: list[int] | None = None) -> list[dict[str, Any]]:
        with self.session() as session:
            stmt = select(schema.households).order_by(schema.households.c.id)
            if household_ids:
                stmt = stmt.where(schema.households.c.id.in_(household_ids))
            return [dict(row) for row in session.execute(stmt).mappings().all()]

    def upsert_households(self, households_payload: list[dict[str, Any]], fixture_passwords: bool = False) -> list[int]:
        created_ids: list[int] = []
        with self.session() as session:
            for index, item in enumerate(households_payload):
                existing = session.execute(
                    select(schema.households).where(schema.households.c.household_code == item["household_code"])
                ).mappings().first()
                if existing:
                    session.execute(
                        update(schema.households)
                        .where(schema.households.c.id == existing["id"])
                        .values(
                            display_name=item["display_name"],
                            region_code=item.get("region_code"),
                            latitude=item.get("latitude"),
                            longitude=item.get("longitude"),
                            metadata_json=item.get("metadata_json", {}),
                        )
                    )
                    household_id = existing["id"]
                else:
                    household_id = int(
                        session.execute(
                            insert(schema.households)
                            .values(
                                household_code=item["household_code"],
                                display_name=item["display_name"],
                                region_code=item.get("region_code"),
                                latitude=item.get("latitude"),
                                longitude=item.get("longitude"),
                                metadata_json=item.get("metadata_json", {}),
                                created_at=utcnow(),
                            )
                            .returning(schema.households.c.id)
                        ).scalar_one()
                    )
                created_ids.append(household_id)
                user_row = session.execute(
                    select(schema.users).where(schema.users.c.household_id == household_id)
                ).mappings().first()
                email = f"{item['household_code'].lower()}@demo.local"
                if user_row is None:
                    is_active = fixture_passwords and index == 0
                    password = "Household123!" if is_active else None
                    session.execute(
                        insert(schema.users).values(
                            email=email,
                            full_name=item["display_name"],
                            role="HOUSEHOLD",
                            password_hash=password_hasher.hash(password) if password else None,
                            activation_token=None if is_active else secrets.token_urlsafe(24),
                            is_active=is_active,
                            household_id=household_id,
                            created_at=utcnow(),
                            activated_at=utcnow() if is_active else None,
                        )
                    )
            existing_admin = session.execute(select(schema.users).where(schema.users.c.email == "admin@demo.local")).mappings().first()
            if existing_admin is None:
                session.execute(
                    insert(schema.users).values(
                        email="admin@demo.local",
                        full_name="Fixture Admin",
                        role="ADMIN",
                        password_hash=password_hasher.hash("Admin123!"),
                        activation_token=None,
                        is_active=True,
                        household_id=None,
                        created_at=utcnow(),
                        activated_at=utcnow(),
                    )
                )
        return created_ids

    def get_latest_ready_dataset(self) -> dict[str, Any] | None:
        with self.session() as session:
            row = session.execute(
                select(schema.dataset_versions)
                .where(schema.dataset_versions.c.status == "READY")
                .order_by(desc(schema.dataset_versions.c.created_at))
                .limit(1)
            ).mappings().first()
            return dict(row) if row else None

    def get_training_run(self, run_id: int) -> dict[str, Any] | None:
        with self.session() as session:
            row = session.execute(select(schema.training_runs).where(schema.training_runs.c.id == run_id)).mappings().first()
            return dict(row) if row else None

    def is_run_cancelled(self, run_id: int) -> bool:
        with self.session() as session:
            status = session.execute(
                select(schema.training_runs.c.status).where(schema.training_runs.c.id == run_id)
            ).scalar_one_or_none()
            return status == "CANCELLED"

    def update_run_status(
        self,
        run_id: int,
        *,
        status: str,
        metrics_summary: dict[str, Any] | None = None,
        privacy_summary: dict[str, Any] | None = None,
        error_message: str | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> None:
        values: dict[str, Any] = {"status": status, "updated_at": utcnow()}
        if metrics_summary is not None:
            values["metrics_summary"] = metrics_summary
        if privacy_summary is not None:
            values["privacy_summary"] = privacy_summary
        if error_message is not None:
            values["error_message"] = error_message
        if started:
            values["started_at"] = utcnow()
        if finished:
            values["finished_at"] = utcnow()
        with self.session() as session:
            session.execute(update(schema.training_runs).where(schema.training_runs.c.id == run_id).values(**values))

    def replace_round_metrics(self, run_id: int, rows: list[dict[str, Any]]) -> None:
        with self.session() as session:
            session.execute(delete(schema.training_round_metrics).where(schema.training_round_metrics.c.run_id == run_id))
            for row in rows:
                session.execute(insert(schema.training_round_metrics).values(run_id=run_id, created_at=utcnow(), **row))

    def append_round_metric(self, run_id: int, row: dict[str, Any]) -> None:
        with self.session() as session:
            session.execute(
                delete(schema.training_round_metrics).where(
                    schema.training_round_metrics.c.run_id == run_id,
                    schema.training_round_metrics.c.round_number == row["round_number"],
                )
            )
            session.execute(insert(schema.training_round_metrics).values(run_id=run_id, created_at=utcnow(), **row))

    def update_run_progress(self, run_id: int, round_row: dict[str, Any], total_rounds: int) -> None:
        with self.session() as session:
            existing = session.execute(
                select(schema.training_runs.c.metrics_summary).where(schema.training_runs.c.id == run_id)
            ).scalar_one_or_none() or {}
            history_summary = dict(existing.get("history_summary") or {})
            history_summary.update(
                {
                    "current_round": int(round_row["round_number"]),
                    "round_count": int(total_rounds),
                    "latest_round_mae": float(round_row["metrics"]["mae"]),
                    "latest_round_rmse": float(round_row["metrics"]["rmse"]),
                    "latest_round_mape": float(round_row["metrics"]["mape"]),
                }
            )
            metrics_summary = dict(existing)
            metrics_summary["history_summary"] = history_summary
            session.execute(
                update(schema.training_runs)
                .where(schema.training_runs.c.id == run_id, schema.training_runs.c.status != "CANCELLED")
                .values(metrics_summary=metrics_summary, updated_at=utcnow())
            )

    def update_run_progress_message(
        self,
        run_id: int,
        *,
        message: str,
        stage: str | None = None,
        current_round: int | None = None,
        round_count: int | None = None,
        completed_clients: int | None = None,
        total_clients: int | None = None,
    ) -> None:
        with self.session() as session:
            existing = session.execute(
                select(schema.training_runs.c.metrics_summary).where(schema.training_runs.c.id == run_id)
            ).scalar_one_or_none() or {}
            progress_summary = dict(existing.get("progress_summary") or {})
            progress_summary["message"] = message
            progress_summary["updated_at"] = utcnow().isoformat()
            if stage is not None:
                progress_summary["stage"] = stage
            if current_round is not None:
                progress_summary["current_round"] = int(current_round)
            if round_count is not None:
                progress_summary["round_count"] = int(round_count)
            if completed_clients is not None:
                progress_summary["completed_clients"] = int(completed_clients)
            if total_clients is not None:
                progress_summary["total_clients"] = int(total_clients)
            metrics_summary = dict(existing)
            metrics_summary["progress_summary"] = progress_summary
            session.execute(
                update(schema.training_runs)
                .where(schema.training_runs.c.id == run_id, schema.training_runs.c.status != "CANCELLED")
                .values(metrics_summary=metrics_summary, updated_at=utcnow())
            )

    def create_model_version(
        self,
        *,
        dataset_version_id: int,
        run_id: int,
        version_label: str,
        metrics: dict[str, Any],
        privacy_budget: dict[str, Any],
        artifact_manifest: dict[str, Any],
    ) -> int:
        with self.session() as session:
            row = session.execute(
                insert(schema.model_versions)
                .values(
                    dataset_version_id=dataset_version_id,
                    run_id=run_id,
                    version_label=version_label,
                    state="DRAFT",
                    metrics=metrics,
                    privacy_budget=privacy_budget,
                    artifact_manifest=artifact_manifest,
                    created_at=utcnow(),
                )
                .returning(schema.model_versions.c.id)
            ).first()
            return int(row[0])

    def get_model_version(self, model_version_id: int) -> dict[str, Any] | None:
        with self.session() as session:
            row = session.execute(
                select(schema.model_versions).where(schema.model_versions.c.id == model_version_id)
            ).mappings().first()
            return dict(row) if row else None

    def mark_model_fixture_approved(self, model_version_id: int) -> None:
        with self.session() as session:
            now = utcnow()
            session.execute(
                update(schema.model_versions)
                .where(schema.model_versions.c.id == model_version_id)
                .values(state="APPROVED", validated_at=now, approved_at=now)
            )

    def mark_model_validated(self, model_version_id: int, notes: str | None = None) -> None:
        with self.session() as session:
            session.execute(
                update(schema.model_versions)
                .where(schema.model_versions.c.id == model_version_id)
                .values(
                    state="VALIDATED",
                    validation_notes=notes,
                    validated_at=utcnow(),
                )
            )

    def mark_model_approved(self, model_version_id: int, notes: str | None = None) -> None:
        with self.session() as session:
            now = utcnow()
            session.execute(
                update(schema.model_versions)
                .where(schema.model_versions.c.id == model_version_id)
                .values(
                    state="APPROVED",
                    validation_notes=notes,
                    validated_at=now,
                    approved_at=now,
                )
            )

    def mark_model_published(self, model_version_id: int) -> None:
        with self.session() as session:
            session.execute(
                update(schema.model_versions).where(schema.model_versions.c.id == model_version_id).values(published_at=utcnow())
            )

    def replace_public_snapshots(
        self,
        *,
        model_version_id: int,
        run_id: int | None,
        histories: list[dict[str, Any]],
        forecasts: list[dict[str, Any]],
        recommendations: list[dict[str, Any]],
    ) -> None:
        with self.session() as session:
            session.execute(delete(schema.household_summary_snapshots).where(schema.household_summary_snapshots.c.model_version_id == model_version_id))
            session.execute(delete(schema.forecast_snapshots).where(schema.forecast_snapshots.c.model_version_id == model_version_id))
            session.execute(delete(schema.recommendation_snapshots).where(schema.recommendation_snapshots.c.model_version_id == model_version_id))
            for row in histories:
                session.execute(insert(schema.household_summary_snapshots).values(model_version_id=model_version_id, source_run_id=run_id, **row))
            for row in forecasts:
                session.execute(insert(schema.forecast_snapshots).values(model_version_id=model_version_id, run_id=run_id, **row))
            for row in recommendations:
                payload = dict(row)
                payload.pop("model_version_id", None)
                session.execute(
                    insert(schema.recommendation_snapshots).values(
                        model_version_id=model_version_id,
                        run_id=run_id,
                        **payload,
                    )
                )

    def audit(self, action: str, entity_type: str, entity_id: str, details: dict[str, Any] | None = None, actor_user_id: int | None = None) -> None:
        with self.session() as session:
            session.execute(
                insert(schema.audit_logs).values(
                    actor_user_id=actor_user_id,
                    action=action,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    details=details or {},
                    created_at=utcnow(),
                )
            )

    def list_pending_runs(self) -> list[dict[str, Any]]:
        with self.session() as session:
            return [
                dict(row)
                for row in session.execute(
                    select(schema.training_runs).where(schema.training_runs.c.status.in_(["PENDING", "RUNNING"])).order_by(schema.training_runs.c.created_at)
                ).mappings()
            ]

    def ready_dataset_exists(self) -> bool:
        with self.session() as session:
            row = session.execute(select(schema.dataset_versions.c.id).where(schema.dataset_versions.c.status == "READY").limit(1)).first()
            return row is not None

    def household_user_view(self) -> list[dict[str, Any]]:
        with self.session() as session:
            rows = session.execute(
                select(
                    schema.households.c.id,
                    schema.households.c.household_code,
                    schema.users.c.email,
                    schema.users.c.activation_token,
                    schema.users.c.is_active,
                ).select_from(schema.households.join(schema.users, and_(schema.users.c.household_id == schema.households.c.id)))
            ).mappings()
            return [dict(row) for row in rows]
