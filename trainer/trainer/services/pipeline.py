from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trainer.core.settings import get_settings
from trainer.data.fixture import generate_fixture_dataset
from trainer.data.preprocess import ImportedDataset, import_cer_dataset
from trainer.db.store import Store
from trainer.fl.pipeline import TrainingCancelled, forecast_next_24h, prepare_client_partitions, run_federated_training
from trainer.services.recommendations import build_recommendations
from trainer.weather.client import WeatherClient, synthesize_weather, write_weather_artifact


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _format_number(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return f"{numeric:.{digits}f}"
        return "inf" if numeric > 0 else "-inf"
    return "n/a"


def _build_fixture_evaluation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fixture Evaluation Report",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Dataset version: {report['dataset_version_id']}",
        f"- Household count: {report['household_count']}",
        f"- Synthetic days: {report.get('synthetic_days', 'n/a')}",
        f"- Weather synced: {'yes' if report['weather_result'].get('weather_artifacts') else 'no'}",
        "",
        "| Experiment | DP | Secure Agg | MAE | RMSE | MAPE | Centralized MAE | Accuracy Gap | Epsilon | Bytes | Duration (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for experiment in report["experiments"]:
        lines.append(
            "| {label} | {dp_enabled} | {secure_aggregation_enabled} | {mae} | {rmse} | {mape} | {baseline_mae} | {gap} | {epsilon} | {bytes_total} | {duration_total} |".format(
                label=experiment["label"],
                dp_enabled="on" if experiment["dp_enabled"] else "off",
                secure_aggregation_enabled="on" if experiment.get("secure_aggregation_enabled") else "off",
                mae=_format_number(experiment["fl_metrics"].get("mae")),
                rmse=_format_number(experiment["fl_metrics"].get("rmse")),
                mape=_format_number(experiment["fl_metrics"].get("mape")),
                baseline_mae=_format_number(experiment["centralized_metrics"].get("mae")),
                gap=_format_number(experiment["accuracy_gap_mae"]),
                epsilon=_format_number(experiment["privacy_budget"].get("epsilon")),
                bytes_total=str(experiment["bytes_transferred_total"]),
                duration_total=_format_number(experiment["duration_seconds_total"]),
            )
        )
    if report.get("published_model_version_id") is not None:
        lines.extend(
            [
                "",
                f"Published model version: {report['published_model_version_id']}",
                f"Published experiment: {report['published_experiment_label']}",
            ]
        )
    return "\n".join(lines) + "\n"


class PipelineService:
    def __init__(self, store: Store | None = None) -> None:
        self.settings = get_settings()
        self.store = store or Store()
        self.weather_client = WeatherClient()

    @staticmethod
    def _log(message: str) -> None:
        print(f"[trainer] {message}", flush=True)

    def _update_run_progress_message(
        self,
        run_id: int,
        message: str,
        *,
        stage: str | None = None,
        current_round: int | None = None,
        round_count: int | None = None,
        completed_clients: int | None = None,
        total_clients: int | None = None,
    ) -> None:
        self.store.update_run_progress_message(
            run_id,
            message=message,
            stage=stage,
            current_round=current_round,
            round_count=round_count,
            completed_clients=completed_clients,
            total_clients=total_clients,
        )

    def ensure_fixture_bootstrap(self, households: int | None = None, days: int | None = None) -> None:
        if self.settings.app_profile != "fixture" or self.store.ready_dataset_exists():
            return
        households = households or self.settings.synthetic_dataset_households
        days = days or self.settings.synthetic_dataset_days
        self._log(f"Bootstrapping fixture profile with {households} households and {days} synthetic days")
        generate_fixture_dataset(
            self.settings.fixture_data_dir,
            households=households,
            days=days,
            seed=self.settings.synthetic_dataset_seed,
        )
        self.bootstrap_profile(
            source_type="fixture",
            profile="fixture",
            dataset_path=self.settings.fixture_data_dir,
            sample_households=households,
            synthetic_days=days,
        )

    def _default_run_config(self, dataset_version_id: int) -> dict[str, Any]:
        return {
            "dataset_version_id": dataset_version_id,
            "household_selector": None,
            "client_sampling_rate": 1.0,
            "rounds": 5,
            "local_epochs": 2,
            "batch_size": 64,
            "learning_rate": 0.001,
            "secure_aggregation_enabled": True,
            "secure_aggregation_degree": 2,
            "secure_aggregation_mask_scale": 4.0,
            "dp_enabled": True,
            "noise_multiplier": 1.1,
            "clipping_norm": 1.0,
        }

    def _dataset_households(self, dataset: dict[str, Any], household_selector: list[int] | None = None) -> list[dict[str, Any]]:
        households = self.store.list_households(household_selector)
        household_codes = set((dataset.get("import_manifest") or {}).get("household_codes") or [])
        if household_codes:
            households = [row for row in households if row["household_code"] in household_codes]
        return households

    def create_implicit_run(self, dataset_version_id: int, config_overrides: dict[str, Any] | None = None) -> int:
        from sqlalchemy import insert
        from trainer.db import schema

        config = self._default_run_config(dataset_version_id)
        if config_overrides:
            config.update(config_overrides)
        with self.store.session() as session:
            run_id = int(
                session.execute(
                    insert(schema.training_runs).values(
                        dataset_version_id=dataset_version_id,
                        status="PENDING",
                        config=config,
                        metrics_summary={},
                        privacy_summary={},
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                    ).returning(schema.training_runs.c.id)
                ).scalar_one()
            )
        return run_id

    def _resolve_dataset_source_path(
        self,
        source_type: str,
        dataset_path: Path | str | None,
        *,
        sample_households: int | None,
        synthetic_days: int | None,
        synthetic_seed: int | None,
        reset_existing: bool,
    ) -> Path:
        if dataset_path is not None:
            return Path(dataset_path)
        if source_type == "fixture":
            self._log(
                f"Generating synthetic fixture dataset: households={sample_households or self.settings.synthetic_dataset_households}, "
                f"days={synthetic_days or self.settings.synthetic_dataset_days}"
            )
            fixture_paths = generate_fixture_dataset(
                self.settings.fixture_data_dir,
                households=sample_households or self.settings.synthetic_dataset_households,
                days=synthetic_days or self.settings.synthetic_dataset_days,
                seed=synthetic_seed or self.settings.synthetic_dataset_seed,
                overwrite=reset_existing,
            )
            return fixture_paths.readings_path.parent
        if source_type == "smart-meters":
            return self.settings.smart_meter_data_dir
        return self.settings.raw_data_dir

    def import_dataset(
        self,
        *,
        source_type: str,
        dataset_path: Path | str | None,
        profile: str,
        sample_households: int | None,
        synthetic_days: int | None = None,
        synthetic_seed: int | None = None,
        reset_existing: bool = False,
    ) -> int:
        source_path = self._resolve_dataset_source_path(
            source_type,
            dataset_path,
            sample_households=sample_households,
            synthetic_days=synthetic_days,
            synthetic_seed=synthetic_seed,
            reset_existing=reset_existing,
        )
        self._log(f"Importing dataset from {source_path}")

        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        output_dir = self.settings.artifacts_dir / "datasets" / f"{source_type}-{timestamp}"
        imported: ImportedDataset = import_cer_dataset(source_path, output_dir, sample_households=sample_households)
        self.store.upsert_households(imported.households, fixture_passwords=(profile == "fixture"))
        household_rows = self.store.household_user_view()
        manifest = imported.manifest | {
            "users": household_rows,
            "synthetic_generation": {
                "days": synthetic_days or self.settings.synthetic_dataset_days,
                "seed": synthetic_seed or self.settings.synthetic_dataset_seed,
            }
            if source_type == "fixture"
            else {},
        }
        dataset_version_id = self.store.create_dataset_version(
            name=f"{source_type}-{timestamp}",
            source_type=source_type,
            profile=profile,
            manifest=manifest,
            household_count=len(imported.households),
        )
        self._log(f"Dataset version {dataset_version_id} ready with {len(imported.households)} households")
        return dataset_version_id

    def bootstrap_profile(
        self,
        *,
        source_type: str,
        profile: str,
        dataset_path: Path | str | None = None,
        sample_households: int | None = None,
        synthetic_days: int | None = None,
        synthetic_seed: int | None = None,
        weather_household_limit: int | None = None,
        run_config_overrides: dict[str, Any] | None = None,
        auto_publish: bool = True,
        reset_existing: bool = False,
    ) -> dict[str, Any]:
        dataset_version_id = self.import_dataset(
            source_type=source_type,
            dataset_path=dataset_path,
            profile=profile,
            sample_households=sample_households,
            synthetic_days=synthetic_days,
            synthetic_seed=synthetic_seed,
            reset_existing=reset_existing,
        )
        weather_result = self.sync_weather(dataset_version_id, household_limit=weather_household_limit)
        run_id = self.create_implicit_run(dataset_version_id, config_overrides=run_config_overrides)
        training_result = self.execute_training_run(run_id)
        model_version_id = training_result["model_version_id"]

        approval_note = f"Bootstrapped via trainer CLI for {profile}"
        self.store.mark_model_validated(model_version_id, notes=approval_note)
        self.store.mark_model_approved(model_version_id, notes=approval_note)

        publication_result: dict[str, Any] = {}
        if auto_publish:
            publication_result = self.publish_model(model_version_id)

        return {
            "dataset_version_id": dataset_version_id,
            "weather_result": weather_result,
            "run_id": run_id,
            "model_version_id": model_version_id,
            "publication_result": publication_result,
        }

    def sync_weather(self, dataset_version_id: int, household_limit: int | None = None) -> dict[str, Any]:
        dataset = self.store.get_dataset_version(dataset_version_id)
        if dataset is None:
            raise ValueError(f"Dataset version {dataset_version_id} not found")
        self._log(f"Syncing weather for dataset {dataset_version_id}")
        artifact_root = Path(dataset["import_manifest"]["artifact_root"])
        households = self._dataset_households(dataset)
        if household_limit:
            households = households[:household_limit]
        start_date = pd.Timestamp(dataset["import_manifest"]["start_at"]).date().isoformat()
        end_date = pd.Timestamp(dataset["import_manifest"]["end_at"]).date().isoformat()
        written: dict[str, str] = {}
        for household in households:
            region_key = household.get("region_code") or "IRELAND"
            try:
                weather = self.weather_client.fetch_archive(
                    latitude=household.get("latitude") or 53.3498,
                    longitude=household.get("longitude") or -6.2603,
                    start=start_date,
                    end=end_date,
                )
                weather["source"] = "open-meteo"
            except Exception:
                frame = pd.read_csv(artifact_root / "households" / f"{household['household_code']}.csv")
                weather = synthesize_weather(frame["timestamp"])
            written[region_key] = str(write_weather_artifact(artifact_root, region_key, weather))

        manifest = dict(dataset["import_manifest"])
        manifest["weather_ready"] = True
        manifest["weather_artifacts"] = written
        manifest["weather_synced_at"] = datetime.now(UTC).isoformat()
        self.store.update_dataset_manifest(dataset_version_id, manifest)
        self._log(f"Weather sync complete for dataset {dataset_version_id}")
        return {"weather_artifacts": written, "dataset_version_id": dataset_version_id}

    def execute_training_run(self, run_id: int) -> dict[str, Any]:
        run = self.store.get_training_run(run_id)
        if run is None:
            raise ValueError(f"Training run {run_id} not found")
        if run["status"] == "CANCELLED":
            raise TrainingCancelled(f"Run {run_id} was cancelled before execution")
        dataset = self.store.get_dataset_version(run["dataset_version_id"])
        if dataset is None:
            raise ValueError(f"Dataset version {run['dataset_version_id']} not found")
        self._log(f"Starting training run {run_id} on dataset {run['dataset_version_id']}")
        self.store.update_run_status(run_id, status="RUNNING", started=True)
        self._update_run_progress_message(run_id, "Preparing client partitions", stage="preparing")
        artifact_root = Path(dataset["import_manifest"]["artifact_root"])
        households = self._dataset_households(dataset, run["config"].get("household_selector"))
        partitions = prepare_client_partitions(
            artifact_root=artifact_root,
            household_rows=households,
            selected_household_ids=run["config"].get("household_selector"),
        )
        self._update_run_progress_message(
            run_id,
            f"Prepared {len(partitions)} client partitions; starting federated training",
            stage="training",
            current_round=0,
            round_count=int(run["config"].get("rounds", 0)),
            completed_clients=0,
            total_clients=len(partitions),
        )
        model_output_dir = self.settings.artifacts_dir / "models" / f"run-{run_id}"
        rounds = int(run["config"].get("rounds", 0))
        try:
            result = run_federated_training(
                partitions=partitions,
                config=run["config"],
                output_dir=model_output_dir,
                round_callback=lambda row: (
                    self.store.append_round_metric(run_id, row),
                    self.store.update_run_progress(run_id, row, rounds),
                ),
                should_abort=lambda: self.store.is_run_cancelled(run_id),
                progress_callback=lambda message: self._handle_progress_message(run_id, message, rounds),
            )
        except TrainingCancelled as exc:
            self.store.update_run_status(run_id, status="CANCELLED", error_message=str(exc), finished=True)
            self._update_run_progress_message(run_id, str(exc), stage="cancelled")
            self._log(f"Training run {run_id} cancelled")
            raise
        self.store.replace_round_metrics(run_id, result["round_metrics"])
        self.store.update_run_status(
            run_id,
            status="SUCCEEDED",
            metrics_summary=result["run_summary"],
            privacy_summary=result["privacy_budget"],
            finished=True,
        )
        self._update_run_progress_message(run_id, "Training completed successfully", stage="completed", current_round=rounds, round_count=rounds)
        model_version_id = self.store.create_model_version(
            dataset_version_id=run["dataset_version_id"],
            run_id=run_id,
            version_label=f"model-run-{run_id}",
            metrics=result["run_summary"],
            privacy_budget=result["privacy_budget"],
            artifact_manifest=result["artifact_manifest"] | {"dataset_artifact_root": str(artifact_root)},
        )
        self._log(f"Training run {run_id} completed successfully")
        return {
            "run_id": run_id,
            "model_version_id": model_version_id,
            "run_summary": result["run_summary"],
            "privacy_budget": result["privacy_budget"],
            "round_metrics": result["round_metrics"],
        }

    def _handle_progress_message(self, run_id: int, message: str, total_rounds: int) -> None:
        self._log(f"Run {run_id}: {message}")
        current_round: int | None = None
        round_count: int | None = total_rounds if total_rounds > 0 else None
        completed_clients: int | None = None
        total_clients: int | None = None
        stage = "training"

        if message.startswith("Round "):
            try:
                prefix, details = message.split(":", maxsplit=1)
                round_text = prefix.removeprefix("Round ").strip()
                current_round_str, total_rounds_str = round_text.split("/", maxsplit=1)
                current_round = int(current_round_str)
                round_count = int(total_rounds_str)
                details = details.strip()
                if details.startswith("training "):
                    stage = "training"
                    count_text = details.removeprefix("training ").removesuffix(" clients")
                    total_clients = int(count_text)
                    completed_clients = 0
                elif details.startswith("completed "):
                    stage = "training"
                    count_text = details.removeprefix("completed ")
                    completed_str, total_clients_str = count_text.split("/", maxsplit=1)
                    completed_clients = int(completed_str)
                    total_clients = int(total_clients_str.split(" ", maxsplit=1)[0])
                elif details.startswith("mae="):
                    stage = "evaluating"
            except (ValueError, IndexError):
                pass
        elif "centralized baseline" in message.lower():
            stage = "baseline"
        elif "holdout test" in message.lower():
            stage = "evaluating"
        elif "artifacts" in message.lower():
            stage = "finalizing"

        self._update_run_progress_message(
            run_id,
            message,
            stage=stage,
            current_round=current_round,
            round_count=round_count,
            completed_clients=completed_clients,
            total_clients=total_clients,
        )

    def evaluate_fixture_profile(
        self,
        *,
        households: int | None = None,
        synthetic_days: int | None = None,
        synthetic_seed: int | None = None,
        weather_household_limit: int | None = None,
        run_config_overrides: dict[str, Any] | None = None,
        publish_dp_on: bool = False,
    ) -> dict[str, Any]:
        households = households or self.settings.synthetic_dataset_households
        synthetic_days = synthetic_days or self.settings.synthetic_dataset_days
        synthetic_seed = synthetic_seed or self.settings.synthetic_dataset_seed
        generate_fixture_dataset(
            self.settings.fixture_data_dir,
            households=households,
            days=synthetic_days,
            seed=synthetic_seed,
            overwrite=True,
        )
        dataset_version_id = self.import_dataset(
            source_type="fixture",
            dataset_path=self.settings.fixture_data_dir,
            profile="fixture",
            sample_households=households,
            synthetic_days=synthetic_days,
            synthetic_seed=synthetic_seed,
        )
        weather_result = self.sync_weather(dataset_version_id, household_limit=weather_household_limit)

        experiment_specs = [
            {
                "label": "dp_on",
                "config": {
                    "rounds": 3,
                    "local_epochs": 1,
                    "batch_size": 16,
                    "learning_rate": 0.001,
                    "dp_enabled": True,
                    "noise_multiplier": 1.1,
                    "clipping_norm": 1.0,
                },
            },
            {
                "label": "dp_off",
                "config": {
                    "rounds": 3,
                    "local_epochs": 1,
                    "batch_size": 16,
                    "learning_rate": 0.001,
                    "dp_enabled": False,
                    "noise_multiplier": 0.0,
                    "clipping_norm": 1.0,
                },
            },
        ]
        if run_config_overrides:
            for experiment in experiment_specs:
                experiment["config"].update(run_config_overrides)

        experiments: list[dict[str, Any]] = []
        published_model_version_id: int | None = None
        published_experiment_label: str | None = None

        for experiment in experiment_specs:
            run_id = self.create_implicit_run(dataset_version_id, config_overrides=experiment["config"])
            training_result = self.execute_training_run(run_id)
            round_metrics = training_result["round_metrics"]
            bytes_transferred_total = int(sum(int(row["bytes_transferred"]) for row in round_metrics))
            duration_seconds_total = float(sum(float(row["duration_seconds"]) for row in round_metrics))

            model_version_id = int(training_result["model_version_id"])
            if publish_dp_on and experiment["label"] == "dp_on":
                approval_note = "Approved and published from fixture evaluation"
                self.store.mark_model_validated(model_version_id, notes=approval_note)
                self.store.mark_model_approved(model_version_id, notes=approval_note)
                self.publish_model(model_version_id)
                published_model_version_id = model_version_id
                published_experiment_label = experiment["label"]

            experiments.append(
                {
                "label": experiment["label"],
                "dp_enabled": bool(experiment["config"]["dp_enabled"]),
                "secure_aggregation_enabled": bool(experiment["config"].get("secure_aggregation_enabled", True)),
                "run_id": run_id,
                "model_version_id": model_version_id,
                "fl_metrics": training_result["run_summary"]["fl_metrics"],
                "centralized_metrics": training_result["run_summary"]["centralized_metrics"],
                "accuracy_gap_mae": training_result["run_summary"]["accuracy_gap_mae"],
                    "privacy_budget": training_result["privacy_budget"],
                    "round_count": len(round_metrics),
                    "bytes_transferred_total": bytes_transferred_total,
                    "duration_seconds_total": duration_seconds_total,
                }
            )

        generated_at = datetime.now(UTC).isoformat()
        report = {
            "generated_at": generated_at,
            "dataset_version_id": dataset_version_id,
            "household_count": households,
            "synthetic_days": synthetic_days,
            "weather_result": weather_result,
            "experiments": experiments,
            "published_model_version_id": published_model_version_id,
            "published_experiment_label": published_experiment_label,
        }

        reports_dir = self.settings.artifacts_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        json_path = reports_dir / f"fixture-evaluation-{stamp}.json"
        markdown_path = reports_dir / f"fixture-evaluation-{stamp}.md"
        json_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
        markdown_path.write_text(_build_fixture_evaluation_markdown(report), encoding="utf-8")

        return report | {"report_json_path": str(json_path), "report_markdown_path": str(markdown_path)}

    def publish_model(self, model_version_id: int) -> dict[str, Any]:
        model = self.store.get_model_version(model_version_id)
        if model is None:
            raise ValueError(f"Model version {model_version_id} not found")
        artifact_manifest = model["artifact_manifest"]
        dataset = self.store.get_dataset_version(model["dataset_version_id"])
        if dataset is None:
            raise ValueError("Dataset version not found")
        artifact_root = Path(artifact_manifest["dataset_artifact_root"])
        households = self._dataset_households(dataset)
        partitions = prepare_client_partitions(artifact_root=artifact_root, household_rows=households)
        partition_map = {partition.household_id: partition for partition in partitions}
        histories: list[dict[str, Any]] = []
        forecasts: list[dict[str, Any]] = []
        recommendations: list[dict[str, Any]] = []
        for household in households:
            partition = partition_map.get(household["id"])
            if partition is None:
                continue
            forecast_frame = forecast_next_24h(Path(artifact_manifest["model_path"]), partition)
            history_frame = partition.full_frame.tail(24 * 30).copy()
            context = {
                "weather_source": "open-meteo" if dataset["import_manifest"].get("weather_ready") else "synthetic",
                "region_code": household.get("region_code"),
                "model_version_id": model_version_id,
                "model_state": model["state"],
                "updated_at": datetime.now(UTC).isoformat(),
                "peak_window": "17:00-20:00",
            }
            issued_at = datetime.now(UTC)
            histories.append(
                {
                    "household_id": household["id"],
                    "issued_at": issued_at,
                    "start_at": pd.to_datetime(history_frame["timestamp"], utc=True).min().to_pydatetime(),
                    "end_at": pd.to_datetime(history_frame["timestamp"], utc=True).max().to_pydatetime(),
                    "history": history_frame[["timestamp", "consumption_kwh"]].assign(timestamp=lambda x: pd.to_datetime(x["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_dict(orient="records"),
                    "context": context,
                }
            )
            forecasts.append(
                {
                    "household_id": household["id"],
                    "issued_at": issued_at,
                    "horizon_start": forecast_frame["timestamp"].min().to_pydatetime(),
                    "horizon_end": forecast_frame["timestamp"].max().to_pydatetime(),
                    "forecast": forecast_frame.assign(timestamp=lambda x: x["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_dict(orient="records"),
                    "context": context | {"next_day_total_kwh": round(float(forecast_frame["prediction_kwh"].sum()), 3)},
                }
            )
            recommendations.extend(
                build_recommendations(
                    household_id=household["id"],
                    forecast_frame=forecast_frame,
                    history_frame=history_frame,
                    model_version_id=model_version_id,
                )
            )
        self.store.replace_public_snapshots(
            model_version_id=model_version_id,
            run_id=model.get("run_id"),
            histories=histories,
            forecasts=forecasts,
            recommendations=recommendations,
        )
        self.store.mark_model_published(model_version_id)
        return {"household_snapshots": len(histories), "recommendations": len(recommendations)}

    def process_job(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job["payload"] or {}
        if job["job_type"] == "DATASET_IMPORT":
            result = {
                "dataset_version_id": self.import_dataset(
                    source_type=payload.get("source_type", "fixture"),
                    dataset_path=payload.get("dataset_path"),
                    profile=payload.get("profile", self.settings.app_profile),
                    sample_households=payload.get("sample_households"),
                    synthetic_days=payload.get("synthetic_days"),
                    synthetic_seed=payload.get("synthetic_seed"),
                    reset_existing=bool(payload.get("reset_existing", False)),
                )
            }
        elif job["job_type"] == "WEATHER_SYNC":
            result = self.sync_weather(payload["dataset_version_id"], household_limit=payload.get("household_limit"))
        elif job["job_type"] == "TRAINING_RUN":
            result = self.execute_training_run(payload["run_id"])
        elif job["job_type"] == "PUBLISH_MODEL":
            result = self.publish_model(payload["model_version_id"])
        else:
            raise ValueError(f"Unsupported job type {job['job_type']}")
        return result

    def worker_loop(self) -> None:
        self.store.recover_stale_execution_state()
        self._log("Worker loop started")
        self.ensure_fixture_bootstrap()
        while True:
            job = self.store.fetch_next_job()
            if job is None:
                time.sleep(self.settings.trainer_poll_interval_seconds)
                continue
            try:
                self._log(f"Processing job #{job['id']} ({job['job_type']})")
                result = self.process_job(job)
                self.store.complete_job(job["id"], result)
                self._log(f"Completed job #{job['id']} ({job['job_type']})")
            except TrainingCancelled as exc:
                self.store.cancel_job(job["id"], str(exc))
                self._log(f"Cancelled job #{job['id']} ({job['job_type']}): {exc}")
            except Exception as exc:  # pragma: no cover - defensive logging path
                self.store.fail_job(job["id"], str(exc))
                self._log(f"Failed job #{job['id']} ({job['job_type']}): {exc}")
