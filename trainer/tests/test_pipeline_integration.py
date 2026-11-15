from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, func, insert, select

from trainer.core.settings import get_settings
from trainer.data.fixture import generate_fixture_dataset
from trainer.db import schema
from trainer.services.pipeline import PipelineService
from trainer.weather.client import synthesize_weather


def test_fixture_pipeline_imports_trains_and_publishes(tmp_path, monkeypatch):
    db_path = tmp_path / "trainer.db"
    artifacts_dir = tmp_path / "artifacts"

    monkeypatch.setenv("TRAINER_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("APP_PROFILE", "fixture")
    monkeypatch.setenv("FIXTURE_DATA_DIR", str(tmp_path / "fixture"))
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    get_settings.cache_clear()

    try:
        engine = create_engine(f"sqlite:///{db_path}", future=True)
        schema.metadata.create_all(engine)

        source_paths = generate_fixture_dataset(tmp_path / "source", households=2, days=21)
        service = PipelineService()

        def fake_fetch_archive(*, latitude: float, longitude: float, start: str, end: str):
            _ = (latitude, longitude)
            timestamps = pd.date_range(
                start=f"{start}T00:00:00Z",
                end=f"{end}T23:00:00Z",
                freq="1h",
                tz="UTC",
            )
            return synthesize_weather(pd.Series(timestamps))

        monkeypatch.setattr(service.weather_client, "fetch_archive", fake_fetch_archive)

        dataset_version_id = service.import_dataset(
            source_type="fixture",
            dataset_path=source_paths.readings_path.parent,
            profile="fixture",
            sample_households=2,
        )
        service.sync_weather(dataset_version_id)

        run_config = {
            "dataset_version_id": dataset_version_id,
            "household_selector": None,
            "client_sampling_rate": 1.0,
            "rounds": 1,
            "local_epochs": 1,
            "batch_size": 16,
            "learning_rate": 0.001,
            "secure_aggregation_enabled": True,
            "secure_aggregation_degree": 2,
            "secure_aggregation_mask_scale": 4.0,
            "dp_enabled": True,
            "noise_multiplier": 1.1,
            "clipping_norm": 1.0,
        }
        with service.store.session() as session:
            run_id = int(
                session.execute(
                    insert(schema.training_runs)
                    .values(
                        dataset_version_id=dataset_version_id,
                        status="PENDING",
                        config=run_config,
                        metrics_summary={},
                        privacy_summary={},
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                    )
                    .returning(schema.training_runs.c.id)
                ).scalar_one()
            )

        result = service.execute_training_run(run_id)
        model_version_id = result["model_version_id"]
        service.store.mark_model_fixture_approved(model_version_id)
        publish_result = service.publish_model(model_version_id)

        run = service.store.get_training_run(run_id)
        model = service.store.get_model_version(model_version_id)

        assert run is not None
        assert run["status"] == "SUCCEEDED"
        assert run["privacy_summary"]["epsilon"] > 0
        assert run["metrics_summary"]["security_summary"]["secure_aggregation_enabled"] is True
        assert model is not None
        assert model["state"] == "APPROVED"
        assert model["published_at"] is not None
        assert publish_result["household_snapshots"] == 2

        with service.store.session() as session:
            summary_count = session.execute(select(func.count()).select_from(schema.household_summary_snapshots)).scalar_one()
            forecast_count = session.execute(select(func.count()).select_from(schema.forecast_snapshots)).scalar_one()
            recommendation_count = session.execute(select(func.count()).select_from(schema.recommendation_snapshots)).scalar_one()
            forecast_lengths = session.execute(select(schema.forecast_snapshots.c.forecast)).scalars().all()

        assert summary_count == 2
        assert forecast_count == 2
        assert recommendation_count >= 2
        assert all(len(payload) == 24 for payload in forecast_lengths)
        assert (artifacts_dir / "models" / f"run-{run_id}" / "model.pt").exists()
    finally:
        get_settings.cache_clear()


def test_fixture_evaluation_runs_dp_on_and_dp_off_and_writes_reports(tmp_path, monkeypatch):
    db_path = tmp_path / "trainer-eval.db"
    artifacts_dir = tmp_path / "artifacts"
    fixture_dir = tmp_path / "fixture"

    monkeypatch.setenv("TRAINER_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("APP_PROFILE", "fixture")
    monkeypatch.setenv("FIXTURE_DATA_DIR", str(fixture_dir))
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    get_settings.cache_clear()

    try:
        engine = create_engine(f"sqlite:///{db_path}", future=True)
        schema.metadata.create_all(engine)

        service = PipelineService()

        def fake_fetch_archive(*, latitude: float, longitude: float, start: str, end: str):
            _ = (latitude, longitude)
            timestamps = pd.date_range(
                start=f"{start}T00:00:00Z",
                end=f"{end}T23:00:00Z",
                freq="1h",
                tz="UTC",
            )
            return synthesize_weather(pd.Series(timestamps))

        monkeypatch.setattr(service.weather_client, "fetch_archive", fake_fetch_archive)

        result = service.evaluate_fixture_profile(
            households=2,
            run_config_overrides={
                "rounds": 1,
                "local_epochs": 1,
                "batch_size": 16,
                "learning_rate": 0.001,
                "secure_aggregation_enabled": True,
                "secure_aggregation_degree": 2,
                "secure_aggregation_mask_scale": 4.0,
                "noise_multiplier": 1.1,
                "clipping_norm": 1.0,
            },
            publish_dp_on=True,
        )

        experiments = {row["label"]: row for row in result["experiments"]}
        assert set(experiments) == {"dp_on", "dp_off"}
        assert experiments["dp_on"]["privacy_budget"]["epsilon"] > 0
        assert experiments["dp_off"]["privacy_budget"]["epsilon"] == float("inf")
        assert experiments["dp_on"]["secure_aggregation_enabled"] is True
        assert experiments["dp_off"]["secure_aggregation_enabled"] is True
        assert experiments["dp_on"]["round_count"] == 1
        assert experiments["dp_off"]["round_count"] == 1
        assert result["published_experiment_label"] == "dp_on"
        assert Path(result["report_json_path"]).exists()
        assert Path(result["report_markdown_path"]).exists()
        assert "Fixture Evaluation Report" in Path(result["report_markdown_path"]).read_text(encoding="utf-8")

        published_model = service.store.get_model_version(result["published_model_version_id"])
        assert published_model is not None
        assert published_model["state"] == "APPROVED"
        assert published_model["published_at"] is not None
    finally:
        get_settings.cache_clear()


def test_cer_dev_bootstrap_uses_raw_data_dir_and_invites_households(tmp_path, monkeypatch):
    db_path = tmp_path / "trainer-cer.db"
    artifacts_dir = tmp_path / "artifacts"
    raw_data_dir = tmp_path / "raw"

    monkeypatch.setenv("TRAINER_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("APP_PROFILE", "cer-dev")
    monkeypatch.setenv("RAW_DATA_DIR", str(raw_data_dir))
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    get_settings.cache_clear()

    try:
        engine = create_engine(f"sqlite:///{db_path}", future=True)
        schema.metadata.create_all(engine)

        generate_fixture_dataset(raw_data_dir, households=2, days=21)
        service = PipelineService()

        def fake_fetch_archive(*, latitude: float, longitude: float, start: str, end: str):
            _ = (latitude, longitude)
            timestamps = pd.date_range(
                start=f"{start}T00:00:00Z",
                end=f"{end}T23:00:00Z",
                freq="1h",
                tz="UTC",
            )
            return synthesize_weather(pd.Series(timestamps))

        monkeypatch.setattr(service.weather_client, "fetch_archive", fake_fetch_archive)

        result = service.bootstrap_profile(
            source_type="cer",
            profile="cer-dev",
            sample_households=2,
            run_config_overrides={
                "rounds": 1,
                "local_epochs": 1,
                "batch_size": 16,
                "learning_rate": 0.001,
                "secure_aggregation_enabled": True,
                "secure_aggregation_degree": 2,
                "secure_aggregation_mask_scale": 4.0,
                "dp_enabled": True,
                "noise_multiplier": 1.1,
                "clipping_norm": 1.0,
            },
        )

        dataset = service.store.get_dataset_version(result["dataset_version_id"])
        model = service.store.get_model_version(result["model_version_id"])
        households = service.store.household_user_view()

        assert dataset is not None
        assert dataset["profile"] == "cer-dev"
        assert dataset["source_type"] == "cer"
        assert model is not None
        assert model["state"] == "APPROVED"
        assert model["published_at"] is not None
        assert result["publication_result"]["household_snapshots"] == 2
        assert all(row["email"].endswith("@demo.local") for row in households)
        assert all(row["activation_token"] for row in households)
        assert all(not row["is_active"] for row in households)

        with service.store.session() as session:
            forecast_count = session.execute(select(func.count()).select_from(schema.forecast_snapshots)).scalar_one()
            recommendation_count = session.execute(select(func.count()).select_from(schema.recommendation_snapshots)).scalar_one()

        assert forecast_count == 2
        assert recommendation_count >= 2
    finally:
        get_settings.cache_clear()
