from __future__ import annotations

from datetime import UTC, datetime
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest
import torch

from trainer.data.fixture import generate_fixture_dataset
from trainer.data.preprocess import add_time_features, create_windows, import_cer_dataset
from trainer.fl.accounting import compute_privacy_budget
from trainer.fl.model import state_dict_to_ndarrays
from trainer.fl.pipeline import ClientPartition, FEATURE_COLUMNS
from trainer.fl.secure_agg import average_client_updates, secure_aggregate_weighted_updates
from trainer.services.recommendations import build_recommendations


def _write_smart_meter_dataset(base_dir, *, households: int = 2, days: int = 21):
    dataset_dir = base_dir / "smart-meters"
    halfhourly_dir = dataset_dir / "halfhourly_dataset" / "halfhourly_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    halfhourly_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    reading_rows = []
    half_hourly = pd.date_range("2012-10-12T00:00:00Z", periods=days * 48, freq="30min", tz="UTC")
    for index in range(households):
        household_id = f"MAC{index + 1:06d}"
        metadata_rows.append(
            {
                "LCLid": household_id,
                "stdorToU": "Std" if index % 2 == 0 else "ToU",
                "Acorn": f"ACORN-{chr(65 + index)}",
                "Acorn_grouped": "Affluent" if index % 2 == 0 else "Comfortable",
                "file": "block_0",
            }
        )
        base_load = 0.18 + (index * 0.04)
        for timestamp in half_hourly:
            reading_rows.append(
                {
                    "LCLid": household_id,
                    "tstp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "energy(kWh/hh)": f" {base_load:.3f} ",
                }
            )

    pd.DataFrame(metadata_rows).to_csv(dataset_dir / "informations_households.csv", index=False)
    pd.DataFrame(reading_rows).to_csv(halfhourly_dir / "block_0.csv", index=False)
    return dataset_dir


def test_fixture_generation_and_import(tmp_path):
    fixture_paths = generate_fixture_dataset(tmp_path, households=3, days=30)
    imported = import_cer_dataset(fixture_paths.readings_path.parent, tmp_path / "artifacts", sample_households=2)
    assert len(imported.households) == 2
    assert imported.manifest["window_input_hours"] == 168
    assert (tmp_path / "artifacts" / "households").exists()


def test_fixture_generation_overwrites_when_requested_shape_changes(tmp_path):
    initial = generate_fixture_dataset(tmp_path, households=2, days=30, seed=11)
    first_readings = pd.read_csv(initial.readings_path)
    updated = generate_fixture_dataset(tmp_path, households=4, days=45, seed=12, overwrite=True)
    second_readings = pd.read_csv(updated.readings_path)

    assert initial.readings_path == updated.readings_path
    assert first_readings["household_id"].nunique() == 2
    assert second_readings["household_id"].nunique() == 4
    assert len(second_readings) == 4 * 45 * 48


def test_cer_trial_zip_import_decodes_interval_codes(tmp_path):
    dataset_dir = tmp_path / "cer"
    dataset_dir.mkdir()
    allocation = pd.DataFrame(
        {
            "Meter_ID": ["1001", "2001"],
            "Code": ["Residential", "SME"],
            "Residential Tariff": ["A", "B"],
        }
    )
    allocation.to_csv(dataset_dir / "allocation_data.csv", index=False)

    periods = 21 * 48
    half_hourly = pd.date_range("2009-01-01T00:00:00Z", periods=periods, freq="30min", tz="UTC")
    rows: list[str] = []
    for index, timestamp in enumerate(half_hourly):
        day_code = (index // 48) + 1
        slot_code = (index % 48) + 1
        interval_code = f"{day_code:03d}{slot_code:02d}"
        rows.append(f"1001 {interval_code} 0.5")
        rows.append(f"2001 {interval_code} 0.7")

    with ZipFile(dataset_dir / "File1.txt.zip", mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("File1.txt", "\n".join(rows))

    imported = import_cer_dataset(dataset_dir, tmp_path / "artifacts", sample_households=1)

    assert len(imported.households) == 1
    assert imported.households[0]["household_code"] == "1001"
    assert imported.manifest["source_format"] == "cer-smart-meter-trial"
    assert imported.manifest["allocation_metadata_loaded"] is True
    series = imported.household_series["1001"]
    assert len(series) == 21 * 24
    assert pd.Timestamp(series["timestamp"].min()) == pd.Timestamp("2009-01-01T00:00:00Z")
    assert pd.Timestamp(series["timestamp"].max()) == pd.Timestamp("2009-01-21T23:00:00Z")
    assert imported.households[0]["metadata_json"]["residential_tariff"] == "A"


def test_cer_import_surfaces_metadata_only_download_error(tmp_path):
    dataset_dir = tmp_path / "issda"
    dataset_dir.mkdir()
    (dataset_dir / "documentation").mkdir()
    (dataset_dir / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as exc_info:
        import_cer_dataset(dataset_dir, tmp_path / "artifacts")

    assert "metadata-only/public-documentation download" in str(exc_info.value)
    assert "download-cer" in str(exc_info.value)


def test_smart_meter_import_detects_half_hourly_blocks_and_metadata(tmp_path):
    dataset_dir = _write_smart_meter_dataset(tmp_path, households=2, days=21)

    imported = import_cer_dataset(dataset_dir, tmp_path / "artifacts", sample_households=1)

    assert len(imported.households) == 1
    assert imported.households[0]["household_code"] == "MAC000001"
    assert imported.households[0]["region_code"] == "LONDON"
    assert imported.households[0]["metadata_json"]["tariff_type"] == "Std"
    assert imported.households[0]["metadata_json"]["acorn"] == "ACORN-A"
    assert imported.households[0]["metadata_json"]["holiday_country"] == "GB"
    assert imported.manifest["source_format"] == "uk-smart-meter-halfhourly"
    assert imported.manifest["household_metadata_loaded"] is True
    series = imported.household_series["MAC000001"]
    assert len(series) == 21 * 24
    assert pd.Timestamp(series["timestamp"].min()) == pd.Timestamp("2012-10-12T00:00:00Z")
    assert pd.Timestamp(series["timestamp"].max()) == pd.Timestamp("2012-11-01T23:00:00Z")


def test_time_features_and_window_creation():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=240, freq="1h", tz="UTC"),
            "consumption_kwh": [0.5] * 240,
            "temperature_2m": [10.0] * 240,
            "apparent_temperature": [9.0] * 240,
            "precipitation": [0.0] * 240,
        }
    )
    enriched = add_time_features(frame)
    X, y, _ = create_windows(
        enriched,
        feature_columns=[
            "consumption_kwh",
            "temperature_2m",
            "apparent_temperature",
            "precipitation",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "is_weekend",
            "is_holiday",
        ],
    )
    assert X.shape[1:] == (168, 10)
    assert y.shape[1] == 24


def test_privacy_budget_accountant_returns_finite_values():
    budget = compute_privacy_budget(noise_multiplier=1.1, sample_rate=1.0, rounds=3)
    assert budget["epsilon"] > 0
    assert budget["delta"] == 1e-5


def test_resolve_training_device_prefers_cuda_when_available(monkeypatch):
    from trainer.fl import pipeline as pipeline_module

    monkeypatch.delenv("TRAINER_DEVICE", raising=False)
    monkeypatch.setattr(pipeline_module.torch.cuda, "is_available", lambda: True)

    device = pipeline_module.resolve_training_device()

    assert device.type == "cuda"


def test_local_partition_moves_model_to_selected_device(monkeypatch):
    from trainer.fl import pipeline as pipeline_module

    to_calls: list[str] = []
    original_model_class = pipeline_module.MultiOutputLSTM

    class RecordingModel(original_model_class):
        def to(self, device):
            to_calls.append(str(device))
            return super().to(device)

    monkeypatch.setattr(pipeline_module, "MultiOutputLSTM", RecordingModel)
    monkeypatch.setattr(pipeline_module, "DEVICE", torch.device("cpu"))

    partition = ClientPartition(
        household_id=1,
        household_code="HOUSEHOLD-0001",
        train_X=np.zeros((16, 168, len(FEATURE_COLUMNS)), dtype=np.float32),
        train_y=np.zeros((16, 24), dtype=np.float32),
        val_X=np.zeros((4, 168, len(FEATURE_COLUMNS)), dtype=np.float32),
        val_y=np.zeros((4, 24), dtype=np.float32),
        test_X=np.zeros((4, 168, len(FEATURE_COLUMNS)), dtype=np.float32),
        test_y=np.zeros((4, 24), dtype=np.float32),
        full_frame=pd.DataFrame(),
        scaler_mean=0.0,
        scaler_std=1.0,
        metadata={},
    )
    initial_model = RecordingModel(input_size=len(FEATURE_COLUMNS))
    global_parameters = state_dict_to_ndarrays(initial_model)

    pipeline_module._train_local_partition(
        partition,
        global_parameters=global_parameters,
        config={
            "batch_size": 8,
            "learning_rate": 0.001,
            "local_epochs": 1,
            "gradient_clip_norm": 1.0,
            "huber_delta": 1.0,
        },
    )

    assert "cpu" in to_calls


def test_centralized_baseline_streams_partitions_without_concatenating(monkeypatch):
    from trainer.fl import pipeline as pipeline_module

    partition = ClientPartition(
        household_id=1,
        household_code="HOUSEHOLD-0001",
        train_X=np.zeros((16, 168, len(FEATURE_COLUMNS)), dtype=np.float32),
        train_y=np.zeros((16, 24), dtype=np.float32),
        val_X=np.zeros((8, 168, len(FEATURE_COLUMNS)), dtype=np.float32),
        val_y=np.zeros((8, 24), dtype=np.float32),
        test_X=np.zeros((8, 168, len(FEATURE_COLUMNS)), dtype=np.float32),
        test_y=np.zeros((8, 24), dtype=np.float32),
        full_frame=pd.DataFrame(),
        scaler_mean=0.0,
        scaler_std=1.0,
        metadata={},
    )
    partitions = [partition, partition]

    original_concatenate = pipeline_module.np.concatenate

    def guarded_concatenate(values, axis=0):
        first = next(iter(values), None)
        if isinstance(first, np.ndarray) and first.ndim >= 2:
            raise AssertionError("run_centralized_baseline should not concatenate partition arrays")
        return original_concatenate(values, axis=axis)

    monkeypatch.setattr(pipeline_module.np, "concatenate", guarded_concatenate)

    metrics = pipeline_module.run_centralized_baseline(
        partitions,
        {
            "batch_size": 8,
            "learning_rate": 0.001,
            "local_epochs": 1,
        },
    )

    assert set(metrics) == {"mae", "rmse", "mape"}


def test_centralized_evaluation_uses_partition_specific_scalers(monkeypatch):
    from trainer.fl import pipeline as pipeline_module

    class ZeroForecastModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.zeros((inputs.shape[0], 24), dtype=inputs.dtype, device=inputs.device)

    monkeypatch.setattr(pipeline_module, "DEVICE", torch.device("cpu"))

    base_features = np.zeros((2, 168, len(FEATURE_COLUMNS)), dtype=np.float32)
    base_targets = np.ones((2, 24), dtype=np.float32)
    partitions = [
        ClientPartition(
            household_id=1,
            household_code="HOUSEHOLD-0001",
            train_X=np.zeros((0, 168, len(FEATURE_COLUMNS)), dtype=np.float32),
            train_y=np.zeros((0, 24), dtype=np.float32),
            val_X=base_features,
            val_y=base_targets,
            test_X=np.zeros((0, 168, len(FEATURE_COLUMNS)), dtype=np.float32),
            test_y=np.zeros((0, 24), dtype=np.float32),
            full_frame=pd.DataFrame(),
            scaler_mean=0.0,
            scaler_std=1.0,
            metadata={},
        ),
        ClientPartition(
            household_id=2,
            household_code="HOUSEHOLD-0002",
            train_X=np.zeros((0, 168, len(FEATURE_COLUMNS)), dtype=np.float32),
            train_y=np.zeros((0, 24), dtype=np.float32),
            val_X=base_features,
            val_y=base_targets,
            test_X=np.zeros((0, 168, len(FEATURE_COLUMNS)), dtype=np.float32),
            test_y=np.zeros((0, 24), dtype=np.float32),
            full_frame=pd.DataFrame(),
            scaler_mean=0.0,
            scaler_std=100.0,
            metadata={},
        ),
    ]

    metrics = pipeline_module._evaluate_centralized_partition_set(
        ZeroForecastModel(),
        partitions,
        split="val",
        batch_size=2,
    )

    assert metrics.mae == pytest.approx(50.5, rel=1e-4)


def test_secure_aggregation_matches_plain_weighted_average():
    client_updates = [
        {
            "client_id": 101,
            "num_examples": 8,
            "update": [
                np.array([[0.10, -0.20], [0.05, 0.15]], dtype=np.float32),
                np.array([0.25, -0.10], dtype=np.float32),
            ],
        },
        {
            "client_id": 205,
            "num_examples": 12,
            "update": [
                np.array([[0.08, -0.05], [0.04, 0.02]], dtype=np.float32),
                np.array([0.12, -0.08], dtype=np.float32),
            ],
        },
        {
            "client_id": 309,
            "num_examples": 10,
            "update": [
                np.array([[0.03, -0.11], [0.01, 0.06]], dtype=np.float32),
                np.array([0.14, -0.03], dtype=np.float32),
            ],
        },
    ]

    plain_average, total_examples = average_client_updates(client_updates)
    secure_average, metadata = secure_aggregate_weighted_updates(
        client_updates=client_updates,
        round_number=7,
        mask_degree=2,
        mask_scale=4.0,
        clipping_norm=1.0,
        run_nonce=b"unit-test-secure-agg",
    )

    assert total_examples == 30
    assert metadata["enabled"] is True
    assert metadata["scheme"] == "ring-pair-mask-v1"
    assert metadata["pair_count"] >= 2
    assert metadata["protected_participant_count"] == 3
    assert metadata["cancellation_error_max_abs"] < 1e-6
    for plain_tensor, secure_tensor in zip(plain_average, secure_average, strict=True):
        np.testing.assert_allclose(plain_tensor, secure_tensor, atol=1e-6)


def test_secure_aggregation_single_client_bypasses_masking():
    client_updates = [
        {
            "client_id": 77,
            "num_examples": 4,
            "update": [np.array([0.3, -0.1, 0.05], dtype=np.float32)],
        }
    ]

    secure_average, metadata = secure_aggregate_weighted_updates(
        client_updates=client_updates,
        round_number=1,
        mask_degree=2,
        mask_scale=4.0,
        clipping_norm=1.0,
        run_nonce=b"single-client",
    )

    np.testing.assert_allclose(secure_average[0], np.array([0.3, -0.1, 0.05], dtype=np.float64), atol=1e-8)
    assert metadata["scheme"] == "single-client-bypass"
    assert metadata["pair_count"] == 0


def test_recommendation_engine_emits_peak_shift_action():
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-02-01", periods=24 * 7, freq="1h", tz="UTC"),
            "consumption_kwh": [0.6] * (24 * 7),
        }
    )
    forecast = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-02-10T00:00:00Z", periods=24, freq="1h", tz="UTC"),
            "prediction_kwh": [0.8] * 18 + [1.9] + [0.8] * 5,
            "temperature_2m": [12.0] * 24,
        }
    )
    rows = build_recommendations(household_id=1, forecast_frame=forecast, history_frame=history, model_version_id=1)
    assert any(row["code"] == "shift_peak_load" for row in rows)
