from __future__ import annotations

import json
import math
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from trainer.data.preprocess import add_time_features, create_windows
from trainer.fl.accounting import compute_privacy_budget
from trainer.fl.secure_agg import SECURE_AGG_SCHEME, average_client_updates, secure_aggregate_weighted_updates
from trainer.fl.model import (
    MetricBundle,
    MultiOutputLSTM,
    dataset_loader,
    evaluate_model,
    ndarrays_to_state_dict,
    state_dict_to_ndarrays,
    train_epoch,
)
from trainer.weather.client import synthesize_weather


FEATURE_COLUMNS = [
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
]
TARGET_COLUMN = "consumption_kwh"
DEVICE = torch.device("cpu")
torch.set_num_threads(int(os.getenv("TRAINER_TORCH_THREADS", "1")))
torch.set_num_interop_threads(1)


class TrainingCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class ClientPartition:
    household_id: int
    household_code: str
    train_X: np.ndarray
    train_y: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    full_frame: pd.DataFrame
    scaler_mean: float
    scaler_std: float
    metadata: dict[str, Any]


def _merge_weather(frame: pd.DataFrame, weather_path: Path | None) -> pd.DataFrame:
    if weather_path and weather_path.exists():
        weather = pd.read_csv(weather_path)
        weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True)
    else:
        weather = synthesize_weather(frame["timestamp"])
    merged = frame.copy()
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
    merged = merged.merge(weather.drop(columns=[col for col in ["source", "latitude", "longitude"] if col in weather]), on="timestamp", how="left")
    for column in ["temperature_2m", "apparent_temperature", "precipitation"]:
        if column not in merged:
            merged[column] = 0.0
        merged[column] = merged[column].interpolate(limit_direction="both").fillna(0.0)
    return add_time_features(merged)


def prepare_client_partitions(*, artifact_root: Path, household_rows: list[dict[str, Any]], selected_household_ids: list[int] | None = None) -> list[ClientPartition]:
    households_dir = artifact_root / "households"
    weather_dir = artifact_root / "weather"
    selected = set(selected_household_ids or [])
    prepared_frames: list[tuple[dict[str, Any], pd.DataFrame]] = []
    for household in household_rows:
        if selected and household["id"] not in selected:
            continue
        household_path = households_dir / f"{household['household_code']}.csv"
        if not household_path.exists():
            continue
        frame = pd.read_csv(household_path)
        weather_path = weather_dir / f"{household.get('region_code') or 'IRELAND'}.csv"
        merged = _merge_weather(frame, weather_path if weather_path.exists() else None)
        prepared_frames.append((household, merged))

    if not prepared_frames:
        raise ValueError("No household partitions available for training")

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    train_feature_chunks: list[np.ndarray] = []
    train_target_chunks: list[np.ndarray] = []
    for _, frame in prepared_frames:
        train_end = int(len(frame) * 0.7)
        train_feature_chunks.append(frame.iloc[:train_end][FEATURE_COLUMNS].to_numpy())
        train_target_chunks.append(frame.iloc[:train_end][[TARGET_COLUMN]].to_numpy())
    feature_scaler.fit(np.concatenate(train_feature_chunks, axis=0))
    target_scaler.fit(np.concatenate(train_target_chunks, axis=0))

    partitions: list[ClientPartition] = []
    for household, frame in prepared_frames:
        scaled = frame.copy()
        scaled[FEATURE_COLUMNS] = feature_scaler.transform(frame[FEATURE_COLUMNS].to_numpy())
        scaled[TARGET_COLUMN] = target_scaler.transform(frame[[TARGET_COLUMN]].to_numpy())
        X, y, target_times = create_windows(scaled, feature_columns=FEATURE_COLUMNS, target_column=TARGET_COLUMN)
        if len(X) < 16:
            continue
        times = pd.to_datetime(pd.Series(target_times), utc=True)
        val_cut = times.quantile(0.70)
        test_cut = times.quantile(0.85)
        train_mask = times < val_cut
        val_mask = (times >= val_cut) & (times < test_cut)
        test_mask = times >= test_cut
        partitions.append(
            ClientPartition(
                household_id=household["id"],
                household_code=household["household_code"],
                train_X=X[train_mask.to_numpy()],
                train_y=y[train_mask.to_numpy()],
                val_X=X[val_mask.to_numpy()],
                val_y=y[val_mask.to_numpy()],
                test_X=X[test_mask.to_numpy()],
                test_y=y[test_mask.to_numpy()],
                full_frame=frame,
                scaler_mean=float(target_scaler.mean_[0]),
                scaler_std=float(target_scaler.scale_[0]),
                metadata={
                    "feature_means": feature_scaler.mean_.tolist(),
                    "feature_scales": feature_scaler.scale_.tolist(),
                    "target_mean": float(target_scaler.mean_[0]),
                    "target_scale": float(target_scaler.scale_[0]),
                },
            )
        )
    return partitions


def _evaluate_global(parameters: list[np.ndarray], partitions: list[ClientPartition], batch_size: int) -> tuple[float, dict[str, float]]:
    model = MultiOutputLSTM(input_size=len(FEATURE_COLUMNS))
    ndarrays_to_state_dict(model, parameters)
    metrics: list[MetricBundle] = []
    for partition in partitions:
        if len(partition.val_X) == 0:
            continue
        loader = dataset_loader(partition.val_X, partition.val_y, batch_size, shuffle=False)
        metrics.append(evaluate_model(model, loader, (partition.scaler_mean, partition.scaler_std), DEVICE))
    if not metrics:
        return 0.0, {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
    mae = float(np.mean([metric.mae for metric in metrics]))
    rmse = float(np.mean([metric.rmse for metric in metrics]))
    mape = float(np.mean([metric.mape for metric in metrics]))
    loss = float(np.mean([metric.loss for metric in metrics]))
    return loss, {"mae": mae, "rmse": rmse, "mape": mape}


def _evaluate_test(parameters: list[np.ndarray], partitions: list[ClientPartition], batch_size: int) -> dict[str, float]:
    model = MultiOutputLSTM(input_size=len(FEATURE_COLUMNS))
    ndarrays_to_state_dict(model, parameters)
    rows: list[MetricBundle] = []
    for partition in partitions:
        if len(partition.test_X) == 0:
            continue
        loader = dataset_loader(partition.test_X, partition.test_y, batch_size, shuffle=False)
        rows.append(evaluate_model(model, loader, (partition.scaler_mean, partition.scaler_std), DEVICE))
    if not rows:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
    return {
        "mae": float(np.mean([row.mae for row in rows])),
        "rmse": float(np.mean([row.rmse for row in rows])),
        "mape": float(np.mean([row.mape for row in rows])),
    }


def run_centralized_baseline(partitions: list[ClientPartition], config: dict[str, Any]) -> dict[str, float]:
    train_X = np.concatenate([partition.train_X for partition in partitions if len(partition.train_X)], axis=0)
    train_y = np.concatenate([partition.train_y for partition in partitions if len(partition.train_y)], axis=0)
    val_X = np.concatenate([partition.val_X for partition in partitions if len(partition.val_X)], axis=0)
    val_y = np.concatenate([partition.val_y for partition in partitions if len(partition.val_y)], axis=0)
    test_X = np.concatenate([partition.test_X for partition in partitions if len(partition.test_X)], axis=0)
    test_y = np.concatenate([partition.test_y for partition in partitions if len(partition.test_y)], axis=0)

    model = MultiOutputLSTM(input_size=len(FEATURE_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    criterion = torch.nn.MSELoss()
    train_loader = dataset_loader(train_X, train_y, int(config["batch_size"]), shuffle=True)
    val_loader = dataset_loader(val_X, val_y, int(config["batch_size"]), shuffle=False)
    target_scale = (partitions[0].scaler_mean, partitions[0].scaler_std)
    best_metrics = {"mae": float("inf"), "rmse": float("inf"), "mape": float("inf")}
    for _ in range(max(4, int(config["local_epochs"]) * 4)):
        train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        metrics = evaluate_model(model, val_loader, target_scale, DEVICE)
        if metrics.mae < best_metrics["mae"]:
            test_loader = dataset_loader(test_X, test_y, int(config["batch_size"]), shuffle=False)
            test_metrics = evaluate_model(model, test_loader, target_scale, DEVICE)
            best_metrics = {"mae": test_metrics.mae, "rmse": test_metrics.rmse, "mape": test_metrics.mape}
    return best_metrics


def _parameter_bytes(parameters: list[np.ndarray]) -> int:
    return int(sum(array.nbytes for array in parameters))


def _sample_partitions(
    partitions: list[ClientPartition],
    *,
    sample_rate: float,
    round_number: int,
) -> list[ClientPartition]:
    sampled_clients = max(1, math.ceil(len(partitions) * sample_rate))
    if sampled_clients >= len(partitions):
        return list(partitions)
    rng = np.random.default_rng(10_000 + round_number)
    indices = rng.choice(len(partitions), size=sampled_clients, replace=False)
    return [partitions[int(index)] for index in sorted(indices)]


def _clip_update(update: list[np.ndarray], clipping_norm: float) -> tuple[list[np.ndarray], float]:
    flat_norm = float(np.sqrt(sum(float(np.sum(array.astype(np.float64) ** 2)) for array in update)))
    if flat_norm == 0.0 or flat_norm <= clipping_norm:
        return update, flat_norm
    scale = clipping_norm / flat_norm
    return [array * scale for array in update], flat_norm


def _aggregate_updates(
    global_parameters: list[np.ndarray],
    client_updates: list[dict[str, Any]],
    *,
    dp_enabled: bool,
    clipping_norm: float,
    noise_multiplier: float,
    round_seed: int,
    secure_aggregation_enabled: bool,
    secure_aggregation_degree: int,
    secure_aggregation_mask_scale: float,
    secure_aggregation_nonce: bytes,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if secure_aggregation_enabled:
        averaged_update, secure_aggregation_metadata = secure_aggregate_weighted_updates(
            client_updates=client_updates,
            round_number=round_seed,
            mask_degree=secure_aggregation_degree,
            mask_scale=secure_aggregation_mask_scale,
            clipping_norm=clipping_norm,
            run_nonce=secure_aggregation_nonce,
        )
    else:
        averaged_update, total_examples = average_client_updates(client_updates)
        secure_aggregation_metadata = {
            "enabled": False,
            "scheme": "disabled",
            "participant_ids": [int(row["client_id"]) for row in client_updates],
            "protected_participant_count": 0,
            "pair_count": 0,
            "mask_degree": 0,
            "mask_target_norm": 0.0,
            "cancellation_error_max_abs": 0.0,
            "total_examples": total_examples,
        }
    if dp_enabled and noise_multiplier > 0:
        rng = np.random.default_rng(round_seed)
        noise_scale = (noise_multiplier * clipping_norm) / max(len(client_updates), 1)
        for index, array in enumerate(averaged_update):
            averaged_update[index] = array + rng.normal(loc=0.0, scale=noise_scale, size=array.shape)
    updated_parameters = [
        (parameter.astype(np.float64, copy=False) + delta.astype(np.float64, copy=False)).astype(parameter.dtype, copy=False)
        for parameter, delta in zip(global_parameters, averaged_update, strict=True)
    ]
    return updated_parameters, secure_aggregation_metadata


def _train_local_partition(
    partition: ClientPartition,
    *,
    global_parameters: list[np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    model = MultiOutputLSTM(input_size=len(FEATURE_COLUMNS))
    ndarrays_to_state_dict(model, global_parameters)
    loader = dataset_loader(partition.train_X, partition.train_y, int(config["batch_size"]), shuffle=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    criterion = torch.nn.HuberLoss(delta=float(config.get("huber_delta", 1.0)))
    grad_clip_norm = float(config.get("gradient_clip_norm", 1.0))
    losses: list[float] = []
    model.train()
    for _ in range(int(config["local_epochs"])):
        for features, target in loader:
            features = features.to(DEVICE)
            target = target.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = criterion(predictions, target)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
    local_parameters = state_dict_to_ndarrays(model)
    return {
        "parameters": local_parameters,
        "num_examples": len(partition.train_X),
        "train_loss": float(np.mean(losses)) if losses else 0.0,
    }


def run_federated_training(
    *,
    partitions: list[ClientPartition],
    config: dict[str, Any],
    output_dir: Path,
    round_callback: Callable[[dict[str, Any]], None] | None = None,
    should_abort: Callable[[], bool] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_model = MultiOutputLSTM(input_size=len(FEATURE_COLUMNS))
    global_parameters = state_dict_to_ndarrays(initial_model)
    best_parameters = [array.copy() for array in global_parameters]
    best_val_mae = float("inf")
    round_rows: list[dict[str, Any]] = []
    sample_rate = float(config["client_sampling_rate"])
    rounds = int(config["rounds"])
    dp_enabled = bool(config["dp_enabled"])
    clipping_norm = float(config["clipping_norm"])
    noise_multiplier = float(config["noise_multiplier"]) if dp_enabled else 0.0
    secure_aggregation_enabled = bool(config.get("secure_aggregation_enabled", True))
    secure_aggregation_degree = int(config.get("secure_aggregation_degree", 2))
    secure_aggregation_mask_scale = float(config.get("secure_aggregation_mask_scale", 4.0))
    secure_aggregation_nonce = secrets.token_bytes(16)
    secure_aggregation_round_summaries: list[dict[str, Any]] = []

    for round_number in range(1, rounds + 1):
        if should_abort and should_abort():
            raise TrainingCancelled(f"Run cancelled before round {round_number}")
        started_at = time.perf_counter()
        selected_partitions = _sample_partitions(partitions, sample_rate=sample_rate, round_number=round_number)
        if progress_callback:
            progress_callback(f"Round {round_number}/{rounds}: training {len(selected_partitions)} clients")
        client_updates: list[dict[str, Any]] = []
        train_losses: list[tuple[int, float]] = []
        for index, partition in enumerate(selected_partitions, start=1):
            if should_abort and should_abort():
                raise TrainingCancelled(f"Run cancelled during round {round_number}")
            local_result = _train_local_partition(partition, global_parameters=global_parameters, config=config)
            update = [local_param - global_param for local_param, global_param in zip(local_result["parameters"], global_parameters, strict=True)]
            raw_update_norm = float(np.sqrt(sum(float(np.sum(array.astype(np.float64) ** 2)) for array in update)))
            clipped_update, clipped_norm = _clip_update(update, clipping_norm) if dp_enabled else (update, raw_update_norm)
            client_updates.append(
                {
                    "client_id": partition.household_id,
                    "update": clipped_update,
                    "num_examples": local_result["num_examples"],
                    "train_loss": local_result["train_loss"],
                    "raw_update_norm": raw_update_norm,
                    "clipped_update_norm": clipped_norm,
                }
            )
            train_losses.append((int(local_result["num_examples"]), float(local_result["train_loss"])))
            if progress_callback and (index == len(selected_partitions) or index % 8 == 0):
                progress_callback(f"Round {round_number}/{rounds}: completed {index}/{len(selected_partitions)} clients")

        global_parameters, secure_aggregation_metadata = _aggregate_updates(
            global_parameters,
            client_updates,
            dp_enabled=dp_enabled,
            clipping_norm=clipping_norm,
            noise_multiplier=noise_multiplier,
            round_seed=20_000 + round_number,
            secure_aggregation_enabled=secure_aggregation_enabled,
            secure_aggregation_degree=secure_aggregation_degree,
            secure_aggregation_mask_scale=secure_aggregation_mask_scale,
            secure_aggregation_nonce=secure_aggregation_nonce,
        )
        secure_aggregation_round_summaries.append(secure_aggregation_metadata)
        validation_loss, validation_metrics = _evaluate_global(global_parameters, partitions, int(config["batch_size"]))
        if validation_metrics["mae"] < best_val_mae:
            best_val_mae = validation_metrics["mae"]
            best_parameters = [array.copy() for array in global_parameters]

        total_examples = sum(num_examples for num_examples, _ in train_losses)
        weighted_train_loss = (
            sum((num_examples / total_examples) * loss for num_examples, loss in train_losses) if total_examples else 0.0
        )
        round_row = {
            "round_number": round_number,
            "metrics": {
                "train_loss": float(weighted_train_loss),
                "server_loss": float(validation_loss),
                "mae": float(validation_metrics["mae"]),
                "rmse": float(validation_metrics["rmse"]),
                "mape": float(validation_metrics["mape"]),
                "mean_update_norm": float(np.mean([row["raw_update_norm"] for row in client_updates])) if client_updates else 0.0,
                "secure_aggregation": {
                    "enabled": bool(secure_aggregation_metadata["enabled"]),
                    "scheme": secure_aggregation_metadata["scheme"],
                    "mask_degree": int(secure_aggregation_metadata["mask_degree"]),
                    "pair_count": int(secure_aggregation_metadata["pair_count"]),
                    "protected_participant_count": int(secure_aggregation_metadata["protected_participant_count"]),
                    "mask_target_norm": float(secure_aggregation_metadata["mask_target_norm"]),
                    "cancellation_error_max_abs": float(secure_aggregation_metadata["cancellation_error_max_abs"]),
                },
            },
            "bytes_transferred": _parameter_bytes(global_parameters) * max(len(selected_partitions), 1) * 2,
            "duration_seconds": time.perf_counter() - started_at,
            "participant_count": len(selected_partitions),
        }
        round_rows.append(round_row)
        if round_callback:
            round_callback(round_row)
        if progress_callback:
            progress_callback(
                "Round {round}/{total}: mae={mae:.4f} rmse={rmse:.4f} mape={mape:.2f} duration={duration:.2f}s".format(
                    round=round_number,
                    total=rounds,
                    mae=float(validation_metrics["mae"]),
                    rmse=float(validation_metrics["rmse"]),
                    mape=float(validation_metrics["mape"]),
                    duration=float(round_row["duration_seconds"]),
                )
            )

    if should_abort and should_abort():
        raise TrainingCancelled("Run cancelled before final evaluation")
    if progress_callback:
        progress_callback("Evaluating final federated model on holdout test data")
    final_parameters = best_parameters
    test_metrics = _evaluate_test(final_parameters, partitions, int(config["batch_size"]))
    if progress_callback:
        progress_callback("Training centralized baseline for comparison")
    baseline_metrics = run_centralized_baseline(partitions, config)
    if progress_callback:
        progress_callback("Writing model artifacts and final metrics")
    privacy_budget = compute_privacy_budget(
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        rounds=rounds,
    )

    model = MultiOutputLSTM(input_size=len(FEATURE_COLUMNS))
    ndarrays_to_state_dict(model, final_parameters)
    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "client_households": [partition.household_id for partition in partitions],
        "partition_count": len(partitions),
        "target_mean": partitions[0].scaler_mean,
        "target_scale": partitions[0].scaler_std,
        "round_metrics": round_rows,
        "best_validation_mae": best_val_mae,
        "security_summary": {
            "secure_aggregation_enabled": secure_aggregation_enabled,
            "secure_aggregation_scheme": SECURE_AGG_SCHEME if secure_aggregation_enabled else "disabled",
            "secure_aggregation_degree": secure_aggregation_degree if secure_aggregation_enabled else 0,
            "secure_aggregation_mask_scale": secure_aggregation_mask_scale if secure_aggregation_enabled else 0.0,
            "max_cancellation_error_abs": max(
                float(row["cancellation_error_max_abs"]) for row in secure_aggregation_round_summaries
            )
            if secure_aggregation_round_summaries
            else 0.0,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "artifact_manifest": {
            "model_path": str(model_path),
            "metadata_path": str(output_dir / "metadata.json"),
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        },
        "run_summary": {
            "fl_metrics": test_metrics,
            "centralized_metrics": baseline_metrics,
            "accuracy_gap_mae": round(test_metrics["mae"] - baseline_metrics["mae"], 5),
            "history_summary": {
                "best_validation_mae": best_val_mae,
                "round_count": rounds,
                "participant_count": len(partitions),
            },
            "security_summary": {
                "secure_aggregation_enabled": secure_aggregation_enabled,
                "secure_aggregation_scheme": SECURE_AGG_SCHEME if secure_aggregation_enabled else "disabled",
                "secure_aggregation_degree": secure_aggregation_degree if secure_aggregation_enabled else 0,
                "secure_aggregation_mask_scale": secure_aggregation_mask_scale if secure_aggregation_enabled else 0.0,
                "max_cancellation_error_abs": max(
                    float(row["cancellation_error_max_abs"]) for row in secure_aggregation_round_summaries
                )
                if secure_aggregation_round_summaries
                else 0.0,
            },
        },
        "privacy_budget": privacy_budget,
        "round_metrics": round_rows,
    }


def load_model_for_inference(model_path: Path) -> MultiOutputLSTM:
    model = MultiOutputLSTM(input_size=len(FEATURE_COLUMNS))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def forecast_next_24h(model_path: Path, partition: ClientPartition) -> pd.DataFrame:
    model = load_model_for_inference(model_path)
    last_window = partition.full_frame.copy()
    merged = _merge_weather(last_window, None)
    scaled = merged.copy()
    for idx, column in enumerate(FEATURE_COLUMNS):
        mean = partition.metadata["feature_means"][idx]
        scale = partition.metadata["feature_scales"][idx] or 1.0
        scaled[column] = (scaled[column] - mean) / scale
    inputs = torch.tensor(scaled[FEATURE_COLUMNS].tail(168).to_numpy(dtype=np.float32)[None, ...], dtype=torch.float32)
    with torch.no_grad():
        forecast_scaled = model(inputs).numpy()[0]
    forecast = (forecast_scaled * partition.scaler_std) + partition.scaler_mean
    start = pd.to_datetime(partition.full_frame["timestamp"].iloc[-1], utc=True) + pd.Timedelta(hours=1)
    timestamps = pd.date_range(start=start, periods=24, freq="1h", tz="UTC")
    weather_context = synthesize_weather(pd.Series(timestamps))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "prediction_kwh": forecast.round(4),
            "temperature_2m": weather_context["temperature_2m"].to_numpy(),
            "apparent_temperature": weather_context["apparent_temperature"].to_numpy(),
            "precipitation": weather_context["precipitation"].to_numpy(),
        }
    )
