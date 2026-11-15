from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


def build_recommendations(
    *,
    household_id: int,
    forecast_frame: pd.DataFrame,
    history_frame: pd.DataFrame,
    model_version_id: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline = float(history_frame["consumption_kwh"].mean())
    peak_threshold = float(history_frame["consumption_kwh"].quantile(0.75))
    peak_hour = forecast_frame.loc[forecast_frame["prediction_kwh"].idxmax()]
    total_forecast = float(forecast_frame["prediction_kwh"].sum())

    if total_forecast > baseline * 24 * 1.12:
        rows.append(
            {
                "household_id": household_id,
                "code": "trim_high_usage",
                "severity": "high",
                "evidence": {"baseline_hourly_mean": round(baseline, 3), "forecast_total_kwh": round(total_forecast, 3)},
                "message": "Tomorrow's forecast is materially above your normal usage. Delay discretionary appliance use where possible.",
                "valid_until": datetime.now(UTC) + timedelta(days=1),
                "created_at": datetime.now(UTC),
            }
        )

    if peak_hour["prediction_kwh"] > peak_threshold and peak_hour["timestamp"].hour in {17, 18, 19, 20}:
        rows.append(
            {
                "household_id": household_id,
                "code": "shift_peak_load",
                "severity": "medium",
                "evidence": {
                    "peak_hour": peak_hour["timestamp"].isoformat(),
                    "predicted_peak_kwh": round(float(peak_hour["prediction_kwh"]), 3),
                    "baseline_75th_percentile": round(peak_threshold, 3),
                },
                "message": "A high evening peak is expected. Shift laundry, dishwashing, or EV charging outside 17:00 to 20:00 if possible.",
                "valid_until": datetime.now(UTC) + timedelta(days=1),
                "created_at": datetime.now(UTC),
            }
        )

    hot_or_cold = forecast_frame[(forecast_frame["temperature_2m"] >= 24) | (forecast_frame["temperature_2m"] <= 3)]
    if not hot_or_cold.empty:
        extreme = hot_or_cold.iloc[0]
        rows.append(
            {
                "household_id": household_id,
                "code": "weather_driven_efficiency",
                "severity": "low",
                "evidence": {
                    "timestamp": extreme["timestamp"].isoformat(),
                    "temperature_2m": round(float(extreme["temperature_2m"]), 2),
                },
                "message": "Weather conditions may drive heating or cooling demand. Pre-condition the home gradually and avoid abrupt thermostat changes.",
                "valid_until": datetime.now(UTC) + timedelta(days=1),
                "created_at": datetime.now(UTC),
            }
        )

    if not rows:
        rows.append(
            {
                "household_id": household_id,
                "code": "steady_state",
                "severity": "info",
                "evidence": {"forecast_total_kwh": round(total_forecast, 3)},
                "message": "Your forecast is close to baseline. Maintain current usage habits and monitor evening demand.",
                "valid_until": datetime.now(UTC) + timedelta(days=1),
                "created_at": datetime.now(UTC),
            }
        )

    for row in rows:
        row["model_version_id"] = model_version_id
    return rows
