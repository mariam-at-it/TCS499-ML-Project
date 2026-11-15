from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from trainer.core.settings import get_settings


DEFAULT_LOCATION = {"latitude": 53.3498, "longitude": -6.2603, "label": "IRELAND"}


class WeatherClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.archive_url = settings.openmeteo_base_url
        self.forecast_url = settings.openmeteo_forecast_url

    def fetch_archive(self, *, latitude: float, longitude: float, start: str, end: str) -> pd.DataFrame:
        response = httpx.get(
            self.archive_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start,
                "end_date": end,
                "hourly": "temperature_2m,apparent_temperature,precipitation",
                "timezone": "UTC",
            },
            timeout=30.0,
        )
        response.raise_for_status()
        payload = response.json()
        hourly = payload.get("hourly", {})
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(hourly.get("time", []), utc=True),
                "temperature_2m": hourly.get("temperature_2m", []),
                "apparent_temperature": hourly.get("apparent_temperature", []),
                "precipitation": hourly.get("precipitation", []),
            }
        )

    def fetch_forecast(self, *, latitude: float, longitude: float) -> pd.DataFrame:
        response = httpx.get(
            self.forecast_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "temperature_2m,apparent_temperature,precipitation",
                "forecast_days": 2,
                "timezone": "UTC",
            },
            timeout=30.0,
        )
        response.raise_for_status()
        payload = response.json()
        hourly = payload.get("hourly", {})
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(hourly.get("time", []), utc=True),
                "temperature_2m": hourly.get("temperature_2m", []),
                "apparent_temperature": hourly.get("apparent_temperature", []),
                "precipitation": hourly.get("precipitation", []),
            }
        )


def synthesize_weather(index: pd.Series, *, latitude: float = DEFAULT_LOCATION["latitude"], longitude: float = DEFAULT_LOCATION["longitude"]) -> pd.DataFrame:
    timestamps = pd.to_datetime(index, utc=True)
    hours = timestamps.dt.hour.to_numpy()
    doy = timestamps.dt.dayofyear.to_numpy()
    temperature = 9 + 7 * np.sin((doy - 30) / 365 * 2 * np.pi) + 2 * np.sin(hours / 24 * 2 * np.pi)
    apparent = temperature - 1.2
    precipitation = np.clip(0.15 + 0.15 * np.sin(doy / 30 * 2 * np.pi), 0, None)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_2m": temperature.round(3),
            "apparent_temperature": apparent.round(3),
            "precipitation": precipitation.round(3),
            "source": "synthetic",
            "latitude": latitude,
            "longitude": longitude,
        }
    )


def write_weather_artifact(base_dir: Path, region_key: str, frame: pd.DataFrame) -> Path:
    weather_dir = base_dir / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)
    path = weather_dir / f"{region_key}.csv"
    frame.to_csv(path, index=False)
    return path

