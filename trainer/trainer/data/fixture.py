from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HALF_HOURS_PER_DAY = 48
GENERATOR_VERSION = "synthetic-v2"


@dataclass(slots=True)
class FixturePaths:
    households_path: Path
    readings_path: Path
    metadata_path: Path


REGION_SPECS = [
    {"region_code": "DUBLIN", "latitude": 53.3498, "longitude": -6.2603, "temp_bias": 0.9},
    {"region_code": "CORK", "latitude": 51.8985, "longitude": -8.4756, "temp_bias": 0.3},
    {"region_code": "GALWAY", "latitude": 53.2707, "longitude": -9.0568, "temp_bias": 0.0},
    {"region_code": "LIMERICK", "latitude": 52.6638, "longitude": -8.6267, "temp_bias": 0.4},
    {"region_code": "WATERFORD", "latitude": 52.2593, "longitude": -7.1101, "temp_bias": 0.7},
    {"region_code": "SLIGO", "latitude": 54.2697, "longitude": -8.4694, "temp_bias": -0.3},
]

ARCHETYPES = [
    {
        "name": "single_professional",
        "occupants": (1, 2),
        "weekday_day_presence": 0.18,
        "weekend_day_presence": 0.58,
        "base_load": (0.09, 0.16),
        "heating_factor": 0.010,
        "water_heating": 0.11,
        "meal_factor": 0.18,
        "ev_probability": 0.18,
        "solar_probability": 0.12,
        "work_from_home_probability": 0.15,
        "heat_pump_probability": 0.20,
    },
    {
        "name": "family_commuter",
        "occupants": (3, 5),
        "weekday_day_presence": 0.28,
        "weekend_day_presence": 0.72,
        "base_load": (0.18, 0.28),
        "heating_factor": 0.014,
        "water_heating": 0.18,
        "meal_factor": 0.32,
        "ev_probability": 0.32,
        "solar_probability": 0.20,
        "work_from_home_probability": 0.20,
        "heat_pump_probability": 0.28,
    },
    {
        "name": "remote_worker",
        "occupants": (1, 3),
        "weekday_day_presence": 0.76,
        "weekend_day_presence": 0.80,
        "base_load": (0.15, 0.24),
        "heating_factor": 0.013,
        "water_heating": 0.14,
        "meal_factor": 0.24,
        "ev_probability": 0.22,
        "solar_probability": 0.18,
        "work_from_home_probability": 0.95,
        "heat_pump_probability": 0.24,
    },
    {
        "name": "retired_couple",
        "occupants": (2, 2),
        "weekday_day_presence": 0.84,
        "weekend_day_presence": 0.88,
        "base_load": (0.13, 0.22),
        "heating_factor": 0.016,
        "water_heating": 0.15,
        "meal_factor": 0.22,
        "ev_probability": 0.10,
        "solar_probability": 0.14,
        "work_from_home_probability": 0.75,
        "heat_pump_probability": 0.18,
    },
    {
        "name": "ev_solar_household",
        "occupants": (3, 5),
        "weekday_day_presence": 0.34,
        "weekend_day_presence": 0.76,
        "base_load": (0.20, 0.32),
        "heating_factor": 0.013,
        "water_heating": 0.17,
        "meal_factor": 0.30,
        "ev_probability": 0.85,
        "solar_probability": 0.88,
        "work_from_home_probability": 0.35,
        "heat_pump_probability": 0.42,
    },
]


def _pulse_curve(hour_of_day: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((hour_of_day - center) / width) ** 2)


def _sunlight_curve(day_of_year: np.ndarray, hour_of_day: np.ndarray) -> np.ndarray:
    daylight_hours = 8.2 + 8.4 * np.sin((day_of_year - 80) / 365 * 2 * np.pi)
    sunrise = 12.0 - (daylight_hours / 2.0)
    sunset = 12.0 + (daylight_hours / 2.0)
    angle = np.clip((hour_of_day - sunrise) / np.maximum(sunset - sunrise, 1.0), 0.0, 1.0)
    return np.sin(np.pi * angle)


def _metadata_matches(metadata_path: Path, *, households: int, days: int, seed: int) -> bool:
    if not metadata_path.exists():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        payload.get("generator_version") == GENERATOR_VERSION
        and payload.get("households") == households
        and payload.get("days") == days
        and payload.get("seed") == seed
    )


def _build_region_weather(
    timestamps: pd.DatetimeIndex,
    *,
    region_index: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    periods = len(timestamps)
    hour_of_day = timestamps.hour.to_numpy() + (timestamps.minute.to_numpy() / 60.0)
    day_of_year = timestamps.dayofyear.to_numpy()
    region = REGION_SPECS[region_index % len(REGION_SPECS)]
    seasonal = 9.0 + 6.8 * np.sin((day_of_year - 82) / 365 * 2 * np.pi)
    diurnal = 2.3 * np.sin((hour_of_day - 14.0) / 24 * 2 * np.pi)
    front_noise = np.zeros(periods)
    innovations = rng.normal(0.0, 0.25, size=periods)
    for index in range(1, periods):
        front_noise[index] = (0.985 * front_noise[index - 1]) + innovations[index]
    temperature = seasonal + diurnal + front_noise + region["temp_bias"]
    rainfall_band = 0.10 + 0.16 * np.sin((day_of_year + (region_index * 13)) / 17.0)
    cloudiness = np.clip(0.45 + rainfall_band + rng.normal(0.0, 0.06, size=periods), 0.08, 0.98)
    precipitation = np.clip((cloudiness - 0.55) * 0.9, 0.0, None)
    return {
        "temperature_2m": temperature,
        "apparent_temperature": temperature - (0.9 + 0.4 * cloudiness),
        "precipitation": precipitation,
        "cloudiness": cloudiness,
    }


def _add_event(series: np.ndarray, start_index: int, duration_slots: int, magnitude: float) -> None:
    end_index = min(len(series), start_index + duration_slots)
    if end_index <= start_index:
        return
    taper = np.linspace(1.0, 0.72, end_index - start_index)
    series[start_index:end_index] += magnitude * taper


def _generate_household_profile(
    timestamps: pd.DatetimeIndex,
    *,
    household_index: int,
    archetype: dict[str, Any],
    region: dict[str, Any],
    days: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], pd.DataFrame]:
    periods = len(timestamps)
    hour_of_day = timestamps.hour.to_numpy() + (timestamps.minute.to_numpy() / 60.0)
    day_of_week = timestamps.dayofweek.to_numpy()
    is_weekend = day_of_week >= 5
    day_of_year = timestamps.dayofyear.to_numpy()
    day_index = np.arange(periods) // HALF_HOURS_PER_DAY

    region_seed = rng.integers(0, 1_000_000)
    weather_rng = np.random.default_rng(int(region_seed))
    weather = _build_region_weather(timestamps, region_index=household_index, rng=weather_rng)

    occupants = int(rng.integers(archetype["occupants"][0], archetype["occupants"][1] + 1))
    has_ev = bool(rng.random() < archetype["ev_probability"])
    has_solar = bool(rng.random() < archetype["solar_probability"])
    works_from_home = bool(rng.random() < archetype["work_from_home_probability"])
    has_heat_pump = bool(rng.random() < archetype["heat_pump_probability"])
    floor_area_m2 = float(rng.uniform(55.0, 165.0))
    solar_kwp = float(rng.uniform(2.5, 6.5)) if has_solar else 0.0
    standby_load = float(rng.uniform(*archetype["base_load"]))

    daylight = _sunlight_curve(day_of_year, hour_of_day)
    weekday_presence = (
        archetype["weekday_day_presence"]
        + (0.24 * _pulse_curve(hour_of_day, 7.5, 1.8))
        + (0.48 * _pulse_curve(hour_of_day, 20.0, 2.6))
    )
    if works_from_home:
        weekday_presence += 0.32 * _pulse_curve(hour_of_day, 13.0, 4.5)
    else:
        weekday_presence -= 0.18 * _pulse_curve(hour_of_day, 13.0, 3.8)
    weekend_presence = (
        archetype["weekend_day_presence"]
        + (0.18 * _pulse_curve(hour_of_day, 10.5, 3.2))
        + (0.20 * _pulse_curve(hour_of_day, 18.5, 2.8))
    )
    presence = np.where(is_weekend, weekend_presence, weekday_presence)
    presence = np.clip(presence, 0.05, 1.0)

    vacation_factor = np.ones(periods)
    if days >= 90 and rng.random() < 0.55:
        vacation_length = int(rng.integers(5, 13))
        vacation_start = int(rng.integers(0, max(days - vacation_length, 1)))
        vacation_mask = (day_index >= vacation_start) & (day_index < (vacation_start + vacation_length))
        vacation_factor[vacation_mask] = 0.55

    standby = standby_load + (0.015 * occupants) + (0.010 * works_from_home) + (floor_area_m2 / 1000.0)
    occupancy_load = presence * (0.035 * occupants + (0.045 if works_from_home else 0.020))
    breakfast = archetype["meal_factor"] * _pulse_curve(hour_of_day, 7.5, 1.0)
    lunch = (archetype["meal_factor"] * 0.55) * _pulse_curve(hour_of_day, 13.0, 1.4)
    dinner = (archetype["meal_factor"] * 1.20) * _pulse_curve(hour_of_day, 18.5, 1.5)
    cooking = (breakfast + dinner + np.where(is_weekend | works_from_home, lunch, lunch * 0.35)) * (0.6 + 0.15 * occupants)

    heating_base = np.clip((19.5 if has_heat_pump else 18.8) - weather["temperature_2m"], 0.0, None)
    heating = heating_base * archetype["heating_factor"] * (0.72 + 0.34 * presence) * (0.85 + floor_area_m2 / 220.0)
    water_heating = archetype["water_heating"] * (
        _pulse_curve(hour_of_day, 6.8, 1.0) + (1.1 * _pulse_curve(hour_of_day, 20.3, 1.5))
    ) * (0.65 + 0.12 * occupants)

    dishwasher = np.zeros(periods)
    laundry = np.zeros(periods)
    ev_load = np.zeros(periods)
    party_spike = np.zeros(periods)
    for day in range(days):
        base_slot = day * HALF_HOURS_PER_DAY
        weekend = bool(day % 7 in {5, 6})
        if rng.random() < (0.82 if weekend else 0.58):
            start_slot = base_slot + int(rng.integers(38, 45))
            _add_event(dishwasher, start_slot, duration_slots=3, magnitude=float(rng.uniform(0.18, 0.34)))
        if rng.random() < (0.42 if weekend else 0.24):
            start_slot = base_slot + int(rng.integers(14, 40))
            _add_event(laundry, start_slot, duration_slots=int(rng.integers(3, 6)), magnitude=float(rng.uniform(0.30, 0.65)))
        if has_ev and rng.random() < 0.46:
            start_slot = base_slot + int(rng.integers(40, 47))
            _add_event(ev_load, start_slot, duration_slots=int(rng.integers(6, 12)), magnitude=float(rng.uniform(0.55, 1.25)))
        if rng.random() < 0.025:
            start_slot = base_slot + int(rng.integers(34, 43))
            _add_event(party_spike, start_slot, duration_slots=int(rng.integers(3, 8)), magnitude=float(rng.uniform(0.25, 0.55)))

    solar_offset = np.zeros(periods)
    if has_solar:
        irradiance = daylight * np.clip(1.02 - (weather["cloudiness"] * 0.65), 0.12, 1.0)
        solar_offset = solar_kwp * irradiance * 0.17

    innovation = rng.normal(0.0, 0.018 + (0.004 * occupants), size=periods)
    autocorrelated_noise = np.zeros(periods)
    for index in range(1, periods):
        autocorrelated_noise[index] = (0.82 * autocorrelated_noise[index - 1]) + innovation[index]

    consumption = (
        standby
        + occupancy_load
        + cooking
        + heating
        + water_heating
        + dishwasher
        + laundry
        + ev_load
        + party_spike
        + autocorrelated_noise
        - solar_offset
    ) * vacation_factor
    consumption = np.clip(consumption, 0.05, None)

    household_code = f"HOUSEHOLD-{household_index + 1:04d}"
    household_row = {
        "household_id": household_code,
        "display_name": f"Synthetic Household {household_index + 1}",
        "region_code": region["region_code"],
        "latitude": region["latitude"],
        "longitude": region["longitude"],
        "archetype": archetype["name"],
        "occupant_count": occupants,
        "floor_area_m2": round(floor_area_m2, 1),
        "has_ev": has_ev,
        "solar_kwp": round(solar_kwp, 2),
        "heating_system": "heat_pump" if has_heat_pump else "gas_boiler",
        "work_pattern": "remote_or_home_daytime" if works_from_home else "commuter",
    }
    readings = pd.DataFrame(
        {
            "household_id": household_code,
            "timestamp": timestamps,
            "consumption_kwh": np.round(consumption, 4),
        }
    )
    return household_row, readings


def generate_fixture_dataset(
    base_dir: Path,
    households: int = 96,
    days: int = 365,
    *,
    seed: int = 42,
    overwrite: bool = False,
) -> FixturePaths:
    base_dir.mkdir(parents=True, exist_ok=True)
    households_path = base_dir / "households.csv"
    readings_path = base_dir / "readings.csv"
    metadata_path = base_dir / "fixture_metadata.json"
    if (
        not overwrite
        and households_path.exists()
        and readings_path.exists()
        and _metadata_matches(metadata_path, households=households, days=days, seed=seed)
    ):
        return FixturePaths(households_path, readings_path, metadata_path)

    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    periods = days * HALF_HOURS_PER_DAY
    timestamps = pd.date_range(start=start, periods=periods, freq="30min", tz="UTC")

    household_rows: list[dict[str, Any]] = []
    reading_frames: list[pd.DataFrame] = []
    archetype_weights = np.array([0.18, 0.34, 0.18, 0.16, 0.14], dtype=float)
    for index in range(households):
        archetype = ARCHETYPES[int(rng.choice(len(ARCHETYPES), p=archetype_weights))]
        region = REGION_SPECS[index % len(REGION_SPECS)]
        household_row, household_readings = _generate_household_profile(
            timestamps,
            household_index=index,
            archetype=archetype,
            region=region,
            days=days,
            rng=np.random.default_rng(int(rng.integers(0, 1_000_000_000))),
        )
        household_rows.append(household_row)
        reading_frames.append(household_readings)

    pd.DataFrame(household_rows).to_csv(households_path, index=False)
    pd.concat(reading_frames, ignore_index=True).to_csv(readings_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "generator_version": GENERATOR_VERSION,
                "households": households,
                "days": days,
                "seed": seed,
                "start_at": timestamps.min().isoformat(),
                "end_at": timestamps.max().isoformat(),
                "interval_minutes": 30,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return FixturePaths(households_path, readings_path, metadata_path)
