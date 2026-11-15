from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import holidays
import numpy as np
import pandas as pd


CER_ALIASES = {
    "household_id": ["household_id", "meter_id", "meterid", "meter_identifier", "id"],
    "timestamp": ["timestamp", "datetime", "date_time", "reading_datetime", "read_datetime"],
    "consumption_kwh": ["consumption_kwh", "kwh", "usage_kwh", "consumption", "reading_value"],
    "display_name": ["display_name", "name"],
    "region_code": ["region_code", "region", "county", "area", "tariff_group"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng"],
}
CER_METADATA_ALIASES = {
    "household_id": CER_ALIASES["household_id"],
    "code": ["code", "group_code", "participant_code", "category"],
    "residential_stimulus": ["residential stimulus", "residential_stimulus", "stimulus"],
    "residential_tariff": ["residential tariff", "residential_tariff", "tariff", "tariff_code"],
    "sme_allocation": ["sme allocation", "sme_allocation", "sme_stimulus"],
}
CER_BASE_TIMESTAMP = pd.Timestamp("2009-01-01T00:00:00Z")
CER_INTERVALS_PER_DAY = 48
CER_CHUNK_SIZE = 250_000
MIN_HOURLY_RECORDS = 24 * 14


@dataclass(slots=True)
class ImportedDataset:
    households: list[dict[str, Any]]
    household_series: dict[str, pd.DataFrame]
    manifest: dict[str, Any]


def _resolve_column(columns: list[str], aliases: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def _normalize_household_id(value: Any) -> str:
    return str(value).strip().upper()


def _compact_dataset_available(dataset_path: Path) -> bool:
    if dataset_path.is_dir():
        return (dataset_path / "readings.csv").exists()
    return dataset_path.suffix.lower() == ".csv"


def load_households_and_readings(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset_path.is_dir():
        households_path = dataset_path / "households.csv"
        readings_path = dataset_path / "readings.csv"
    else:
        households_path = dataset_path.parent / "households.csv"
        readings_path = dataset_path

    if not readings_path.exists():
        raise FileNotFoundError(f"Readings file not found: {readings_path}")

    readings = pd.read_csv(readings_path)
    households = pd.read_csv(households_path) if households_path.exists() else pd.DataFrame()
    return households, readings


def _import_compact_dataset(dataset_path: Path, output_dir: Path, sample_households: int | None = None) -> ImportedDataset:
    households_df, readings_df = load_households_and_readings(dataset_path)
    household_col = _resolve_column(readings_df.columns.tolist(), CER_ALIASES["household_id"])
    timestamp_col = _resolve_column(readings_df.columns.tolist(), CER_ALIASES["timestamp"])
    consumption_col = _resolve_column(readings_df.columns.tolist(), CER_ALIASES["consumption_kwh"])
    if household_col is None or timestamp_col is None or consumption_col is None:
        raise ValueError("Dataset is missing required household/timestamp/consumption columns")

    readings = readings_df.rename(
        columns={
            household_col: "household_id",
            timestamp_col: "timestamp",
            consumption_col: "consumption_kwh",
        }
    )[["household_id", "timestamp", "consumption_kwh"]]
    readings["timestamp"] = pd.to_datetime(readings["timestamp"], utc=True, errors="coerce")
    readings["consumption_kwh"] = pd.to_numeric(readings["consumption_kwh"], errors="coerce")
    readings = readings.dropna(subset=["timestamp", "consumption_kwh", "household_id"])
    readings["household_id"] = readings["household_id"].map(_normalize_household_id)

    household_meta = pd.DataFrame({"household_id": readings["household_id"].astype(str).unique()})
    if not households_df.empty:
        rename_map: dict[str, str] = {}
        for canonical, aliases in CER_ALIASES.items():
            resolved = _resolve_column(households_df.columns.tolist(), aliases)
            if resolved:
                rename_map[resolved] = canonical
        household_meta = households_df.rename(columns=rename_map)
        if "household_id" not in household_meta:
            household_meta["household_id"] = household_meta.index.astype(str)

    household_meta["household_id"] = household_meta["household_id"].map(_normalize_household_id)

    grouped: dict[str, pd.DataFrame] = {}
    household_rows: list[dict[str, Any]] = []
    count = 0
    for household_id, group in readings.groupby("household_id"):
        hourly = (
            group.sort_values("timestamp")
            .set_index("timestamp")[["consumption_kwh"]]
            .resample("1h")
            .sum(min_count=1)
            .rename(columns={"consumption_kwh": "consumption_kwh"})
        )
        hourly = hourly.interpolate(limit=4).ffill().dropna()
        if len(hourly) < (24 * 14):
            continue
        meta_row = household_meta.loc[household_meta["household_id"] == household_id].head(1)
        region_code = meta_row["region_code"].iloc[0] if "region_code" in meta_row and not meta_row.empty else "IRELAND"
        latitude = (
            float(meta_row["latitude"].iloc[0])
            if "latitude" in meta_row and not meta_row.empty and pd.notna(meta_row["latitude"].iloc[0])
            else 53.3498
        )
        longitude = (
            float(meta_row["longitude"].iloc[0])
            if "longitude" in meta_row and not meta_row.empty and pd.notna(meta_row["longitude"].iloc[0])
            else -6.2603
        )
        display_name = (
            str(meta_row["display_name"].iloc[0]) if "display_name" in meta_row and not meta_row.empty else f"Household {household_id}"
        )
        grouped[household_id] = hourly.reset_index()
        household_rows.append(
            {
                "household_code": household_id.upper(),
                "display_name": display_name,
                "region_code": str(region_code).upper() if region_code else "IRELAND",
                "latitude": latitude,
                "longitude": longitude,
                "metadata_json": {
                    "source_household_id": household_id,
                    "records_hourly": len(hourly),
                    "start_at": hourly.index.min().isoformat(),
                    "end_at": hourly.index.max().isoformat(),
                },
            }
        )
        count += 1
        if sample_households and count >= sample_households:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    households_dir = output_dir / "households"
    households_dir.mkdir(exist_ok=True)
    all_starts: list[pd.Timestamp] = []
    all_ends: list[pd.Timestamp] = []
    for household_code, frame in grouped.items():
        frame.to_csv(households_dir / f"{household_code.upper()}.csv", index=False)
        all_starts.append(frame["timestamp"].min())
        all_ends.append(frame["timestamp"].max())

    manifest = {
        "household_codes": list(grouped.keys()),
        "artifact_root": str(output_dir),
        "processed_households_dir": str(households_dir),
        "window_input_hours": 168,
        "window_output_hours": 24,
        "weather_ready": False,
        "fallback_weather_used": False,
        "start_at": min(all_starts).isoformat() if all_starts else None,
        "end_at": max(all_ends).isoformat() if all_ends else None,
        "created_at": datetime.now(UTC).isoformat(),
    }
    return ImportedDataset(households=household_rows, household_series=grouped, manifest=manifest)


def _discover_cer_reading_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file():
        if dataset_path.suffix.lower() in {".zip", ".txt"}:
            return [dataset_path]
        return []

    reading_files: list[Path] = []
    for pattern in ("File*.txt.zip", "File*.txt"):
        reading_files.extend(sorted(dataset_path.rglob(pattern)))
    return reading_files


def _missing_cer_readings_message(dataset_path: Path) -> str:
    base = f"No CER trial reading files found under {dataset_path}."
    if not dataset_path.exists():
        return f"{base} The dataset path does not exist."
    if dataset_path.is_file():
        return f"{base} Provide a CER `File*.txt.zip` archive or a normalized `readings.csv` path instead."

    metadata_path = dataset_path / "metadata.json"
    has_docs = (dataset_path / "documentation").is_dir()
    has_data_dir = (dataset_path / "data").is_dir()
    if metadata_path.exists() and has_docs and not has_data_dir:
        return (
            f"{base} This looks like an ISSDA metadata-only/public-documentation download. "
            "The restricted smart meter archives were not downloaded. "
            "Provide an authorized ISSDA Dataverse API key and rerun `download-cer`, or point `RAW_DATA_DIR` "
            "at a directory that already contains `File1.txt.zip` to `File6.txt.zip`."
        )
    if metadata_path.exists() and has_data_dir:
        return (
            f"{base} An ISSDA dataset layout was found, but the restricted smart meter archives are still missing. "
            "Expected `File1.txt.zip` to `File6.txt.zip` under the dataset tree."
        )
    return (
        f"{base} Expected one of: a directory containing `readings.csv`, a direct `readings.csv` path, "
        "or the original CER `File1.txt.zip` to `File6.txt.zip` trial files."
    )


def _discover_cer_allocation_file(dataset_root: Path) -> Path | None:
    candidates: list[Path] = []
    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".csv", ".tsv", ".tab", ".txt", ".xls", ".xlsx"}:
            continue
        if "allocation" in path.name.lower():
            candidates.append(path)
    return sorted(candidates)[0] if candidates else None


def _read_metadata_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    if suffix in {".xls", ".xlsx"}:
        try:
            return pd.read_excel(path)
        except ImportError:
            return pd.DataFrame()
    return pd.DataFrame()


def _load_cer_allocation_metadata(dataset_root: Path) -> pd.DataFrame:
    allocation_path = _discover_cer_allocation_file(dataset_root)
    if allocation_path is None:
        return pd.DataFrame()

    metadata = _read_metadata_table(allocation_path)
    if metadata.empty:
        return metadata

    rename_map: dict[str, str] = {}
    for canonical, aliases in CER_METADATA_ALIASES.items():
        resolved = _resolve_column(metadata.columns.tolist(), aliases)
        if resolved:
            rename_map[resolved] = canonical
    normalized = metadata.rename(columns=rename_map).copy()
    if "household_id" not in normalized.columns:
        normalized["household_id"] = normalized.iloc[:, 0]
    normalized = normalized.dropna(subset=["household_id"])
    normalized["household_id"] = normalized["household_id"].map(_normalize_household_id)
    return normalized


def _select_cer_households(metadata: pd.DataFrame, sample_households: int | None) -> list[str]:
    if metadata.empty:
        return []

    candidates = metadata
    if "code" in metadata.columns:
        code = metadata["code"].astype(str).str.strip()
        residential_mask = code.str.contains("res", case=False, na=False)
        if residential_mask.any():
            candidates = metadata.loc[residential_mask]

    household_ids = candidates["household_id"].drop_duplicates().tolist()
    return household_ids[:sample_households] if sample_households else household_ids


def _iter_cer_reading_chunks(reading_file: Path):
    compression = "zip" if reading_file.suffix.lower() == ".zip" else None
    yield from pd.read_csv(
        reading_file,
        header=None,
        names=["household_id", "interval_code", "consumption_kwh"],
        dtype={"household_id": "string", "interval_code": "string", "consumption_kwh": "string"},
        sep=r"[\s,;]+",
        engine="python",
        compression=compression,
        chunksize=CER_CHUNK_SIZE,
    )


def _decode_cer_interval_codes(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.extract(r"(\d+)")[0].str.zfill(5)
    day_code = pd.to_numeric(normalized.str[:3], errors="coerce")
    time_code = pd.to_numeric(normalized.str[3:], errors="coerce")
    valid_mask = day_code.notna() & time_code.between(1, CER_INTERVALS_PER_DAY)
    timestamps = pd.Series(
        pd.DatetimeIndex([pd.NaT] * len(values), tz="UTC"),
        index=values.index,
    )
    if valid_mask.any():
        offsets = pd.to_timedelta(day_code.loc[valid_mask] - 1, unit="D") + pd.to_timedelta(
            (time_code.loc[valid_mask] - 1) * 30,
            unit="min",
        )
        timestamps.loc[valid_mask] = CER_BASE_TIMESTAMP + offsets
    return pd.to_datetime(timestamps, utc=True, errors="coerce")


def _append_stage_rows(stage_dir: Path, household_id: str, frame: pd.DataFrame) -> None:
    stage_path = stage_dir / f"{household_id}.csv"
    frame.to_csv(stage_path, mode="a", header=not stage_path.exists(), index=False)


def _build_household_row_from_metadata(
    household_id: str,
    hourly: pd.DataFrame,
    meta_row: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    has_region = bool(meta_row.get("region_code")) and pd.notna(meta_row.get("region_code"))
    has_latitude = bool(meta_row.get("latitude")) and pd.notna(meta_row.get("latitude"))
    has_longitude = bool(meta_row.get("longitude")) and pd.notna(meta_row.get("longitude"))

    metadata_json = {
        "source_household_id": household_id,
        "records_hourly": len(hourly),
        "start_at": pd.to_datetime(hourly["timestamp"], utc=True).min().isoformat(),
        "end_at": pd.to_datetime(hourly["timestamp"], utc=True).max().isoformat(),
    }
    for key in ("code", "residential_stimulus", "residential_tariff", "sme_allocation"):
        value = meta_row.get(key)
        if value is not None and pd.notna(value):
            metadata_json[key] = str(value)

    household_row = {
        "household_code": household_id,
        "display_name": str(meta_row.get("display_name") or f"Household {household_id}"),
        "region_code": str(meta_row.get("region_code")).upper() if has_region else "IRELAND",
        "latitude": float(meta_row["latitude"]) if has_latitude else 53.3498,
        "longitude": float(meta_row["longitude"]) if has_longitude else -6.2603,
        "metadata_json": metadata_json,
    }
    return household_row, not (has_region and has_latitude and has_longitude)


def _import_cer_trial_dataset(dataset_path: Path, output_dir: Path, sample_households: int | None = None) -> ImportedDataset:
    dataset_root = dataset_path if dataset_path.is_dir() else dataset_path.parent
    reading_files = _discover_cer_reading_files(dataset_path)
    if not reading_files:
        raise FileNotFoundError(_missing_cer_readings_message(dataset_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = output_dir / "_stage"
    stage_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_cer_allocation_metadata(dataset_root)
    metadata_lookup = {
        row["household_id"]: row
        for row in metadata.to_dict(orient="records")
        if row.get("household_id")
    }
    selected_ids = _select_cer_households(metadata, sample_households)
    selected_set = set(selected_ids)

    for reading_file in reading_files:
        for chunk in _iter_cer_reading_chunks(reading_file):
            chunk["household_id"] = chunk["household_id"].map(_normalize_household_id)
            if sample_households and not selected_set:
                for household_id in chunk["household_id"].dropna().drop_duplicates().tolist():
                    if household_id in selected_set:
                        continue
                    selected_ids.append(household_id)
                    selected_set.add(household_id)
                    if len(selected_set) >= sample_households:
                        break

            if selected_set:
                chunk = chunk.loc[chunk["household_id"].isin(selected_set)]
                if chunk.empty:
                    continue

            chunk["timestamp"] = _decode_cer_interval_codes(chunk["interval_code"])
            chunk["consumption_kwh"] = pd.to_numeric(chunk["consumption_kwh"], errors="coerce")
            chunk = chunk.dropna(subset=["household_id", "timestamp", "consumption_kwh"])
            if chunk.empty:
                continue

            chunk["timestamp"] = chunk["timestamp"].dt.floor("1h")
            hourly = (
                chunk.groupby(["household_id", "timestamp"], as_index=False)["consumption_kwh"]
                .sum()
                .sort_values(["household_id", "timestamp"])
            )
            for household_id, household_frame in hourly.groupby("household_id", sort=False):
                _append_stage_rows(stage_dir, household_id, household_frame)

    households_dir = output_dir / "households"
    households_dir.mkdir(exist_ok=True)
    grouped: dict[str, pd.DataFrame] = {}
    household_rows: list[dict[str, Any]] = []
    all_starts: list[pd.Timestamp] = []
    all_ends: list[pd.Timestamp] = []
    fallback_weather_used = False

    for stage_path in sorted(stage_dir.glob("*.csv")):
        household_id = stage_path.stem
        hourly = pd.read_csv(stage_path)
        hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True, errors="coerce")
        hourly["consumption_kwh"] = pd.to_numeric(hourly["consumption_kwh"], errors="coerce")
        hourly = hourly.dropna(subset=["timestamp", "consumption_kwh"])
        hourly = (
            hourly.groupby("timestamp", as_index=False)["consumption_kwh"]
            .sum()
            .sort_values("timestamp")
            .set_index("timestamp")[["consumption_kwh"]]
            .resample("1h")
            .sum(min_count=1)
        )
        hourly = hourly.interpolate(limit=4).ffill().dropna().reset_index()
        if len(hourly) < MIN_HOURLY_RECORDS:
            stage_path.unlink(missing_ok=True)
            continue

        hourly.to_csv(households_dir / f"{household_id}.csv", index=False)
        meta_row = metadata_lookup.get(household_id, {})
        household_row, uses_fallback_weather = _build_household_row_from_metadata(household_id, hourly, meta_row)
        household_rows.append(household_row)
        fallback_weather_used = fallback_weather_used or uses_fallback_weather
        if sample_households is not None and len(grouped) <= max(sample_households, 32):
            grouped[household_id] = hourly.copy()
        all_starts.append(pd.to_datetime(hourly["timestamp"], utc=True).min())
        all_ends.append(pd.to_datetime(hourly["timestamp"], utc=True).max())
        stage_path.unlink(missing_ok=True)

    if stage_dir.exists():
        try:
            stage_dir.rmdir()
        except OSError:
            pass

    if not household_rows:
        raise ValueError("No eligible CER household series were produced from the supplied dataset")

    manifest = {
        "household_codes": [row["household_code"] for row in household_rows],
        "artifact_root": str(output_dir),
        "processed_households_dir": str(households_dir),
        "window_input_hours": 168,
        "window_output_hours": 24,
        "weather_ready": False,
        "fallback_weather_used": fallback_weather_used,
        "source_format": "cer-smart-meter-trial",
        "source_files": [str(path) for path in reading_files],
        "allocation_metadata_loaded": not metadata.empty,
        "start_at": min(all_starts).isoformat() if all_starts else None,
        "end_at": max(all_ends).isoformat() if all_ends else None,
        "created_at": datetime.now(UTC).isoformat(),
    }
    return ImportedDataset(households=household_rows, household_series=grouped, manifest=manifest)


def import_cer_dataset(dataset_path: Path, output_dir: Path, sample_households: int | None = None) -> ImportedDataset:
    if _compact_dataset_available(dataset_path):
        return _import_compact_dataset(dataset_path, output_dir, sample_households=sample_households)
    return _import_cer_trial_dataset(dataset_path, output_dir, sample_households=sample_households)


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    ie_holidays = holidays.country_holidays("IE")
    enriched = frame.copy()
    enriched["hour"] = ts.dt.hour
    enriched["day_of_week"] = ts.dt.dayofweek
    enriched["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    enriched["is_holiday"] = ts.dt.date.astype(str).isin({str(day) for day in ie_holidays}).astype(int)
    enriched["hour_sin"] = np.sin(2 * np.pi * enriched["hour"] / 24)
    enriched["hour_cos"] = np.cos(2 * np.pi * enriched["hour"] / 24)
    enriched["dow_sin"] = np.sin(2 * np.pi * enriched["day_of_week"] / 7)
    enriched["dow_cos"] = np.cos(2 * np.pi * enriched["day_of_week"] / 7)
    return enriched


def create_windows(
    frame: pd.DataFrame,
    *,
    input_hours: int = 168,
    forecast_hours: int = 24,
    feature_columns: list[str],
    target_column: str = "consumption_kwh",
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    values = frame[feature_columns].to_numpy(dtype=np.float32)
    targets = frame[target_column].to_numpy(dtype=np.float32)
    timestamps = pd.to_datetime(frame["timestamp"], utc=True).tolist()
    X: list[np.ndarray] = []
    y: list[np.ndarray] = []
    target_times: list[pd.Timestamp] = []
    max_start = len(frame) - input_hours - forecast_hours + 1
    for start in range(max_start):
        stop = start + input_hours
        forecast_stop = stop + forecast_hours
        X.append(values[start:stop])
        y.append(targets[stop:forecast_stop])
        target_times.append(pd.Timestamp(timestamps[stop]))
    return np.stack(X), np.stack(y), target_times
