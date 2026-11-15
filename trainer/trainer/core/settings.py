from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    trainer_database_url: str = "sqlite:///./trainer.db"
    app_profile: str = "fixture"
    raw_data_dir: Path = Path("../data/raw")
    fixture_data_dir: Path = Path("../data/fixture")
    smart_meter_data_dir: Path = Path("../data/artifacts/datasets/smart-meters")
    artifacts_dir: Path = Path("../data/artifacts")
    synthetic_dataset_households: int = 96
    synthetic_dataset_days: int = 365
    synthetic_dataset_seed: int = 42
    openmeteo_base_url: str = "https://archive-api.open-meteo.com/v1/archive"
    openmeteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    trainer_poll_interval_seconds: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()
