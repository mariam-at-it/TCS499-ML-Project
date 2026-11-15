from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class DatasetImportRequest(BaseModel):
    source_type: Literal["fixture", "cer", "smart-meters"] = Field(default="fixture")
    dataset_path: str | None = None
    profile: Literal["fixture", "cer-dev", "cer-full", "smart-meters"] = "fixture"
    sample_households: int | None = Field(default=None, ge=1)
    synthetic_days: int | None = Field(default=None, ge=30, le=1460)
    synthetic_seed: int | None = Field(default=None, ge=1)
    reset_existing: bool = False

    @model_validator(mode="after")
    def validate_profile(self) -> "DatasetImportRequest":
        if self.source_type == "fixture" and self.profile != "fixture":
            raise ValueError("Fixture imports must use the fixture profile")
        if self.source_type == "cer" and self.profile not in {"cer-dev", "cer-full"}:
            raise ValueError("CER imports must use the cer-dev or cer-full profile")
        if self.source_type == "smart-meters" and self.profile != "smart-meters":
            raise ValueError("Smart meter imports must use the smart-meters profile")
        return self


class WeatherSyncRequest(BaseModel):
    dataset_version_id: int
    household_limit: int | None = None


class TrainingRunConfig(BaseModel):
    dataset_version_id: int
    household_selector: list[int] | None = None
    client_sampling_rate: float = Field(default=1.0, ge=0.1, le=1.0)
    rounds: int = Field(default=5, ge=1, le=200)
    local_epochs: int = Field(default=2, ge=1, le=10)
    batch_size: int = Field(default=64, ge=1, le=256)
    learning_rate: float = Field(default=0.001, gt=0, le=1.0)
    secure_aggregation_enabled: bool = True
    secure_aggregation_degree: int = Field(default=2, ge=0, le=8)
    secure_aggregation_mask_scale: float = Field(default=4.0, gt=0.0, le=32.0)
    dp_enabled: bool = True
    noise_multiplier: float = Field(default=1.2, ge=0.0, le=20.0)
    clipping_norm: float = Field(default=1.0, gt=0.0, le=20.0)


class TrainingRunRequest(BaseModel):
    config: TrainingRunConfig


class ModelActionRequest(BaseModel):
    notes: str | None = None


class PublishModelRequest(BaseModel):
    model_version_id: int


class HouseholdAdminResponse(BaseModel):
    id: int
    household_code: str
    display_name: str
    region_code: str | None
    metadata_json: dict[str, Any]
    user_email: str | None
    activation_token: str | None
