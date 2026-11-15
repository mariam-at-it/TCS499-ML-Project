from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.core.security import validate_password_strength
from app.models import UserRole


class ActivateRequest(BaseModel):
    token: str
    password: str = Field(min_length=8)

    @field_validator("password")
    @classmethod
    def _validate_password(cls, value: str) -> str:
        return validate_password_strength(value)


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
            raise ValueError("A valid email address is required")
        return normalized


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str | None = None


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: UserRole
    household_id: int | None
    is_active: bool
    activated_at: datetime | None

    model_config = {"from_attributes": True}


class ActivationResponse(BaseModel):
    user: UserResponse
    tokens: TokenPair
