from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.rate_limit import enforce_rate_limit
from app.db.session import get_db
from app.schemas.auth import (
    ActivateRequest,
    ActivationResponse,
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    TokenPair,
    UserResponse,
)
from app.services.auth import activate_user, authenticate_user, issue_tokens, refresh_tokens, revoke_refresh_token

router = APIRouter(prefix="/auth", tags=["auth"])


def _client_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


@router.post("/activate", response_model=ActivationResponse)
def activate(request: ActivateRequest, http_request: Request, db: Session = Depends(get_db)) -> ActivationResponse:
    enforce_rate_limit(http_request, scope="auth.activate", limit=10, window_seconds=300)
    user = activate_user(db, request.token, request.password)
    tokens = issue_tokens(
        db,
        user,
        ip_address=_client_ip(http_request),
        user_agent=http_request.headers.get("user-agent"),
    )
    return ActivationResponse(user=UserResponse.model_validate(user), tokens=TokenPair.model_validate(tokens))


@router.post("/login", response_model=TokenPair)
def login(request: LoginRequest, http_request: Request, db: Session = Depends(get_db)) -> TokenPair:
    enforce_rate_limit(http_request, scope="auth.login", limit=8, window_seconds=300)
    user = authenticate_user(db, request.email, request.password)
    return TokenPair.model_validate(
        issue_tokens(
            db,
            user,
            ip_address=_client_ip(http_request),
            user_agent=http_request.headers.get("user-agent"),
        )
    )


@router.post("/refresh", response_model=TokenPair)
def refresh(request: RefreshRequest, http_request: Request, db: Session = Depends(get_db)) -> TokenPair:
    enforce_rate_limit(http_request, scope="auth.refresh", limit=20, window_seconds=300)
    return TokenPair.model_validate(
        refresh_tokens(
            db,
            request.refresh_token,
            ip_address=_client_ip(http_request),
            user_agent=http_request.headers.get("user-agent"),
        )
    )


@router.post("/logout")
def logout(request: LogoutRequest | None = None, db: Session = Depends(get_db)) -> dict[str, str]:
    revoke_refresh_token(db, request.refresh_token if request else None)
    return {"status": "ok"}


@router.get("/me", response_model=UserResponse)
def me(user=Depends(get_current_user)) -> UserResponse:
    return UserResponse.model_validate(user)
