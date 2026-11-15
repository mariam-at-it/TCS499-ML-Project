from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi import HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import create_token, decode_raw_token, hash_password, verify_password
from app.models import AuditLog, User, UserSession


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _record_audit(db: Session, *, action: str, entity_id: str, actor_user_id: int | None, details: dict | None = None) -> None:
    db.add(
        AuditLog(
            actor_user_id=actor_user_id,
            action=action,
            entity_type="auth",
            entity_id=entity_id,
            details=details or {},
        )
    )


def activate_user(db: Session, token: str, password: str) -> User:
    user = db.scalar(select(User).where(User.activation_token == token))
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Activation token not found")
    user.password_hash = hash_password(password)
    user.is_active = True
    user.activation_token = None
    user.activated_at = _utcnow()
    db.add(user)
    _record_audit(db, action="auth.activated", entity_id=str(user.id), actor_user_id=user.id)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> User:
    user = db.scalar(select(User).where(User.email == email))
    if user is None or user.password_hash is None or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User is not active")
    return user


def _insert_refresh_session(
    db: Session,
    *,
    user: User,
    refresh_jti: str,
    token_family: str,
    expires_at: datetime,
    ip_address: str | None,
    user_agent: str | None,
) -> None:
    db.add(
        UserSession(
            user_id=user.id,
            refresh_jti=refresh_jti,
            token_family=token_family,
            ip_address=ip_address,
            user_agent=user_agent,
            issued_at=_utcnow(),
            expires_at=expires_at,
            last_used_at=_utcnow(),
        )
    )


def issue_tokens(
    db: Session,
    user: User,
    *,
    ip_address: str | None = None,
    user_agent: str | None = None,
    token_family: str | None = None,
) -> dict[str, str]:
    settings = get_settings()
    access_jti = uuid4().hex
    refresh_jti = uuid4().hex
    token_family = token_family or uuid4().hex
    refresh_expires_at = _utcnow() + timedelta(minutes=settings.refresh_token_expire_minutes)
    access_token = create_token(
        subject=str(user.id),
        token_type="access",
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
        extra_claims={"role": user.role.value},
        jti=access_jti,
    )
    refresh_token = create_token(
        subject=str(user.id),
        token_type="refresh",
        expires_delta=timedelta(minutes=settings.refresh_token_expire_minutes),
        extra_claims={"role": user.role.value, "family": token_family},
        jti=refresh_jti,
    )
    _insert_refresh_session(
        db,
        user=user,
        refresh_jti=refresh_jti,
        token_family=token_family,
        expires_at=refresh_expires_at,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    _record_audit(
        db,
        action="auth.session_issued",
        entity_id=refresh_jti,
        actor_user_id=user.id,
        details={"token_family": token_family},
    )
    db.commit()
    return {"access_token": access_token, "refresh_token": refresh_token}


def _get_refresh_session(db: Session, refresh_jti: str) -> UserSession | None:
    return db.scalar(select(UserSession).where(UserSession.refresh_jti == refresh_jti))


def _revoke_token_family(db: Session, token_family: str) -> None:
    db.execute(
        update(UserSession)
        .where(UserSession.token_family == token_family, UserSession.revoked_at.is_(None))
        .values(revoked_at=_utcnow())
    )


def refresh_tokens(
    db: Session,
    refresh_token: str,
    *,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> dict[str, str]:
    payload = decode_raw_token(refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
    refresh_jti = str(payload.get("jti") or "")
    session = _get_refresh_session(db, refresh_jti)
    if session is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh session not found")
    if session.user_id != int(payload["sub"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token does not match the user session")
    if session.revoked_at is not None:
        _revoke_token_family(db, session.token_family)
        db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token has already been used")
    if _as_utc(session.expires_at) <= _utcnow():
        session.revoked_at = _utcnow()
        db.add(session)
        db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token expired")

    user = db.get(User, int(payload["sub"]))
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    session.revoked_at = _utcnow()
    next_refresh_jti = uuid4().hex
    session.replaced_by_jti = next_refresh_jti
    session.last_used_at = _utcnow()
    db.add(session)

    settings = get_settings()
    access_token = create_token(
        subject=str(user.id),
        token_type="access",
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
        extra_claims={"role": user.role.value},
    )
    new_refresh_token = create_token(
        subject=str(user.id),
        token_type="refresh",
        expires_delta=timedelta(minutes=settings.refresh_token_expire_minutes),
        extra_claims={"role": user.role.value, "family": session.token_family},
        jti=next_refresh_jti,
    )
    _insert_refresh_session(
        db,
        user=user,
        refresh_jti=next_refresh_jti,
        token_family=session.token_family,
        expires_at=_utcnow() + timedelta(minutes=settings.refresh_token_expire_minutes),
        ip_address=ip_address,
        user_agent=user_agent,
    )
    _record_audit(
        db,
        action="auth.session_rotated",
        entity_id=next_refresh_jti,
        actor_user_id=user.id,
        details={"previous_refresh_jti": refresh_jti, "token_family": session.token_family},
    )
    db.commit()
    return {"access_token": access_token, "refresh_token": new_refresh_token}


def revoke_refresh_token(db: Session, refresh_token: str | None) -> None:
    if not refresh_token:
        return
    try:
        payload = decode_raw_token(refresh_token)
    except HTTPException:
        return
    if payload.get("type") != "refresh":
        return
    refresh_jti = str(payload.get("jti") or "")
    session = _get_refresh_session(db, refresh_jti)
    if session is None or session.revoked_at is not None:
        return
    session.revoked_at = _utcnow()
    session.last_used_at = _utcnow()
    db.add(session)
    _record_audit(
        db,
        action="auth.session_revoked",
        entity_id=refresh_jti,
        actor_user_id=session.user_id,
        details={"token_family": session.token_family},
    )
    db.commit()
