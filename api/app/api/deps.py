from collections.abc import Generator

from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import bearer_scheme, decode_token
from app.db.session import get_db
from app.models import User, UserRole


def get_current_user(db: Session = Depends(get_db), credentials=Depends(bearer_scheme)) -> User:
    payload = decode_token(credentials)
    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
    user = db.scalar(select(User).where(User.id == int(payload["sub"])))
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user


def require_household_or_admin(user: User = Depends(get_current_user)) -> User:
    return user


DbSession = Generator[Session, None, None]

