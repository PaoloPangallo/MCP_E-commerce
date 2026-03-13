from fastapi import Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import Optional

from app.db.database import get_db
from app.models.user import User

from app.auth.jwt_handler import (
    decode_token,
    get_user_id_from_payload,
    TokenError
)


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/token",
    auto_error=False
)


def get_optional_user(
    token_header: Optional[str] = Depends(oauth2_scheme),
    token_query: Optional[str] = Query(None, alias="token"),
    db: Session = Depends(get_db),
):
    token = token_header or token_query
    
    if not token:
        return None

    try:
        payload = decode_token(token)
        user_id = get_user_id_from_payload(payload)

        if not user_id:
            return None

        user = db.query(User).filter(User.id == user_id).first()
        return user

    except TokenError:
        return None
    except Exception:
        return None


def get_current_user(
    token_header: Optional[str] = Depends(oauth2_scheme),
    token_query: Optional[str] = Query(None, alias="token"),
    db: Session = Depends(get_db),
):
    token = token_header or token_query
    
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated"
        )

    try:

        payload = decode_token(token)

        user_id = get_user_id_from_payload(payload)

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )

        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            raise HTTPException(
                status_code=401,
                detail="User not found"
            )

        return user

    except TokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )