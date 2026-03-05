from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from typing import Any, Dict, Optional

from app.auth.config import settings

class TokenError(Exception):
    pass

def create_access_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    payload: Dict[str, Any] = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "iss": settings.ISSUER,
        "aud": settings.AUDIENCE,
        "type": "access",
    }

    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def decode_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            audience=settings.AUDIENCE,
            issuer=settings.ISSUER,
        )
        return payload
    except JWTError as e:
        raise TokenError(str(e)) from e

def get_user_id_from_payload(payload: Dict[str, Any]) -> Optional[int]:
    sub = payload.get("sub")
    if not sub:
        return None
    try:
        return int(sub)
    except ValueError:
        return None