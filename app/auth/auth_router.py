from fastapi import APIRouter, Depends, HTTPException, Request
import logging
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from typing import Optional, List
import time
from collections import defaultdict

from app.auth.dependencies import get_current_user
from app.db.database import get_db
from app.models.user import User

from app.auth.password import (
    hash_password,
    verify_password,
    validate_password_strength,
)

from app.auth.jwt_handler import create_access_token


router = APIRouter(
    prefix="/auth",
    tags=["auth"]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Rate Limiting (Redis-backed)
# ---------------------------------------------------

_RATE_LIMIT_WINDOW = 60  # 1 minute
_RATE_LIMIT_MAX = 5     # max attempts per window

from app.db.redis import redis_client

def _check_rate_limit(key: str) -> None:
    """Raise 429 if too many requests from this key in the time window using Redis."""
    redis_key = f"rate_limit:{key}"
    
    try:
        # Increment and get the current count
        current_count = redis_client.client.incr(redis_key)
        
        # If it's the first hit, set the expiration
        if current_count == 1:
            redis_client.client.expire(redis_key, _RATE_LIMIT_WINDOW)
            
        if current_count > _RATE_LIMIT_MAX:
            raise HTTPException(
                status_code=429,
                detail="Troppi tentativi. Riprova tra un minuto."
            )
    except HTTPException:
        raise
    except Exception as e:
        # Fallback to allow request if Redis is down, but log it
        logger.warning("Redis rate limit check failed: %s", e)
        return


# ---------------------------------------------------
# Schemi
# ---------------------------------------------------

class RegisterRequest(BaseModel):

    email: EmailStr
    password: str

    favorite_brands: Optional[List[str]] = None
    price_preference: Optional[str] = None


class LoginRequest(BaseModel):

    email: EmailStr
    password: str


class AuthResponse(BaseModel):

    access_token: str
    token_type: str = "bearer"
    user_id: int


# ---------------------------------------------------
# REGISTER
# ---------------------------------------------------

@router.post("/register", response_model=AuthResponse)
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db)
):

    email = request.email.lower().strip()
    _check_rate_limit(f"register:{email}")

    existing = db.query(User).filter(User.email == email).first()

    if existing:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    validate_password_strength(request.password)

    user = User(
        email=email,
        password_hash=hash_password(request.password),
        favorite_brands=",".join(request.favorite_brands) if request.favorite_brands else None,
        price_preference=request.price_preference,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id)

    return AuthResponse(
        access_token=token,
        user_id=user.id
    )


# ---------------------------------------------------
# LOGIN
# ---------------------------------------------------

@router.post("/login", response_model=AuthResponse)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db)
):

    email = request.email.lower().strip()
    _check_rate_limit(f"login:{email}")

    user = db.query(User).filter(User.email == email).first()

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    if not verify_password(
        request.password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    token = create_access_token(user.id)

    return AuthResponse(
        access_token=token,
        user_id=user.id
    )


# ---------------------------------------------------
# OAUTH TOKEN (Swagger)
# ---------------------------------------------------

@router.post("/token", response_model=AuthResponse)
def token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):

    email = form_data.username.lower().strip()
    _check_rate_limit(f"token:{email}")

    user = db.query(User).filter(User.email == email).first()

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    if not verify_password(
        form_data.password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    token = create_access_token(user.id)

    return AuthResponse(
        access_token=token,
        user_id=user.id
    )


# ---------------------------------------------------
# ME / PREFERENCES
# ---------------------------------------------------

class CustomInstructionsUpdate(BaseModel):
    custom_instructions: Optional[str] = None


@router.get("/me")
def get_me(user=Depends(get_current_user)):

    return {
        "id": user.id,
        "email": user.email,
        "favorite_brands": user.favorite_brands,
        "price_preference": user.price_preference,
        "custom_instructions": user.custom_instructions
    }


@router.patch("/me/instructions")
def update_instructions(
    request: CustomInstructionsUpdate,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        user.custom_instructions = request.custom_instructions
        db.commit()
        db.refresh(user)
        return {"status": "success", "custom_instructions": user.custom_instructions}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
