from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from typing import Optional, List

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
# ME
# ---------------------------------------------------

@router.get("/me")
def get_me(user=Depends(get_current_user)):

    return {
        "id": user.id,
        "email": user.email,
        "favorite_brands": user.favorite_brands,
        "price_preference": user.price_preference
    }