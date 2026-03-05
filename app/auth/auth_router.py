from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.db.database import get_db
from app.models.user import User

from app.auth.password import (
    hash_password,
    verify_password,
    validate_password_strength,
)
from app.auth.jwt_handler import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


# ---------------------------
# Schemi
# ---------------------------
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    favorite_brands: str | None = None
    price_preference: str | None = None

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# ---------------------------
# REGISTER
# ---------------------------
@router.post("/register", response_model=AuthResponse)
def register(request: RegisterRequest, db: Session = Depends(get_db)):

    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    try:
        validate_password_strength(request.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    user = User(
        email=request.email,
        password_hash=hash_password(request.password),
        favorite_brands=request.favorite_brands,
        price_preference=request.price_preference,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id)

    return AuthResponse(access_token=token, user_id=user.id)


# ---------------------------
# LOGIN (JSON - per frontend)
# ---------------------------
@router.post("/login", response_model=AuthResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):

    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(user.id)

    return AuthResponse(access_token=token, user_id=user.id)


# ---------------------------
# TOKEN (FORM - per Swagger OAuth2)
# ---------------------------
@router.post("/token", response_model=AuthResponse)
def token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Standard OAuth2 flow: username/password in form-data.
    Swagger UI usa questo per il pulsante "Authorize".
    """
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(user.id)
    return AuthResponse(access_token=token, user_id=user.id)


@router.get("/me")
def get_me(user = Depends(get_current_user)):

    return {
        "id": user.id,
        "email": user.email,
        "favorite_brands": user.favorite_brands,
        "price_preference": user.price_preference
    }