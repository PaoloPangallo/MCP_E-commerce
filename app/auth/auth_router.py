from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from typing import Optional, List
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

from app.auth.dependencies import get_current_user
from app.db.database import get_db
from app.models.user import User

from app.auth.password import (
    hash_password,
    verify_password,
    verify_password_async,
    validate_password_strength,
)

from app.auth.jwt_handler import create_access_token


router = APIRouter(
    prefix="/auth",
    tags=["auth"]
)


# ---------------------------------------------------
# Rate Limiting
# ---------------------------------------------------

_RATE_LIMIT_WINDOW = 60.0  # 1 minute
_RATE_LIMIT_MAX = 5  # max attempts per window
_rate_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(key: str) -> None:
    """Raise 429 if too many requests from this key in the time window."""
    now = time.time()
    bucket = _rate_store[key]
    # Purge old entries
    _rate_store[key] = [t for t in bucket if now - t < _RATE_LIMIT_WINDOW]
    if len(_rate_store[key]) >= _RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail="Troppi tentativi. Riprova tra un minuto."
        )
    _rate_store[key].append(now)


# ---------------------------------------------------
# Schemi
# ---------------------------------------------------

class RegisterRequest(BaseModel):

    email: EmailStr
    password: str

    favorite_brands: Optional[List[str]] = None
    price_preference: Optional[str] = None
    theme: Optional[str] = "light"
    conversation_tone: Optional[str] = "neutral"


class LoginRequest(BaseModel):

    email: EmailStr
    password: str


class AuthResponse(BaseModel):

    access_token: str
    token_type: str = "bearer"
    user_id: int
    email: str
    favorite_brands: Optional[str] = None
    price_preference: Optional[str] = None
    custom_instructions: Optional[str] = None
    theme: Optional[str] = "light"
    conversation_tone: Optional[str] = "neutral"


# ---------------------------------------------------
# REGISTER
# ---------------------------------------------------

@router.post("/register", response_model=AuthResponse)
async def register(
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
        theme=request.theme,
        conversation_tone=request.conversation_tone,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id)

    return AuthResponse(
        access_token=token,
        user_id=user.id,
        email=user.email,
        favorite_brands=user.favorite_brands,
        price_preference=user.price_preference,
        custom_instructions=user.custom_instructions,
        theme=user.theme,
        conversation_tone=user.conversation_tone,
    )


# ---------------------------------------------------
# LOGIN
# ---------------------------------------------------

@router.post("/login", response_model=AuthResponse)
async def login(
    request: LoginRequest,
    db: Session = Depends(get_db)
):

    email = request.email.lower().strip()
    _check_rate_limit(f"login:{email}")

    t0 = time.perf_counter()
    user = db.query(User).filter(User.email == email).first()
    t1 = time.perf_counter()
    print(f"[LOGIN] DB query: {t1-t0:.3f}s")

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    t2 = time.perf_counter()
    valid = await verify_password_async(
        request.password,
        user.password_hash
    )
    t3 = time.perf_counter()
    print(f"[LOGIN] bcrypt verify: {t3-t2:.3f}s")

    if not valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    t4 = time.perf_counter()
    token = create_access_token(user.id)
    t5 = time.perf_counter()
    print(f"[LOGIN] JWT create: {t5-t4:.3f}s")
    print(f"[LOGIN] TOTAL: {t5-t0:.3f}s")

    return AuthResponse(
        access_token=token,
        user_id=user.id,
        email=user.email,
        favorite_brands=user.favorite_brands,
        price_preference=user.price_preference,
        custom_instructions=user.custom_instructions,
        theme=user.theme,
        conversation_tone=user.conversation_tone,
    )


# ---------------------------------------------------
# OAUTH TOKEN (Swagger)
# ---------------------------------------------------

@router.post("/token", response_model=AuthResponse)
async def token(
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

    if not await verify_password_async(
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
        user_id=user.id,
        email=user.email,
        favorite_brands=user.favorite_brands,
        price_preference=user.price_preference,
        custom_instructions=user.custom_instructions,
        theme=user.theme,
        conversation_tone=user.conversation_tone,
    )


# ---------------------------------------------------
# ME / PREFERENCES
# ---------------------------------------------------

class UserResponse(BaseModel):
    id: int
    email: str
    favorite_brands: Optional[str] = None
    price_preference: Optional[str] = None
    custom_instructions: Optional[str] = None
    theme: Optional[str] = "light"
    conversation_tone: Optional[str] = "neutral"

@router.get("/me", response_model=UserResponse)
def get_me(user: User = Depends(get_current_user)):
    return UserResponse(
        id=user.id,
        email=user.email,
        favorite_brands=user.favorite_brands,
        price_preference=user.price_preference,
        custom_instructions=user.custom_instructions,
        theme=user.theme,
        conversation_tone=user.conversation_tone,
    )

class UserPreferencesUpdate(BaseModel):
    theme: Optional[str] = None
    conversation_tone: Optional[str] = None
    custom_instructions: Optional[str] = None
    favorite_brands: Optional[str] = None
    price_preference: Optional[str] = None

@router.patch("/me/preferences")
def update_preferences(
    request: UserPreferencesUpdate,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        if request.theme is not None:
            user.theme = request.theme
        if request.conversation_tone is not None:
            user.conversation_tone = request.conversation_tone
        if request.custom_instructions is not None:
            user.custom_instructions = request.custom_instructions
        if request.favorite_brands is not None:
            user.favorite_brands = request.favorite_brands
        if request.price_preference is not None:
            user.price_preference = request.price_preference
        
        db.commit()
        db.refresh(user)
        return {
            "status": "success", 
            "theme": user.theme,
            "conversation_tone": user.conversation_tone,
            "custom_instructions": user.custom_instructions,
            "favorite_brands": user.favorite_brands,
            "price_preference": user.price_preference
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

class InstructionsUpdate(BaseModel):
    custom_instructions: str

@router.patch("/me/instructions")
def update_instructions(
    request: InstructionsUpdate,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        user.custom_instructions = request.custom_instructions
        db.commit()
        db.refresh(user)
        return {
            "status": "success", 
            "custom_instructions": user.custom_instructions
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

class RecoverRequest(BaseModel):
    email: str

@router.post("/recover-password")
async def recover_password(request: RecoverRequest, db: Session = Depends(get_db)):
    # Mock semplice per evitare 404 in fase di login se l'utente clicca su recupero
    user = db.query(User).filter(User.email == request.email.lower().strip()).first()
    if not user:
         return {"status": "error", "message": "Se l'utente esiste, un'email è stata inviata."}
    return {"status": "success", "message": "Se l'utente esiste, un'email è stata inviata."}


# ---------------------------------------------------
# EBAY ACCOUNT INFO (real user token)
# ---------------------------------------------------

@router.get("/ebay/me")
async def get_ebay_identity():
    """
    Returns the authenticated eBay user's public profile info using the
    EBAY_USER_TOKEN from the .env file.
    This does NOT require our own JWT — it's a server-side call.
    """
    from app.services.ebay_user import get_ebay_user_info, get_ebay_watchlist
    from app.config.settings import settings

    logger.info("Accesso endpoint /auth/ebay/me | Token presente: %s", bool(settings.EBAY_USER_TOKEN))

    if not settings.EBAY_USER_TOKEN:
        raise HTTPException(status_code=503, detail="EBAY_USER_TOKEN non configurato.")

    user_info = await get_ebay_user_info()
    if not user_info:
        raise HTTPException(status_code=502, detail="Impossibile recuperare le info da eBay. Il token potrebbe essere scaduto.")

    # Also fetch watchlist count for the badge
    try:
        watchlist = await get_ebay_watchlist(entries_per_page=1)
        watchlist_count = len(watchlist)  # partial count from first page
    except Exception:
        watchlist_count = None

    return {
        "username": user_info.get("username"),
        "feedback_score": user_info.get("feedback_score"),
        "status": user_info.get("status"),
        "watchlist_items": watchlist_count,
        "token_configured": True,
    }

