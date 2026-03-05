from passlib.context import CryptContext

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def validate_password_strength(password: str) -> None:
    """
    Minimal checks (adjust as you want).
    Raises ValueError if password is weak.
    """
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")