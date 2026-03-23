from passlib.context import CryptContext
import re


import asyncio
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=4)

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12
)


def hash_password(password: str) -> str:

    # bcrypt silently truncates at 72 bytes; enforce explicitly
    password = password[:72]

    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:

    plain_password = plain_password[:72]

    return pwd_context.verify(plain_password, hashed_password)


async def verify_password_async(plain: str, hashed: str) -> bool:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        verify_password,
        plain,
        hashed
    )


def validate_password_strength(password: str) -> None:
    """
    Minimal password validation.
    Raise ValueError if password is weak.
    """

    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")

    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must contain at least one uppercase letter.")

    if not re.search(r"[a-z]", password):
        raise ValueError("Password must contain at least one lowercase letter.")

    if not re.search(r"[0-9]", password):
        raise ValueError("Password must contain at least one number.")

    if len(password) > 128:
        raise ValueError("Password too long")