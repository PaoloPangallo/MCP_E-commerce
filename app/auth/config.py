import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AuthSettings:

    # MUST be set in production
    SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev_only_change_me")

    ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")

    # token lifetime (1h default)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("JWT_EXPIRE_MINUTES", "60")
    )

    # metadata (optional but recommended)
    ISSUER: str = os.getenv("JWT_ISSUER", "mcp-ecommerce")
    AUDIENCE: str = os.getenv("JWT_AUDIENCE", "mcp-ecommerce-client")


settings = AuthSettings()