import logging
import os
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


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

# --- Production safety guard ---
_env = os.getenv("ENV", "development").strip().lower()
if settings.SECRET_KEY == "dev_only_change_me":
    if _env in {"production", "prod"}:
        raise RuntimeError(
            "FATAL: JWT_SECRET_KEY is set to the insecure default value. "
            "Set a strong secret via the JWT_SECRET_KEY environment variable before running in production."
        )
    _logger.warning(
        "JWT_SECRET_KEY is using the default dev value. "
        "Set JWT_SECRET_KEY in your .env for production use."
    )