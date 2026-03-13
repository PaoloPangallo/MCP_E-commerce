import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Very simple local memory fallback for dict-like interface if Redis is unavailable.
class _LocalCacheDict:
    def __init__(self):
        self._store = {}
        self._lock = threading.Lock()

    def setex(self, name: str, time_sec: int, value: str) -> None:
        with self._lock:
            self._store[name] = {
                "val": value,
                "expires_at": time.time() + time_sec
            }

    def get(self, name: str) -> Optional[str]:
        with self._lock:
            item = self._store.get(name)
            if not item:
                return None
            if time.time() > item["expires_at"]:
                del self._store[name]
                return None
            return item["val"]

    def delete(self, name: str) -> None:
        with self._lock:
            if name in self._store:
                del self._store[name]


class RedisManager:
    """Wrapper that tries to use Redis, but falls back to in-memory dict."""
    def __init__(self):
        self._redis = None
        self._local = None

        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("Successfully connected to Redis cache.")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis at {redis_url}: {e}. Falling back to in-memory cache.")
                self._redis = None
                self._local = _LocalCacheDict()
        else:
            logger.info("No REDIS_URL provided. Using in-memory cache fallback.")
            self._local = _LocalCacheDict()

    def set_json(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        try:
            val_str = json.dumps(value, ensure_ascii=False)
            if self._redis:
                self._redis.setex(key, ttl_seconds, val_str)
            else:
                self._local.setex(key, ttl_seconds, val_str)
        except Exception as e:
            logger.error(f"Cache write error for {key}: {e}")

    def get_json(self, key: str) -> Optional[Any]:
        try:
            if self._redis:
                val_str = self._redis.get(key)
            else:
                val_str = self._local.get(key)
                
            if val_str:
                return json.loads(val_str)
            return None
        except Exception as e:
            logger.error(f"Cache read error for {key}: {e}")
            return None

    def delete(self, key: str) -> None:
        try:
            if self._redis:
                self._redis.delete(key)
            else:
                self._local.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error for {key}: {e}")

redis_client = RedisManager()
