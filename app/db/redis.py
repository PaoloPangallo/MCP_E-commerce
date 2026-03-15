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

    def push_list(self, name: str, value: str, limit: int = 5) -> None:
        with self._lock:
            item = self._store.get(name)
            if not item or not isinstance(item.get("val"), list):
                self._store[name] = {
                    "val": [],
                    "expires_at": time.time() + 86400  # Default 24h
                }
                item = self._store[name]
            
            lst = item["val"]
            lst.insert(0, value)
            if len(lst) > limit:
                item["val"] = lst[:limit]
            
            # Reset expiration
            item["expires_at"] = time.time() + 86400

    def get_list(self, name: str) -> list:
        with self._lock:
            item = self._store.get(name)
            if not item:
                return []
            if time.time() > item["expires_at"]:
                del self._store[name]
                return []
            val = item["val"]
            return val if isinstance(val, list) else []


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

    def push_user_query(self, user_id: str, query: str, limit: int = 5) -> None:
        key = f"user_queries:{user_id}"
        try:
            if self._redis:
                # Usa una pipeline per eseguire LPUSH e LTRIM atomici
                pipe = self._redis.pipeline()
                pipe.lpush(key, query)
                pipe.ltrim(key, 0, limit - 1)
                pipe.expire(key, 86400) # Mantieni storia per 24h
                pipe.execute()
            else:
                self._local.push_list(key, query, limit=limit)
        except Exception as e:
            logger.error(f"Cache push_user_query error for {key}: {e}")

    def get_user_queries(self, user_id: str) -> list[str]:
        key = f"user_queries:{user_id}"
        try:
            if self._redis:
                return self._redis.lrange(key, 0, -1)
            else:
                return self._local.get_list(key)
        except Exception as e:
            logger.error(f"Cache get_user_queries error for {key}: {e}")
            return []

redis_client = RedisManager()
