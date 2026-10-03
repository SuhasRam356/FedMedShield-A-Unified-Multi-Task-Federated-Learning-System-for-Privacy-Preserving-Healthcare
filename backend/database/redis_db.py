"""
Redis Database and In-Memory Pub/Sub Manager
FedMedShield Framework - Real-Time Cache & WebSocket Metrics Pub/Sub
"""

import os
import logging
from typing import Optional, Any, Dict
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("FedMedShield.RedisDB")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class InMemoryRedisFallback:
    """Mock Redis client for local development when daemon is not running."""
    def __init__(self):
        self._store: Dict[str, str] = {}

    async def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self._store[key] = value
        return True

    async def delete(self, key: str) -> int:
        return 1 if self._store.pop(key, None) is not None else 0

    async def publish(self, channel: str, message: str) -> int:
        return 1

    async def ping(self) -> bool:
        return True

    async def close(self):
        pass


class RedisDBManager:
    client: Any = None
    is_live: bool = False

    @classmethod
    async def connect(cls):
        try:
            import redis.asyncio as redis
            r = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
            await r.ping()
            cls.client = r
            cls.is_live = True
            logger.info("Connected to Redis server successfully.")
        except Exception as e:
            logger.info("Redis daemon optional/not found (%s). Utilizing in-memory pub/sub cache.", e)
            cls.client = InMemoryRedisFallback()
            cls.is_live = False

    @classmethod
    async def disconnect(cls):
        if cls.client:
            try:
                await cls.client.close()
            except Exception:
                pass
            logger.info("Redis client disconnected.")

    @classmethod
    def get_client(cls):
        if cls.client is None:
            cls.client = InMemoryRedisFallback()
        return cls.client


async def get_redis_client():
    """FastAPI dependency for Redis operations."""
    return RedisDBManager.get_client()
