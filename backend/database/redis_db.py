"""
Redis Database and In-Memory Pub/Sub Manager
FedMedShield Framework - Real-Time Cache & WebSocket Metrics Pub/Sub
"""

import os
import fnmatch
import asyncio
import logging
from typing import Optional, Any, Dict, List, Set
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("FedMedShield.RedisDB")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_REQUIRED = os.getenv("REDIS_REQUIRED", "false").lower() in ("true", "1", "yes")


class InMemoryPubSub:
    """In-memory PubSub supporting subscribe, unsubscribe, and async listen iterator."""

    def __init__(self, fallback: "InMemoryRedisFallback"):
        self.fallback = fallback
        self.queue: asyncio.Queue = asyncio.Queue()
        self.subscribed_channels: Set[str] = set()
        self._closed = False

    async def subscribe(self, *channels: str):
        for ch in channels:
            self.subscribed_channels.add(ch)
            if ch not in self.fallback._subscribers:
                self.fallback._subscribers[ch] = set()
            self.fallback._subscribers[ch].add(self.queue)

    async def unsubscribe(self, *channels: str):
        target_channels = channels if channels else list(self.subscribed_channels)
        for ch in target_channels:
            self.subscribed_channels.discard(ch)
            if ch in self.fallback._subscribers:
                self.fallback._subscribers[ch].discard(self.queue)
                if not self.fallback._subscribers[ch]:
                    del self.fallback._subscribers[ch]

    async def listen(self):
        while not self._closed:
            try:
                msg = await self.queue.get()
                if msg is None:
                    break
                yield msg
            except asyncio.CancelledError:
                break

    async def close(self):
        self._closed = True
        await self.unsubscribe()
        await self.queue.put(None)


class InMemoryRedisFallback:
    """In-memory Redis client with real async pub/sub and key-value cache."""

    def __init__(self):
        self._store: Dict[str, str] = {}
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}

    async def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self._store[key] = str(value)
        return True

    async def setex(self, key: str, time: int, value: str) -> bool:
        self._store[key] = str(value)
        return True

    async def keys(self, pattern: str = "*") -> List[str]:
        return [k for k in self._store.keys() if fnmatch.fnmatch(k, pattern)]

    async def delete(self, key: str) -> int:
        return 1 if self._store.pop(key, None) is not None else 0

    async def publish(self, channel: str, message: str) -> int:
        count = 0
        if channel in self._subscribers:
            for q in list(self._subscribers[channel]):
                q.put_nowait({
                    "type": "message",
                    "channel": channel,
                    "data": message,
                })
                count += 1
        return count

    def pubsub(self) -> InMemoryPubSub:
        return InMemoryPubSub(self)

    async def ping(self) -> bool:
        return True

    async def close(self):
        self._store.clear()
        for ch, queues in list(self._subscribers.items()):
            for q in queues:
                q.put_nowait(None)
        self._subscribers.clear()


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
            if REDIS_REQUIRED:
                logger.error("Redis connection required (REDIS_REQUIRED=true) but failed: %s", e)
                raise RuntimeError(f"Strict Redis connection failed: {e}") from e
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
    """FastAPI dependency or awaitable for Redis client."""
    return RedisDBManager.get_client()
