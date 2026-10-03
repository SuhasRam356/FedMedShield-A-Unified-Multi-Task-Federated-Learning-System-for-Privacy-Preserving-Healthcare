"""
═══════════════════════════════════════════════════════════════
FedMedShield — MongoDB Document Database Manager
Features connection retries, serverSelectionTimeoutMS,
ping health checks, and in-memory fallback.
═══════════════════════════════════════════════════════════════
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

load_dotenv()
logger = logging.getLogger("FedMedShield.MongoDB")

MONGO_URL = os.getenv("MONGODB_URL", os.getenv("MONGO_URL", "mongodb://localhost:27017/"))
MONGO_DB_NAME = os.getenv("MONGODB_DB", os.getenv("MONGO_DB_NAME", "fedmedshield"))


class InMemoryFallbackCollection:
    """In-memory document mock for seamless zero-dependency developer experience."""
    def __init__(self, name: str):
        self.name = name
        self.documents: List[Dict[str, Any]] = []

    async def insert_one(self, doc: Dict[str, Any]):
        self.documents.append(doc)
        return type("InsertResult", (), {"inserted_id": str(len(self.documents))})()

    async def find(self, query: Optional[Dict[str, Any]] = None):
        return self

    async def to_list(self, length: int = 100):
        return self.documents[-length:]


class InMemoryFallbackDB:
    def __init__(self):
        self.collections: Dict[str, InMemoryFallbackCollection] = {}

    def __getitem__(self, item: str):
        if item not in self.collections:
            self.collections[item] = InMemoryFallbackCollection(item)
        return self.collections[item]


def get_mongo_client(max_retries: int = 3, allow_fallback: bool = True) -> Optional[MongoClient]:
    """
    Connect to MongoDB with retry logic and health check ping.
    If allow_fallback is True, gracefully falls back to in-memory store if daemon is offline.
    """
    mongo_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
    
    for attempt in range(max_retries):
        try:
            client = MongoClient(mongo_url, serverSelectionTimeoutMS=2000)
            client.admin.command('ping')  # Test connection
            logger.info("MongoDB connected on attempt %d", attempt + 1)
            print(f"[MongoDB] Connected successfully on attempt {attempt + 1}")
            return client
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            logger.warning("MongoDB attempt %d failed: %s", attempt + 1, e)
            print(f"[MongoDB] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
    
    if not allow_fallback:
        raise Exception("MongoDB connection failed after all retries!")
        
    logger.info("MongoDB daemon offline — operating in resilient fallback mode.")
    print("[MongoDB] Operating in resilient in-memory fallback mode.")
    return None


# Initialize module-level client and database
_raw_client = get_mongo_client(max_retries=1, allow_fallback=True)
if _raw_client:
    mongo_client = _raw_client
    db = mongo_client[MONGO_DB_NAME]
else:
    mongo_client = None
    db = InMemoryFallbackDB()


class MongoDBManager:
    client: Any = mongo_client
    db: Any = db
    is_connected: bool = mongo_client is not None

    @classmethod
    async def connect(cls):
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            cls.client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=2000)
            await cls.client.admin.command('ping')
            cls.db = cls.client[MONGO_DB_NAME]
            cls.is_connected = True
            logger.info("Connected to real MongoDB instance successfully.")
        except Exception as e:
            logger.info("MongoDB daemon not found (%s). Utilizing resilient in-memory document store.", e)
            cls.db = InMemoryFallbackDB()
            cls.is_connected = False

    @classmethod
    async def disconnect(cls):
        if cls.client and hasattr(cls.client, "close"):
            try:
                cls.client.close()
            except Exception:
                pass
            logger.info("MongoDB connection closed.")

    @classmethod
    def get_database(cls):
        if cls.db is None:
            cls.db = InMemoryFallbackDB()
        return cls.db


async def get_mongo_db():
    """FastAPI dependency for accessing Mongo collections."""
    return MongoDBManager.get_database()
