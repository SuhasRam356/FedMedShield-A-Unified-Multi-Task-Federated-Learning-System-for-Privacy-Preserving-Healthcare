"""
MongoDB Document Database Manager
FedMedShield Framework - Unstructured Healthcare Logs & Raw Audit Trail
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("FedMedShield.MongoDB")

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "fedmedshield_logs")


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


class MongoDBManager:
    client: Any = None
    db: Any = None
    is_connected: bool = False

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
            logger.info("MongoDB daemon optional/not found (%s). Utilizing fast in-memory document store.", e)
            cls.db = InMemoryFallbackDB()
            cls.is_connected = False

    @classmethod
    async def disconnect(cls):
        if cls.client:
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
