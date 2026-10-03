"""
═══════════════════════════════════════════════════════════════
FedMedShield — Resilient Database Configuration Facade
═══════════════════════════════════════════════════════════════
"""

import os
import logging
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

from backend.database.postgres_db import Base, engine, AsyncSessionLocal, get_postgres_session as get_pg_db
from backend.database.mongo_db import MongoDBManager, get_mongo_db
from backend.database.redis_db import RedisDBManager, get_redis_client as get_redis


async def init_mongo():
    await MongoDBManager.connect()


async def close_mongo():
    await MongoDBManager.disconnect()


async def init_redis():
    await RedisDBManager.connect()


async def close_redis():
    await RedisDBManager.disconnect()
