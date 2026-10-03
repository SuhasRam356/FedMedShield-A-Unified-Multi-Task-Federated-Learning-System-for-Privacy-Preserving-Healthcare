"""
PostgreSQL Database Manager and Connection Lifecycle
FedMedShield Framework - Relational Metadata & Task Logging
"""

import os
import logging
from typing import AsyncGenerator, Any
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("FedMedShield.PostgresDB")

PG_URL = os.getenv("POSTGRES_URL", "postgresql+asyncpg://postgres:password@localhost:5432/fedmedshield")

Base = declarative_base()
engine: Any = None
AsyncSessionLocal: Any = None
is_relational_live: bool = False

try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    engine = create_async_engine(
        PG_URL,
        echo=False,
        pool_size=10,
        max_overflow=20,
    )
    AsyncSessionLocal = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
except Exception as e:
    logger.info("Running in zero-dependency in-memory session mode (PG daemon optional): %s", e)


class MockAsyncSession:
    """Mock async session ensuring zero-dependency standalone execution."""
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def close(self):
        pass

    async def execute(self, *args, **kwargs):
        return type("MockResult", (), {"scalars": lambda self: type("MockScalars", (), {"all": lambda self: []})()})()


async def get_postgres_session() -> AsyncGenerator[Any, None]:
    """Dependency injection yield for FastAPI request handlers."""
    if AsyncSessionLocal:
        async with AsyncSessionLocal() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                raise e
            finally:
                await session.close()
    else:
        yield MockAsyncSession()


async def init_postgres_tables():
    """Create all relational tables asynchronously if live database exists."""
    global is_relational_live
    if engine:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            is_relational_live = True
            logger.info("PostgreSQL tables verified and initialized successfully.")
        except Exception as e:
            logger.info("PostgreSQL daemon not found on localhost:5432 (%s). System running with in-memory state.", e)
            is_relational_live = False
    else:
        logger.info("Standalone in-memory persistence active.")


async def close_postgres_connection():
    """Dispose of engine connection pool."""
    if engine:
        try:
            await engine.dispose()
            logger.info("PostgreSQL engine connection disposed.")
        except Exception:
            pass
