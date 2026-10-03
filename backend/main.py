"""
═══════════════════════════════════════════════════════════════
FedMedShield — Production Backend Entry Point (FastAPI)
Unified Multi-Task Federated Learning System for Privacy-Preserving Healthcare
Mounts all clinical routers, WebSockets, database pools, and CORS
═══════════════════════════════════════════════════════════════
"""

import logging
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure the backend directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Database connections
from backend.database.postgres_db import init_postgres_tables, close_postgres_connection
from backend.database.mongo_db import MongoDBManager
from backend.database.redis_db import RedisDBManager

# Routers (Legacy api/ & New routers/)
from api.websockets import router as ws_router
from routers.auth_router import router as auth_router
from routers.fl_router import router as fl_router
from routers.prediction_router import router as prediction_router
from routers.imaging_router import router as imaging_router
from routers.drug_router import router as drug_router
from routers.ids_router import router as ids_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FedMedShield.Main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Graceful asynchronous startup and shutdown lifecycle."""
    logger.info("Initializing FedMedShield Central Healthcare API Server...")
    
    # 1. Initialize Postgres / SQLite fallback
    try:
        await init_postgres_tables()
    except Exception as e:
        logger.warning("Postgres init encountered issue, running in resilience mode: %s", e)

    # 2. Initialize MongoDB / Document fallback
    try:
        await MongoDBManager.connect()
    except Exception as e:
        logger.warning("Mongo init notice: %s", e)

    # 3. Initialize Redis / In-memory pub-sub
    try:
        await RedisDBManager.connect()
    except Exception as e:
        logger.warning("Redis init notice: %s", e)

    logger.info("FedMedShield Core Services active & ready for federated telemetry.")
    yield

    # Shutdown
    logger.info("Shutting down FedMedShield services...")
    await close_postgres_connection()
    await MongoDBManager.disconnect()
    await RedisDBManager.disconnect()


app = FastAPI(
    title="FedMedShield API",
    description="Unified Multi-Task Federated Learning System for Privacy-Preserving Healthcare",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration
cors_env = os.getenv("CORS_ORIGINS")
allowed_origins = [orig.strip() for orig in cors_env.split(",") if orig.strip()] if cors_env else [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:3000",   # CRA dev server
    "http://127.0.0.1:5173",  # Alternative localhost
    "http://localhost:4173",   # Vite preview server
    "http://127.0.0.1:4173",  # Alternative preview
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Mount all Routers under /api
app.include_router(auth_router, prefix="/api")
app.include_router(fl_router, prefix="/api")
app.include_router(prediction_router, prefix="/api")
app.include_router(imaging_router, prefix="/api")
app.include_router(drug_router, prefix="/api")
app.include_router(ids_router, prefix="/api")
app.include_router(ws_router)  # WebSocket routes directly or under /ws


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "FedMedShield Core API",
        "version": "2.0.0",
        "disclaimer": "RESEARCH & DEMONSTRATION PROTOTYPE ONLY — Not validated for clinical diagnostic use, patient care, medical decision-making, or production security deployment.",
        "modules": [
            "Module 1: Multi-Task Clinical EHR (Sepsis & COVID-19 Demonstration Heuristic)",
            "Module 2: ResNet18 Medical Imaging (Tumor & Glaucoma Demonstration Heuristic)",
            "Module 3: Federated Drug Discovery & Bioactivity Screening (Heuristic Prototype)",
            "Module 4: Cybersecurity Intrusion Detection System (Demonstration Rules)"
        ],
        "privacy_mechanisms": [
            "Differential Privacy (Prototype Gaussian Clipping/Noise Simulation)",
            "Secure Aggregation (Pairwise Masking Demonstration Prototype)"
        ]
    }



@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mongo_connected": MongoDBManager.is_connected,
        "redis_live": RedisDBManager.is_live
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
