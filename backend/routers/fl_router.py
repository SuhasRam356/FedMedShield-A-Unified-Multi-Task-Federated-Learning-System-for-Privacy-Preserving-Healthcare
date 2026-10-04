"""
Federated Learning Task & Node Management Router
FedMedShield Framework - Multi-Task Orchestration API
"""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

import json
import logging

from backend.models.fl_models import FLTaskCreate, FLTaskResponse, FLMetricsUpdate, HospitalNodeInfo
from backend.middleware.auth_middleware import get_current_user
from backend.database.db_config import get_redis

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/fl",
    tags=["Federated Learning"],
    dependencies=[Depends(get_current_user)]
)

from fl_engine.orchestrator import orchestrator
from backend.api.websockets import manager


async def _broadcast_telemetry(task_id: str, payload: dict):
    """Broadcasts to active WebSockets and publishes to Redis pub/sub channel."""
    await manager.broadcast(task_id, payload)
    try:
        redis = await get_redis()
        if redis and hasattr(redis, "publish"):
            await redis.publish(f"channel_task_{task_id}", json.dumps(payload))
    except Exception as e:
        logger.debug(f"Redis publish notice: {e}")

MOCK_HOSPITAL_NODES: List[Dict[str, Any]] = [
    {
        "client_id": "hospital_ny",
        "name": "General Hospital - New York",
        "region": "US-East",
        "status": "online",
        "data_size": "45.2 GB (EHR & Imaging)",
        "latency_ms": 12,
        "last_seen": "Just now"
    },
    {
        "client_id": "hospital_chicago",
        "name": "University Medical Center - Chicago",
        "region": "US-Central",
        "status": "online",
        "data_size": "102.8 GB (Multi-Modal)",
        "latency_ms": 24,
        "last_seen": "Just now"
    },
    {
        "client_id": "hospital_sf",
        "name": "VA Medical Center - San Francisco",
        "region": "US-West",
        "status": "online",
        "data_size": "88.1 GB (Drug & Genomics)",
        "latency_ms": 45,
        "last_seen": "Just now"
    },
    {
        "client_id": "hospital_austin",
        "name": "Community Health Network - Austin",
        "region": "US-South",
        "status": "online",
        "data_size": "12.4 GB (EHR Clinical)",
        "latency_ms": 31,
        "last_seen": "Just now"
    }
]


@router.get("/tasks", response_model=List[FLTaskResponse])
async def list_tasks():
    return orchestrator.list_tasks()


@router.get("/tasks/{task_id}", response_model=FLTaskResponse)
async def get_task(task_id: int):
    task = orchestrator.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("/tasks", response_model=FLTaskResponse)
async def create_task(task_input: FLTaskCreate):
    data = task_input.model_dump() if hasattr(task_input, "model_dump") else task_input.dict()
    return orchestrator.create_task(data)


@router.post("/tasks/{task_id}/start")
async def start_task(task_id: int):
    task = orchestrator.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    updated_task = orchestrator.start_task(task_id, broadcast_fn=_broadcast_telemetry)
    return {"message": f"Task {task_id} orchestration started", "task": updated_task}


@router.post("/tasks/{task_id}/stop")
async def stop_task(task_id: int):
    task = orchestrator.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    updated_task = orchestrator.stop_task(task_id)
    return {"message": f"Task {task_id} orchestration stopped", "task": updated_task}


@router.get("/nodes", response_model=List[HospitalNodeInfo])
async def list_nodes():
    return MOCK_HOSPITAL_NODES


@router.get("/status")
async def get_fl_status():
    """Returns the current real-time FL network and aggregation status."""
    return orchestrator.get_status()
