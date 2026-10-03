"""
Federated Learning Task & Node Management Router
FedMedShield Framework - Multi-Task Orchestration API
"""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from backend.models.fl_models import FLTaskCreate, FLTaskResponse, FLMetricsUpdate, HospitalNodeInfo

router = APIRouter(prefix="/fl", tags=["Federated Learning"])

# In-memory tasks store
MOCK_TASKS: Dict[int, Dict[str, Any]] = {
    1: {
        "id": 1,
        "name": "Global Multi-Task EHR Cohort",
        "task_type": "ehr",
        "status": "idle",
        "target_rounds": 10,
        "current_round": 0,
        "num_clients": 4,
        "dp_epsilon": 2.5,
        "created_at": datetime.utcnow().isoformat()
    },
    2: {
        "id": 2,
        "name": "Tumor & Glaucoma ResNet50 Federation",
        "task_type": "imaging",
        "status": "idle",
        "target_rounds": 10,
        "current_round": 0,
        "num_clients": 4,
        "dp_epsilon": 2.5,
        "created_at": datetime.utcnow().isoformat()
    }
}

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
    return list(MOCK_TASKS.values())


@router.get("/tasks/{task_id}", response_model=FLTaskResponse)
async def get_task(task_id: int):
    task = MOCK_TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("/tasks", response_model=FLTaskResponse)
async def create_task(task_input: FLTaskCreate):
    new_id = max(MOCK_TASKS.keys(), default=0) + 1
    new_task = {
        "id": new_id,
        "name": task_input.name,
        "task_type": task_input.task_type,
        "status": "idle",
        "target_rounds": task_input.target_rounds,
        "current_round": 0,
        "num_clients": task_input.num_clients,
        "dp_epsilon": task_input.dp_epsilon,
        "created_at": datetime.utcnow().isoformat()
    }
    MOCK_TASKS[new_id] = new_task
    return new_task


@router.post("/tasks/{task_id}/start")
async def start_task(task_id: int):
    task = MOCK_TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task["status"] = "training"
    task["current_round"] = 1
    return {"message": f"Task {task_id} orchestration started", "task": task}


@router.post("/tasks/{task_id}/stop")
async def stop_task(task_id: int):
    task = MOCK_TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task["status"] = "idle"
    return {"message": f"Task {task_id} orchestration stopped", "task": task}


@router.get("/nodes", response_model=List[HospitalNodeInfo])
async def list_nodes():
    return MOCK_HOSPITAL_NODES
