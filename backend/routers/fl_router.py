"""
Federated Learning Task & Node Management Router
FedMedShield Framework - Multi-Task Orchestration API
"""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from backend.models.fl_models import FLTaskCreate, FLTaskResponse, FLMetricsUpdate, HospitalNodeInfo
from backend.middleware.auth_middleware import get_current_user

router = APIRouter(
    prefix="/fl",
    tags=["Federated Learning"],
    dependencies=[Depends(get_current_user)]
)

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
        "name": "Tumor & Glaucoma ResNet18 Federation",
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


FL_SYSTEM_STATE = {
    "currentRound": 0,
    "totalRounds": 20,
    "globalAccuracy": 0.720,
    "globalLoss": 0.650,
    "activeHospitals": 4,
    "totalHospitals": 4,
    "privacyBudget": {
        "epsilon": 0.5,
        "delta": 1e-5,
        "noiseScale": 0.01,
        "maxBudget": 10.0
    },
    "status": "idle"
}

_TRAINING_TASKS: Dict[int, asyncio.Task] = {}


async def _simulate_fl_rounds(task_id: int):
    task = MOCK_TASKS.get(task_id)
    if not task:
        return
    target_rounds = task.get("target_rounds", 10)
    FL_SYSTEM_STATE["status"] = "training"
    FL_SYSTEM_STATE["totalRounds"] = target_rounds

    try:
        for r in range(1, target_rounds + 1):
            if task.get("status") != "training":
                break
            task["current_round"] = r
            FL_SYSTEM_STATE["currentRound"] = r
            FL_SYSTEM_STATE["globalAccuracy"] = round(min(0.965, 0.72 + (r * 0.025)), 3)
            FL_SYSTEM_STATE["globalLoss"] = round(max(0.12, 0.65 - (r * 0.05)), 3)
            FL_SYSTEM_STATE["privacyBudget"]["epsilon"] = round(min(10.0, 0.5 + (r * 0.35)), 2)

            try:
                from backend.api.websockets import manager
                await manager.broadcast(str(task_id), {
                    "round": r,
                    "totalRounds": target_rounds,
                    "accuracy": FL_SYSTEM_STATE["globalAccuracy"],
                    "loss": FL_SYSTEM_STATE["globalLoss"],
                    "epsilon": FL_SYSTEM_STATE["privacyBudget"]["epsilon"],
                    "status": "training"
                })
            except Exception:
                pass

            await asyncio.sleep(2.0)

        if task.get("status") == "training":
            task["status"] = "completed"
            FL_SYSTEM_STATE["status"] = "completed"
    except asyncio.CancelledError:
        pass


@router.post("/tasks/{task_id}/start")
async def start_task(task_id: int):
    task = MOCK_TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task["status"] = "training"
    task["current_round"] = 1
    
    if task_id in _TRAINING_TASKS and not _TRAINING_TASKS[task_id].done():
        _TRAINING_TASKS[task_id].cancel()
    _TRAINING_TASKS[task_id] = asyncio.create_task(_simulate_fl_rounds(task_id))

    return {"message": f"Task {task_id} orchestration started", "task": task}


@router.post("/tasks/{task_id}/stop")
async def stop_task(task_id: int):
    task = MOCK_TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task["status"] = "idle"
    if task_id in _TRAINING_TASKS and not _TRAINING_TASKS[task_id].done():
        _TRAINING_TASKS[task_id].cancel()
    FL_SYSTEM_STATE["status"] = "idle"
    return {"message": f"Task {task_id} orchestration stopped", "task": task}


@router.get("/nodes", response_model=List[HospitalNodeInfo])
async def list_nodes():
    return MOCK_HOSPITAL_NODES


@router.get("/status")
async def get_fl_status():
    """Returns the current real-time FL network and aggregation status."""
    return FL_SYSTEM_STATE
