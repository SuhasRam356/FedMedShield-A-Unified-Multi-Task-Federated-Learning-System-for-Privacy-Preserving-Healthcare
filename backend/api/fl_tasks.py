"""
═══════════════════════════════════════════════════════════════
FedMedShield — FL Tasks API
REST Endpoints to orchestrate the Federated Learning network.
Allows starting, stopping, and viewing the status of FL tasks.
═══════════════════════════════════════════════════════════════
"""

import os
import subprocess
import asyncio
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel

from database.db_config import get_pg_db, get_redis
from database.postgres_models import FLTask, User
from api.auth import get_current_user

router = APIRouter(prefix="/api/tasks", tags=["Federated Learning"])

# Keep track of active orchestrator processes in memory
active_processes: Dict[int, subprocess.Popen] = {}

# --- Schemas ---
class TaskCreate(BaseModel):
    name: str
    task_type: str # ehr, imaging_tumor, imaging_glaucoma, drug, ids
    target_rounds: int = 5
    num_clients: int = 4

class TaskResponse(BaseModel):
    id: int
    name: str
    task_type: str
    status: str
    target_rounds: int
    owner_id: int

    class Config:
        orm_mode = True


# --- Background Worker ---
def run_orchestrator(task_id: int, task_type: str, rounds: int, clients: int):
    """
    Spawns the orchestrator.py script which handles Server + N Clients.
    This runs in a separate process.
    """
    import sys
    # Project root is one level up from backend
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root

    cmd = [
        sys.executable, "fl_engine/orchestrator.py",
        "--task", task_type,
        "--rounds", str(rounds),
        "--clients", str(clients)
    ]
    
    print(f"[API] Starting Orchestrator for Task {task_id} in {project_root}...")
    process = subprocess.Popen(cmd, env=env, cwd=project_root)
    active_processes[task_id] = process
    
    # Wait for completion (blocks the background thread)
    process.wait()
    
    # Cleanup when done
    if task_id in active_processes:
        del active_processes[task_id]
        
    print(f"[API] Orchestrator for Task {task_id} finished with code {process.returncode}")
    # Note: In a real production app, we would use an Async DB session here to mark the 
    # task as 'completed' in Postgres. For simplicity, we rely on the client to poll.


# --- Routes ---
@router.post("/", response_model=TaskResponse)
async def create_and_start_task(
    payload: TaskCreate, 
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_pg_db),
    current_user: User = Depends(get_current_user)
):
    """Creates a new FL Task and starts the federated network in the background."""
    
    # 1. Save to DB
    new_task = FLTask(
        name=payload.name,
        task_type=payload.task_type,
        target_rounds=payload.target_rounds,
        status="running",
        owner_id=current_user.id
    )
    db.add(new_task)
    await db.commit()
    await db.refresh(new_task)
    
    # 2. Update Redis initial state
    redis = get_redis()
    if redis:
        from database.redis_cache import MetricsCache
        await MetricsCache.set_task_progress(
            task_id=new_task.id, 
            current_round=0, 
            total_rounds=new_task.target_rounds, 
            current_loss=0.0
        )
    
    # 3. Start orchestrator script as a background task
    background_tasks.add_task(
        run_orchestrator, 
        task_id=new_task.id, 
        task_type=payload.task_type, 
        rounds=payload.target_rounds, 
        clients=payload.num_clients
    )
    
    return new_task


@router.get("/", response_model=List[TaskResponse])
async def list_tasks(db: AsyncSession = Depends(get_pg_db), current_user: User = Depends(get_current_user)):
    """List all FL tasks."""
    result = await db.execute(select(FLTask).order_by(FLTask.created_at.desc()))
    tasks = result.scalars().all()
    
    # Check if process is still running, update status dynamically if needed
    for task in tasks:
        if task.status == "running" and task.id not in active_processes:
            task.status = "completed"
            db.add(task)
            
    await db.commit()
    return tasks


@router.post("/{task_id}/stop")
async def stop_task(task_id: int, db: AsyncSession = Depends(get_pg_db), current_user: User = Depends(get_current_user)):
    """Forcefully terminate a running FL task."""
    result = await db.execute(select(FLTask).filter(FLTask.id == task_id))
    task = result.scalars().first()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
        
    if task_id in active_processes:
        process = active_processes[task_id]
        process.terminate()
        del active_processes[task_id]
        
    task.status = "failed" # or stopped
    db.add(task)
    await db.commit()
    
    return {"message": f"Task {task_id} stopped successfully."}
