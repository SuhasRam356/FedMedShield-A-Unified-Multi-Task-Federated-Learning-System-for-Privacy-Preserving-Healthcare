"""
Pydantic Schemas for Federated Learning Orchestration and Status
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class FLTaskCreate(BaseModel):
    name: str = Field(..., example="EHR Sepsis Early Detection")
    task_type: str = Field(..., example="ehr")  # ehr, imaging, drug, ids
    target_rounds: int = Field(default=10, ge=1, le=100)
    num_clients: int = Field(default=4, ge=1, le=20)
    strategy: str = Field(default="FedAvg")
    dp_epsilon: float = Field(default=2.5)
    dp_delta: float = Field(default=1e-5)
    clip_norm: float = Field(default=1.0)


class FLTaskResponse(BaseModel):
    id: int
    name: str
    task_type: str
    status: str
    target_rounds: int
    current_round: int
    num_clients: int
    dp_epsilon: float
    created_at: Optional[str] = None


class FLMetricsUpdate(BaseModel):
    task_id: int
    current_round: int
    total_rounds: int
    current_loss: float
    current_accuracy: float
    progress_percent: float
    active_nodes: int
    dp_budget_consumed: float
    timestamp: str


class HospitalNodeInfo(BaseModel):
    client_id: str
    name: str
    region: str
    status: str  # online, training, offline
    data_size: str
    latency_ms: int
    last_seen: str
