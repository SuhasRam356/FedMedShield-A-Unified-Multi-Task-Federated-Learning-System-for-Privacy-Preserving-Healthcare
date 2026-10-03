"""
═══════════════════════════════════════════════════════════════
FedMedShield — MongoDB Document Schemas
Uses Pydantic to validate unstructured/semi-structured data
before inserting it into MongoDB.
- FL Training Logs (Detailed metrics per round)
- Privacy Audits (Epsilon tracking per client)
- IDS Anomaly Reports
═══════════════════════════════════════════════════════════════
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class TrainingLogSchema(BaseModel):
    """
    Detailed log for a single FL round.
    Stored in Mongo because metric shapes differ vastly
    between EHR, Imaging, Drug, and IDS models.
    """
    task_id: int
    round_number: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # E.g. {"hospital-a": 500, "hospital-b": 1200}
    client_samples: Dict[str, int]
    
    # E.g. {"loss": 0.45, "accuracy": 0.88, "f1_score": 0.85}
    aggregated_metrics: Dict[str, float]
    
    # Any additional unstructured metadata
    meta_info: Optional[Dict[str, Any]] = None


class PrivacyAuditSchema(BaseModel):
    """
    Tracks Differential Privacy consumption.
    Critical for compliance and auditing.
    """
    task_id: int
    round_number: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # E.g. {"hospital-a": {"epsilon": 1.2, "delta": 1e-5}, ...}
    client_budgets: Dict[str, Dict[str, float]]
    
    # Global flag if any client exhausted their budget
    budget_exhausted: bool = False


class IDSAnomalyReportSchema(BaseModel):
    """
    If the IDS model detects a network attack on the FL infrastructure,
    it logs the unstructured packet metadata here.
    """
    client_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    attack_type: str
    confidence_score: float
    
    # Raw packet headers or metadata (JSON)
    packet_metadata: Dict[str, Any]
    
    action_taken: str = "logged" # e.g., "blocked", "logged"
