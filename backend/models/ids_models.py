"""
Pydantic Schemas for Network Intrusion Detection (IDS) & Healthcare Cyber Defense
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class NetworkFlowPacket(BaseModel):
    source_ip: str = Field(..., example="192.168.1.105")
    dest_ip: str = Field(..., example="10.0.0.1")
    protocol: str = Field(..., example="TCP")
    packet_length: int = Field(..., example=1420)
    duration_sec: float = Field(..., example=0.45)
    flag: str = Field(..., example="SYN_SENT")
    bytes_in: int = Field(..., example=5400)
    bytes_out: int = Field(..., example=120)


class IntrusionDetectionAlert(BaseModel):
    alert_id: str
    timestamp: str
    attack_detected: bool
    attack_type: str  # Benign, DDoS, PortScan, BruteForce, DataExfiltration
    severity: str  # Low, Medium, High, Critical
    confidence: float
    affected_node: str
    mitigation_action: str
    disclaimer: str = "RESEARCH DEMONSTRATION ONLY — Simulated detection rule. No active network mitigation, firewall rule, or certificate revocation was executed."
    is_synthetic_simulation: bool = True

