"""
Structured Security & Privacy Audit Logger
FedMedShield Framework
Records non-sensitive security events, key rotations, participant authentication,
differential privacy budget checks, and aggregation events for compliance review.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger("FedMedShield.SecurityAudit")
AUDIT_LOG_FILE = os.getenv("AUDIT_LOG_FILE", "logs/security_audit.jsonl")


class SecurityAuditLogger:
    """Writes immutable structured audit events to disk and logger."""

    @staticmethod
    def log_event(
        event_type: str,
        participant_id: Optional[str] = None,
        task_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "INFO",
    ) -> Dict[str, Any]:
        """
        Record a structured security event.
        NEVER logs raw patient data, unnoised gradients, or private keys.
        """
        os.makedirs(os.path.dirname(AUDIT_LOG_FILE), exist_ok=True)
        now_utc = datetime.now(timezone.utc).isoformat()
        
        record = {
            "timestamp": now_utc,
            "event_type": event_type,
            "severity": severity,
            "participant_id": participant_id or "system",
            "task_id": str(task_id) if task_id else None,
            "details": details or {},
        }

        # Write to JSONL
        try:
            with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to append to audit log {AUDIT_LOG_FILE}: {e}")

        # Mirror to standard logger
        log_msg = f"[AUDIT-{event_type}] participant={participant_id} task={task_id} details={details}"
        if severity == "WARNING":
            logger.warning(log_msg)
        elif severity == "ERROR":
            logger.error(log_msg)
        else:
            logger.info(log_msg)

        return record

    @staticmethod
    def get_recent_events(limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent audit events for inspection."""
        if not os.path.exists(AUDIT_LOG_FILE):
            return []
        try:
            events = []
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line.strip()))
            return events[-limit:]
        except Exception as e:
            logger.error(f"Error reading audit log: {e}")
            return []
