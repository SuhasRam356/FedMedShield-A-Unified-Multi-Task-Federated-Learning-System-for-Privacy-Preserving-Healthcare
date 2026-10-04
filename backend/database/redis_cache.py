"""
═══════════════════════════════════════════════════════════════
FedMedShield — Redis Cache & Real-Time Metrics
Provides helper functions to store and retrieve high-frequency
metrics (like current FL round, loss) for the frontend dashboard.
═══════════════════════════════════════════════════════════════
"""

import json
from typing import Dict, Any, List, Optional
import logging
from .db_config import get_redis

logger = logging.getLogger(__name__)

class MetricsCache:
    """Helper class for Redis operations related to live metrics."""
    
    PREFIX_TASK = "task_live:"
    PREFIX_CLIENT = "client_status:"

    @staticmethod
    async def set_task_progress(task_id: int, current_round: int, total_rounds: int, current_loss: float):
        """Update the real-time progress of an FL task."""
        redis = await get_redis()
        if not redis:
            return
            
        data = {
            "current_round": current_round,
            "total_rounds": total_rounds,
            "progress_percent": int((current_round / total_rounds) * 100),
            "current_loss": current_loss
        }
        
        # Cache expires in 24 hours
        await redis.setex(
            f"{MetricsCache.PREFIX_TASK}{task_id}", 
            86400, 
            json.dumps(data)
        )
        
        # Publish to a channel for WebSockets
        await redis.publish(f"channel_task_{task_id}", json.dumps(data))

    @staticmethod
    async def get_task_progress(task_id: int) -> Optional[Dict[str, Any]]:
        """Get the latest progress of an FL task."""
        redis = await get_redis()
        if not redis:
            return None
            
        data = await redis.get(f"{MetricsCache.PREFIX_TASK}{task_id}")
        if data:
            return json.loads(data)
        return None

    @staticmethod
    async def set_client_status(client_id: str, is_online: bool, current_task: Optional[int] = None):
        """Update whether a hospital node is online and computing."""
        redis = await get_redis()
        if not redis:
            return
            
        data = {
            "is_online": is_online,
            "current_task": current_task
        }
        
        # Heartbeat cache (expires in 60 seconds)
        # If the client doesn't ping within 60s, it's considered offline
        await redis.setex(
            f"{MetricsCache.PREFIX_CLIENT}{client_id}", 
            60, 
            json.dumps(data)
        )
        
        await redis.publish(f"channel_clients", json.dumps({"client_id": client_id, **data}))

    @staticmethod
    async def get_active_clients() -> List[str]:
        """Get a list of all currently connected hospitals."""
        redis = await get_redis()
        if not redis:
            return []
            
        # Scan for all client keys
        keys = await redis.keys(f"{MetricsCache.PREFIX_CLIENT}*")
        active_clients = []
        for key in keys:
            # Extract client_id from key
            client_id = key.replace(MetricsCache.PREFIX_CLIENT, "")
            active_clients.append(client_id)
            
        return active_clients
