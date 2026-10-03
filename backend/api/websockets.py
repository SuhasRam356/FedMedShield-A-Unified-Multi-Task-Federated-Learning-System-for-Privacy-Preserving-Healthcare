"""
═══════════════════════════════════════════════════════════════
FedMedShield — WebSockets API
Bridges Redis Pub/Sub directly to the React frontend.
Allows real-time streaming of FL round progress, privacy budget
exhaustion, and live metrics without polling.
═══════════════════════════════════════════════════════════════
"""

import asyncio
import json
import logging
from typing import List, Dict, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.database.db_config import get_redis
from backend.utils.jwt_utils import decode_access_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws", tags=["WebSockets"])

class ConnectionManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        # Maps task_id -> list of active connections listening to it
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)
        logger.info(f"WebSocket connected to task {task_id}. Total: {len(self.active_connections[task_id])}")

    def disconnect(self, websocket: WebSocket, task_id: str):
        if task_id in self.active_connections:
            if websocket in self.active_connections[task_id]:
                self.active_connections[task_id].remove(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]
        logger.info(f"WebSocket disconnected from task {task_id}.")

    async def broadcast(self, task_id: str, message: dict):
        if task_id in self.active_connections:
            for connection in self.active_connections[task_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending WS message: {e}")

manager = ConnectionManager()


async def redis_listener(task_id: str):
    """
    Background task that listens to a Redis channel and broadcasts 
    to all WebSockets connected to this task_id.
    """
    try:
        redis = await get_redis()
    except Exception:
        redis = None

    if not redis or not hasattr(redis, "pubsub"):
        logger.info(f"Live Redis pubsub not available for task {task_id}. Operating in memory mode.")
        return

    channel_name = f"channel_task_{task_id}"
    try:
        pubsub = redis.pubsub()
        await pubsub.subscribe(channel_name)
        logger.info(f"Started Redis listener for {channel_name}")

        async for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                await manager.broadcast(task_id, data)
    except asyncio.CancelledError:
        logger.info(f"Redis listener for {channel_name} cancelled.")
    except Exception as e:
        logger.error(f"Redis listener error: {e}")
    finally:
        try:
            await pubsub.unsubscribe(channel_name)
        except Exception:
            pass


@router.websocket("/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str, token: Optional[str] = None):
    """
    Protected endpoint for React UI to receive live metrics for a specific task.
    Requires valid JWT token query parameter: /ws/{task_id}?token={jwt}
    """
    user = decode_access_token(token) if token else None
    if not user:
        logger.warning(f"Unauthorized WebSocket connection attempt to task {task_id}")
        await websocket.close(code=1008)  # 1008: Policy Violation / Unauthorized
        return

    await manager.connect(websocket, task_id)
    listener_task = asyncio.create_task(redis_listener(task_id))
    
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, task_id)
        listener_task.cancel()
