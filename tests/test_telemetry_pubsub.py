"""
Telemetry Publisher/Subscriber and WebSocket Protocol Tests
Validates the in-memory pub/sub engine, strict WebSocket authentication,
and unified telemetry message streaming.
"""

import sys
import os
import json
import asyncio
import pytest

sys.path.insert(0, os.path.abspath("."))

from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.main import app
from backend.database.redis_db import InMemoryRedisFallback
from backend.utils.jwt_utils import create_access_token


def test_in_memory_redis_pubsub():
    """Verify in-memory pub/sub supports subscribe, publish, and queue listening."""
    async def _run():
        redis = InMemoryRedisFallback()
        pubsub = redis.pubsub()

        channel = "channel_task_999"
        await pubsub.subscribe(channel)

        # Publish message
        test_payload = json.dumps({"current_round": 1, "loss": 0.42})
        delivered = await redis.publish(channel, test_payload)
        assert delivered == 1

        # Listen for published message
        listener = pubsub.listen()
        message = await asyncio.wait_for(anext(listener), timeout=2.0)
        assert message is not None
        assert message["type"] == "message"
        assert message["channel"] == channel
        assert json.loads(message["data"]) == {"current_round": 1, "loss": 0.42}

        # Clean close
        await pubsub.close()

    asyncio.run(_run())


def test_in_memory_redis_kv_operations():
    """Verify key-value and pattern search operations in InMemoryRedisFallback."""
    async def _run():
        redis = InMemoryRedisFallback()
        await redis.setex("task_live:1", 60, "active")
        await redis.setex("task_live:2", 60, "active")
        await redis.setex("client_status:hospital_ny", 60, "online")

        val = await redis.get("task_live:1")
        assert val == "active"

        task_keys = await redis.keys("task_live:*")
        assert len(task_keys) == 2
        assert "task_live:1" in task_keys
        assert "task_live:2" in task_keys

        deleted = await redis.delete("task_live:1")
        assert deleted == 1
        assert await redis.get("task_live:1") is None

    asyncio.run(_run())


def test_websocket_unauthorized_rejected():
    """WebSocket connection without token or with invalid token must be rejected with 1008."""
    client = TestClient(app)

    # Missing token
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/1") as ws:
            pass
    assert exc_info.value.code == 1008

    # Invalid token
    with pytest.raises(WebSocketDisconnect) as exc_info_inv:
        with client.websocket_connect("/ws/1?token=invalid.jwt.token") as ws:
            pass
    assert exc_info_inv.value.code == 1008


def test_websocket_telemetry_protocol():
    """Verify valid token connects, sends ping, and receives protocol response."""
    client = TestClient(app)
    token = create_access_token({"sub": "researcher_1", "role": "researcher"})

    with client.websocket_connect(f"/ws/101?token={token}") as websocket:
        websocket.send_text("ping")
        data = websocket.receive_json()
        assert data["type"] == "pong"
        assert data["task_id"] == "101"
        assert data["status"] == "connected"
