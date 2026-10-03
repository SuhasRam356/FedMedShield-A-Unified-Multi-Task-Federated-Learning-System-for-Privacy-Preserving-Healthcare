"""
API Contract Tests for FedMedShield Backend
Validates all endpoints invoked by the React frontend services.
"""

import sys
import os
import io
import pytest
from fastapi.testclient import TestClient

# Ensure repo root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.main import app

client = TestClient(app)

MANDATORY_DISCLAIMER_SUBSTRING = "DEMO / SYNTHETIC / NOT FOR CLINICAL OR SECURITY DECISIONS"


@pytest.fixture(scope="module")
def auth_token():
    """Obtain a valid JWT token via login endpoint."""
    response = client.post(
        "/api/auth/login",
        json={"username": "dr_smith", "password": "password123"},
    )
    assert response.status_code == 200, f"Login failed: {response.text}"
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    return data["access_token"]


@pytest.fixture(scope="module")
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}


def test_auth_me_contract(auth_headers):
    """GET /api/auth/me returns current user identity."""
    response = client.get("/api/auth/me", headers=auth_headers)
    assert response.status_code == 200
    user = response.json()
    assert "sub" in user or "username" in user or "id" in user


def test_unauthenticated_request_rejected():
    """Sensitive endpoints reject requests missing auth token."""
    response = client.get("/api/fl/tasks")
    assert response.status_code == 401


def test_ehr_prediction_contract(auth_headers):
    """POST /api/prediction/ehr returns risk scores and mandatory disclaimer."""
    payload = {
        "age": 62,
        "gender": "female",
        "heart_rate": 96.0,
        "systolic_bp": 118.0,
        "diastolic_bp": 74.0,
        "respiratory_rate": 20.0,
        "temperature": 38.1,
        "spo2": 95.0,
        "white_blood_cell": 12.8,
        "platelets": 190.0,
        "creatinine": 1.2,
        "crp": 38.0,
        "comorbidities": ["Hypertension"],
    }
    response = client.post("/api/prediction/ehr", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "sepsis_risk_score" in data
    assert "covid_outcome_prob" in data
    assert "disclaimer" in data
    assert MANDATORY_DISCLAIMER_SUBSTRING in data["disclaimer"]
    assert data.get("is_synthetic_simulation") is True


def test_prediction_history_contract(auth_headers):
    """GET /api/prediction/history returns list of historical predictions."""
    response = client.get("/api/prediction/history", headers=auth_headers)
    assert response.status_code == 200
    history = response.json()
    assert isinstance(history, list)
    assert len(history) > 0
    assert "patientId" in history[0]
    assert "primaryDiagnosis" in history[0]
    assert "predictions" in history[0]


def test_drug_screen_contract(auth_headers):
    """POST /api/drug/screen returns simulated affinity and mandatory disclaimer."""
    payload = {
        "compound_id": "CMP-TEST-01",
        "smiles_string": "CC(=O)Oc1ccccc1C(=O)O",
        "target_protein": "EGFR_HUMAN (Kinase Domain)",
    }
    response = client.post("/api/drug/screen", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "binding_affinity_score" in data
    assert "predicted_kd_nm" in data
    assert "disclaimer" in data
    assert MANDATORY_DISCLAIMER_SUBSTRING in data["disclaimer"]
    assert data.get("is_synthetic_simulation") is True


def test_ids_inspect_contract(auth_headers):
    """POST /api/ids/inspect returns alert structure and mandatory disclaimer."""
    payload = {
        "source_ip": "192.168.1.100",
        "dest_ip": "10.0.0.1",
        "protocol": "TCP",
        "packet_length": 1400,
        "duration_sec": 0.2,
        "flag": "SYN_SENT",
        "bytes_in": 120000,
        "bytes_out": 6000,
    }
    response = client.post("/api/ids/inspect", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "attack_detected" in data
    assert "severity" in data
    assert "mitigation_action" in data
    assert "disclaimer" in data
    assert MANDATORY_DISCLAIMER_SUBSTRING in data["disclaimer"]
    assert data.get("is_synthetic_simulation") is True


def test_imaging_analyze_contract(auth_headers):
    """POST /api/imaging/analyze handles multipart form data and disclaimer."""
    fake_img = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    files = {"file": ("scan.png", fake_img, "image/png")}
    data = {
        "modality": "Chest X-Ray",
        "target_condition": "Pulmonary Infiltration",
        "patient_id": "PT-TEST-001",
    }
    response = client.post(
        "/api/imaging/analyze",
        data=data,
        files=files,
        headers=auth_headers,
    )
    assert response.status_code == 200
    res = response.json()
    assert "prediction_label" in res
    assert "probability" in res
    assert "disclaimer" in res
    assert MANDATORY_DISCLAIMER_SUBSTRING in res["disclaimer"]
    assert res.get("is_synthetic_simulation") is True


def test_fl_management_endpoints(auth_headers):
    """Tests /api/fl/tasks, /status, /nodes, /start, and /stop lifecycle."""
    # List tasks
    res_tasks = client.get("/api/fl/tasks", headers=auth_headers)
    assert res_tasks.status_code == 200
    tasks = res_tasks.json()
    assert len(tasks) >= 2

    # Get status
    res_status = client.get("/api/fl/status", headers=auth_headers)
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert "status" in status_data
    assert "currentRound" in status_data

    # Get nodes
    res_nodes = client.get("/api/fl/nodes", headers=auth_headers)
    assert res_nodes.status_code == 200
    nodes = res_nodes.json()
    assert len(nodes) >= 4

    # Start and stop task 1
    res_start = client.post("/api/fl/tasks/1/start", headers=auth_headers)
    assert res_start.status_code == 200
    assert res_start.json()["task"]["status"] in ("training", "idle", "completed")

    res_stop = client.post("/api/fl/tasks/1/stop", headers=auth_headers)
    assert res_stop.status_code == 200
    assert res_stop.json()["task"]["status"] in ("idle", "stopped")
