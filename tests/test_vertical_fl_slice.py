"""
Vertical Federated Learning Slice Test (Module 4: IDS)
Tests a reproducible 2-client, 1-round FL training and aggregation cycle.
"""

import sys
import os
import pytest
import numpy as np
import torch

# Ensure repo root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fl_engine.client import HospitalClient
from modules.module4_ids.ids_model import IntrusionDetectionModel
from fl_engine.server import SecureFedAvgStrategy
from flwr.common import (
    FitRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


def test_vertical_fl_slice_ids_two_clients_one_round():
    """Validates 2 clients train on synthetic IDS traffic and aggregate successfully."""
    # 1. Initialize global model and extract reference parameters
    global_model = IntrusionDetectionModel()
    init_params = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    assert len(init_params) > 0, "Model state_dict returned no parameters"

    # 2. Instantiate 2 hospital clients
    client_a = HospitalClient(client_id="hospital-a")
    client_b = HospitalClient(client_id="hospital-b")

    # 3. Fit config: scalar configuration contract
    config = {
        "task": "ids",
        "current_round": 1,
        "epochs": 1,
        "fedprox_mu": 0.0,
        "use_dp": False,
        "use_secagg": False,
    }

    # 4. Execute local training round on each client
    params_a, num_a, metrics_a = client_a.fit(init_params, config)
    params_b, num_b, metrics_b = client_b.fit(init_params, config)

    assert len(params_a) == len(init_params), "Client A parameter count mismatch"
    assert len(params_b) == len(init_params), "Client B parameter count mismatch"
    assert num_a > 0, "Client A trained on 0 samples"
    assert num_b > 0, "Client B trained on 0 samples"

    # Verify no NaN or Inf values in returned weights
    for idx, p in enumerate(params_a):
        assert np.all(np.isfinite(p)), f"Client A weight {idx} contains NaN/Inf"
    for idx, p in enumerate(params_b):
        assert np.all(np.isfinite(p)), f"Client B weight {idx} contains NaN/Inf"

    # Verify that weights were actually updated by SGD
    a_diff = np.sum([np.sum(np.abs(p - init_p)) for p, init_p in zip(params_a, init_params)])
    b_diff = np.sum([np.sum(np.abs(p - init_p)) for p, init_p in zip(params_b, init_params)])
    assert a_diff > 1e-4, "Client A weights did not update during training"
    assert b_diff > 1e-4, "Client B weights did not update during training"

    # 5. Aggregate with strategy
    strategy = SecureFedAvgStrategy(
        task_name="ids",
        num_hospitals=2,
        min_fit_clients=2,
        min_available_clients=2,
        use_secagg=False,
    )

    fit_results = [
        (
            None,
            FitRes(
                status=Status(Code.OK, "Success"),
                parameters=ndarrays_to_parameters(params_a),
                num_examples=num_a,
                metrics=metrics_a,
            ),
        ),
        (
            None,
            FitRes(
                status=Status(Code.OK, "Success"),
                parameters=ndarrays_to_parameters(params_b),
                num_examples=num_b,
                metrics=metrics_b,
            ),
        ),
    ]

    agg_params, agg_metrics = strategy.aggregate_fit(
        server_round=1, results=fit_results, failures=[]
    )
    assert agg_params is not None, "Federated aggregation returned None"

    agg_ndarrays = parameters_to_ndarrays(agg_params)
    assert len(agg_ndarrays) == len(init_params), "Aggregated param count mismatch"

    # 6. Verify aggregated weights can be safely loaded and run forward inference
    fresh_model = IntrusionDetectionModel()
    state_keys = list(fresh_model.state_dict().keys())
    new_state_dict = {
        key: torch.tensor(agg_ndarrays[i]) for i, key in enumerate(state_keys)
    }
    fresh_model.load_state_dict(new_state_dict)
    fresh_model.eval()

    dummy_input = torch.randn(4, 41)
    with torch.no_grad():
        preds = fresh_model.predict(dummy_input)

    assert "results" in preds
    assert len(preds["results"]) == 4
    for r in preds["results"]:
        assert r["attack_type"] in IntrusionDetectionModel.ATTACK_CLASSES
        assert 0.0 <= r["confidence"] <= 1.0
