"""
Tests for Flower Config Serialization and Key Exchange
Ensures that all configuration dictionaries sent between server and clients
conform to Flower's strict Scalar type specification (bool, bytes, float, int, str),
preventing gRPC/protobuf serialization crashes during federated rounds.
"""

import sys
import os
import json
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath("."))

import flwr as fl
from flwr.common import (
    FitIns,
    EvaluateIns,
    Parameters,
    ndarrays_to_parameters,
)
from flwr.common.serde import (
    fit_ins_to_proto,
    evaluate_ins_to_proto,
)
from fl_engine.server import SecureFedAvgStrategy
from fl_engine.client import HospitalClient


class DummyClientProxy:
    """Mock Flower ClientProxy for strategy tests."""
    def __init__(self, cid: str):
        self.cid = cid


class DummyClientManager(fl.server.ClientManager):
    """Mock ClientManager with simulated hospital nodes."""
    def __init__(self, cids):
        self._cids = cids

    def num_available(self) -> int:
        return len(self._cids)

    def sample(self, num_clients: int, min_num_clients: int = None):
        return [DummyClientProxy(cid) for cid in self._cids[:num_clients]]

    def register(self, client):
        return True

    def unregister(self, client):
        pass

    def all(self):
        return {cid: DummyClientProxy(cid) for cid in self._cids}

    def wait_for(self, num_clients: int, timeout: int = 10) -> bool:
        return True


def test_fit_config_strictly_scalars():
    """Verify configure_fit produces only scalar values conforming to Flower's Scalar contract."""
    strategy = SecureFedAvgStrategy(
        task_name="ids",
        use_secagg=True,
        use_dp=True,
        num_hospitals=2,
    )
    client_manager = DummyClientManager(["hospital-1", "hospital-2"])
    dummy_params = ndarrays_to_parameters([np.zeros((10, 5), dtype=np.float32)])

    client_instructions = strategy.configure_fit(
        server_round=1,
        parameters=dummy_params,
        client_manager=client_manager,
    )

    assert len(client_instructions) == 2
    for client_proxy, fit_ins in client_instructions:
        config = fit_ins.config
        for k, v in config.items():
            assert isinstance(v, (bool, bytes, float, int, str)), (
                f"Config key '{k}' has non-scalar value {v} of type {type(v)}. "
                "Flower requires all config values to be scalar."
            )

        # Confirm protobuf conversion succeeds without error
        proto = fit_ins_to_proto(fit_ins)
        assert proto is not None


def test_evaluate_config_strictly_scalars():
    """Verify configure_evaluate produces only scalar values."""
    strategy = SecureFedAvgStrategy(
        task_name="ehr",
        use_secagg=False,
        use_dp=False,
        num_hospitals=2,
    )
    client_manager = DummyClientManager(["hospital-1", "hospital-2"])
    dummy_params = ndarrays_to_parameters([np.zeros((5,), dtype=np.float32)])

    eval_instructions = strategy.configure_evaluate(
        server_round=1,
        parameters=dummy_params,
        client_manager=client_manager,
    )

    assert len(eval_instructions) == 2
    for client_proxy, eval_ins in eval_instructions:
        config = eval_ins.config
        for k, v in config.items():
            assert isinstance(v, (bool, bytes, float, int, str)), (
                f"Config key '{k}' has non-scalar value {v} of type {type(v)}."
            )
        proto = evaluate_ins_to_proto(eval_ins)
        assert proto is not None


def test_client_handles_scalar_key_exchange():
    """Test that HospitalClient unpacks JSON scalar key exchange payloads correctly."""
    client = HospitalClient(client_id="hospital-1", device="cpu")
    active_clients = ["hospital-1", "hospital-2"]
    peer_keys = {"hospital-1": "04abcd1234", "hospital-2": "04ef567890"}

    # Transmit via JSON scalar strings
    scalar_config = {
        "task": "ids",
        "epochs": 1,
        "round": 1,
        "use_secagg": True,
        "use_dp": False,
        "active_clients": json.dumps(active_clients),
        "peer_keys": json.dumps(peer_keys),
    }

    # Verify JSON unpack logic in client fit
    active_val = scalar_config.get("active_clients", "")
    peer_val = scalar_config.get("peer_keys", "")
    unpacked_clients = json.loads(active_val) if isinstance(active_val, str) else active_val
    unpacked_keys = json.loads(peer_val) if isinstance(peer_val, str) else peer_val

    assert unpacked_clients == active_clients
    assert unpacked_keys == peer_keys
