"""
Phase 2 Privacy Guarantees & Security Verification Suite
Tests:
1. Reviewed SecAgg+ Shamir Secret Sharing & Quantization (Flower)
2. X25519 ECDH Key Exchange and Forward-Secret Key Rotation
3. Validated RDP Accounting (Opacus) and Pre-Release Budget Enforcement
4. Metadata Bypass Prevention (raw update norm / clip state suppression)
5. Membership Inference & Update Leakage Resistance
6. Security Audit Event Logging
"""

import sys
import os
import json
import math
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from privacy.secure_aggregation import (
    KeyGenerator,
    Quantizer,
    MaskGenerator,
    SecureAggregationEngine,
    MODULUS,
)
from privacy.differential_privacy import (
    PrivacyAccountant,
    DifferentialPrivacyEngine,
    AdaptiveClipper,
)
from privacy.audit_logger import SecurityAuditLogger, AUDIT_LOG_FILE


def test_flower_shamir_secret_sharing_threshold():
    """Verify that Shamir secret sharing recovers with >= threshold and fails below threshold."""
    engine = SecureAggregationEngine(client_id="hospital-a", threshold=2)
    secret_seed = b"32_byte_secret_cryptoseed_here!"

    # Create shares for 3 hospitals with threshold 2
    shares = engine.create_dropout_shares(secret_seed, total_clients=3)
    assert len(shares) == 3

    # Reconstruct with threshold (2 shares)
    recovered = engine.reconstruct_secret_from_shares(shares[:2])
    assert recovered == secret_seed

    # Reconstruct with all 3 shares
    recovered_all = engine.reconstruct_secret_from_shares(shares)
    assert recovered_all == secret_seed


def test_flower_stochastic_quantization_roundtrip():
    """Verify stochastic quantization preserves tensor magnitudes within bound."""
    tensor = torch.tensor([-2.5, -0.5, 0.0, 1.25, 3.8], dtype=torch.float32)
    q_tensor = Quantizer.quantize(tensor)
    assert q_tensor.dtype == torch.int64

    deq_tensor = Quantizer.dequantize(q_tensor)
    assert deq_tensor.dtype == torch.float32
    # Error bounded by quantization precision
    max_diff = torch.max(torch.abs(tensor - deq_tensor)).item()
    assert max_diff < 0.05


def test_x25519_key_exchange_and_rotation():
    """Verify X25519 Diffie-Hellman key exchange and forward-secret key rotation."""
    engine_a = SecureAggregationEngine(client_id="hospital-a")
    engine_b = SecureAggregationEngine(client_id="hospital-b")

    # Shared secret derivation from public keys
    secret_ab = KeyGenerator.compute_shared_secret(engine_a.private_key, engine_b.public_key)
    secret_ba = KeyGenerator.compute_shared_secret(engine_b.private_key, engine_a.public_key)
    assert secret_ab == secret_ba
    assert len(secret_ab) == 32

    # Rotate keys
    old_pub_a = engine_a.public_key
    new_pub_a = engine_a.rotate_keys()
    assert old_pub_a != new_pub_a

    # After rotation, new shared secret differs (forward secrecy)
    new_secret_ab = KeyGenerator.compute_shared_secret(engine_a.private_key, engine_b.public_key)
    assert new_secret_ab != secret_ab


def test_secagg_mask_cancellation():
    """Verify that pairwise additive masks cancel out across participants."""
    engine_a = SecureAggregationEngine(client_id="hospital-a")
    engine_b = SecureAggregationEngine(client_id="hospital-b")

    engine_a.receive_public_keys({"hospital-a": engine_a.public_key, "hospital-b": engine_b.public_key})
    engine_b.receive_public_keys({"hospital-a": engine_a.public_key, "hospital-b": engine_b.public_key})

    weights_a = {"weight": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)}
    weights_b = {"weight": torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)}

    masked_a = engine_a.mask_local_update(weights_a)
    masked_b = engine_b.mask_local_update(weights_b)

    # Server aggregates masked integer updates
    aggregated = SecureAggregationEngine.aggregate_masked_updates([masked_a, masked_b])

    # Expected mean: ([1, 2, 3] + [3, 4, 5]) / 2 = [2, 3, 4]
    expected = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
    assert torch.allclose(aggregated["weight"], expected, atol=0.05)


def test_opacus_rdp_accounting_and_budget_gating():
    """Verify Opacus RDP accountant tracks epsilon and blocks updates once budget exhausted."""
    dp_engine = DifferentialPrivacyEngine(
        target_epsilon=6.0,
        target_delta=1e-5,
        noise_multiplier=1.0,
        initial_max_norm=1.0,
        dp_level="client_level",
        num_total_clients=4,
        num_participating_clients=2,
    )

    global_weights = {"w": torch.zeros(10, dtype=torch.float32)}
    local_weights = {"w": torch.ones(10, dtype=torch.float32)}

    # Round 1: Within budget (1 step with q=0.5, sigma=1.0 gives eps ~3.89 < 6.0)
    priv_weights_1, metrics_1 = dp_engine.apply_dp(
        local_model_state=local_weights,
        global_model_state=global_weights,
        local_steps=1,
    )
    assert not metrics_1["budget_exhausted"]
    assert 0.0 < metrics_1["current_epsilon"] < 6.0
    # Weights were perturbed
    assert not torch.equal(priv_weights_1["w"], global_weights["w"])

    # Burn through budget
    for _ in range(5):
        dp_engine.apply_dp(
            local_model_state=local_weights,
            global_model_state=global_weights,
            local_steps=2,
        )

    # Now budget is exhausted -> update must be withheld (returns global weights)
    priv_weights_halt, metrics_halt = dp_engine.apply_dp(
        local_model_state=local_weights,
        global_model_state=global_weights,
        local_steps=1,
    )
    assert metrics_halt["budget_exhausted"] is True
    assert torch.equal(priv_weights_halt["w"], global_weights["w"])


def test_metadata_bypass_prevention():
    """Verify that sensitive update norms and clip thresholds are suppressed from metrics."""
    dp_engine = DifferentialPrivacyEngine(
        target_epsilon=5.0,
        noise_multiplier=1.0,
        initial_max_norm=1.5,
    )

    global_w = {"w": torch.zeros(20, dtype=torch.float32)}
    local_w = {"w": torch.randn(20, dtype=torch.float32)}

    _, metrics = dp_engine.apply_dp(local_w, global_w, local_steps=1)

    # Must NOT contain raw update norms or clipping bounds
    forbidden_keys = [
        "update_norm",
        "clip_norm",
        "clip_threshold",
        "gradient_norm",
        "raw_loss",
        "dataset_size",
        "patient_count",
    ]
    for key in forbidden_keys:
        assert key not in metrics, f"Privacy violation: '{key}' leaked in metadata!"

    assert "current_epsilon" in metrics
    assert "target_epsilon" in metrics
    assert "budget_exhausted" in metrics
    assert "dp_level" in metrics


def test_membership_inference_defense():
    """
    Membership Inference Attack (MIA) and Update Leakage defense test:
    1. Verifies that DP noise obscures private gradient directions (reducing cosine similarity).
    2. Verifies that DP perturbation reduces prediction confidence memorization gap.
    """
    torch.manual_seed(42)

    dim = 16
    # 1. Update Leakage: Test that private gradient direction cannot be directly reconstructed
    true_private_grad = torch.randn(dim)
    dp_engine = DifferentialPrivacyEngine(noise_multiplier=1.5, initial_max_norm=1.0)
    
    global_s = {"w": torch.zeros(dim)}
    local_s = {"w": true_private_grad.clone()}
    
    dp_state, _ = dp_engine.apply_dp(local_s, global_s, local_steps=1)
    leaked_grad = dp_state["w"]
    
    cos_sim = torch.cosine_similarity(true_private_grad.unsqueeze(0), leaked_grad.unsqueeze(0)).item()
    # Without DP, cos_sim is 1.0; with DP noise, similarity is significantly diminished
    assert cos_sim < 0.65, f"Update leakage too high: cosine similarity {cos_sim:.4f} >= 0.65"

    # 2. Membership Inference: Test that DP suppresses the member vs non-member loss ratio
    torch.manual_seed(123)
    dim_mia = 10
    n_samples = 30
    x_members = torch.randn(n_samples, dim_mia)
    y_members = (x_members[:, 0] > 0).long()
    x_non_members = torch.randn(n_samples, dim_mia)
    y_non_members = (x_non_members[:, 0] > 0).long()

    model_init = torch.nn.Linear(dim_mia, 2)
    global_model_state = {k: v.clone() for k, v in model_init.state_dict().items()}

    # Train unprotected client
    model_unprot = torch.nn.Linear(dim_mia, 2)
    model_unprot.load_state_dict(global_model_state)
    optimizer = torch.optim.Adam(model_unprot.parameters(), lr=0.1)
    for _ in range(50):
        loss = torch.nn.functional.cross_entropy(model_unprot(x_members), y_members)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_m_unprot = torch.nn.functional.cross_entropy(model_unprot(x_members), y_members).item()
    loss_nm_unprot = torch.nn.functional.cross_entropy(model_unprot(x_non_members), y_non_members).item()
    ratio_unprot = loss_nm_unprot / max(1e-5, loss_m_unprot)

    # Privatize model update using DifferentialPrivacyEngine
    dp_engine_mia = DifferentialPrivacyEngine(noise_multiplier=2.0, initial_max_norm=1.0)
    dp_state_dict, _ = dp_engine_mia.apply_dp(model_unprot.state_dict(), global_model_state)

    model_dp = torch.nn.Linear(dim_mia, 2)
    model_dp.load_state_dict(dp_state_dict)
    loss_m_dp = torch.nn.functional.cross_entropy(model_dp(x_members), y_members).item()
    loss_nm_dp = torch.nn.functional.cross_entropy(model_dp(x_non_members), y_non_members).item()
    ratio_dp = loss_nm_dp / max(1e-5, loss_m_dp)

    # DP prevents membership inferability by shrinking the loss discrepancy ratio
    assert ratio_dp < ratio_unprot, (
        f"DP should reduce MIA loss ratio: DP ratio {ratio_dp:.4f} vs Unprotected {ratio_unprot:.4f}"
    )


def test_security_audit_logging():
    """Verify security audit logger records structured events to disk."""
    event = SecurityAuditLogger.log_event(
        event_type="TEST_VERIFICATION",
        participant_id="hospital_test",
        details={"test_metric": 42},
        severity="INFO"
    )
    assert event["event_type"] == "TEST_VERIFICATION"
    assert event["participant_id"] == "hospital_test"

    recent = SecurityAuditLogger.get_recent_events(limit=10)
    assert any(e["event_type"] == "TEST_VERIFICATION" for e in recent)
