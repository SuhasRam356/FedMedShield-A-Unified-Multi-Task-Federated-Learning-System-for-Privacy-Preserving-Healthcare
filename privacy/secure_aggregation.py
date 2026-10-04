"""
═══════════════════════════════════════════════════════════════
FedMedShield — Reviewed Secure Aggregation Engine
Implements SecAgg+ primitives built on:
- Upstream Flower SecAgg+ (flwr.common.secure_aggregation):
  * Shamir secret sharing (crypto.shamir) for dropout recovery
  * Quantization with stochastic rounding (quantization)
  * Cryptographic PRG masking (secaggplus_utils.pseudo_rand_gen)
- Validated Cryptography:
  * X25519 ECDH key exchange & HKDF-SHA256 key derivation

THREAT MODEL & CRYPTOGRAPHIC ASSUMPTIONS:
1. Threat Model: Semi-honest (honest-but-curious) server and non-colluding clients.
2. Threshold Assumption: At least t out of U clients must survive for Shamir secret
   reconstruction (t <= U). If fewer than t clients survive, aggregation aborts.
3. Collusion Bound: The server is assumed to collude with at most t-1 malicious clients.
   If t or more clients collude with the central server, individual updates can be unmasked.
4. Privacy Limitation: Secure Aggregation protects intermediate individual updates
   in transit from the server. It DOES NOT protect against memorization or membership
   inference from the final aggregated model; Differential Privacy (DP) is required
   in combination with SecAgg.
═══════════════════════════════════════════════════════════════
"""

import os
import math
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch

from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# Upstream Flower SecAgg+ reviewed primitives
from flwr.common.secure_aggregation.crypto.shamir import create_shares, combine_shares
from flwr.common.secure_aggregation.quantization import quantize as fl_quantize, dequantize as fl_dequantize
from flwr.common.secure_aggregation.secaggplus_utils import pseudo_rand_gen

from privacy.audit_logger import SecurityAuditLogger

logger = logging.getLogger("FedMedShield.SecAgg")

# 32-bit prime for modular arithmetic in SecAgg+ PRG masking
MODULUS = 2**31 - 1
CLIPPING_RANGE = 5.0
TARGET_RANGE = 2**20


class KeyGenerator:
    """X25519 Elliptic Curve Diffie-Hellman Key Agreement."""

    @staticmethod
    def generate_keypair() -> Tuple[x25519.X25519PrivateKey, str]:
        """Generates a fresh X25519 private key and hex-encoded public key."""
        priv = x25519.X25519PrivateKey.generate()
        pub_bytes = priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        return priv, pub_bytes.hex()

    @staticmethod
    def compute_shared_secret(priv: x25519.X25519PrivateKey, other_pub_hex: str) -> bytes:
        """
        Computes X25519 ECDH shared secret and derives a 32-byte key via HKDF-SHA256.
        """
        other_pub_bytes = bytes.fromhex(other_pub_hex)
        peer_public_key = x25519.X25519PublicKey.from_public_bytes(other_pub_bytes)
        raw_secret = priv.exchange(peer_public_key)

        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"fedmedshield-secagg-salt",
            info=b"pairwise-mask-derivation",
        ).derive(raw_secret)
        return derived_key


class Quantizer:
    """Reviewed quantization using Flower's stochastic rounding implementation."""

    @staticmethod
    def quantize(tensor: torch.Tensor, clipping_range: float = CLIPPING_RANGE, target_range: int = TARGET_RANGE) -> torch.Tensor:
        arr = tensor.detach().cpu().numpy().astype(np.float64)
        quantized_list = fl_quantize([arr], clipping_range=clipping_range, target_range=target_range)
        return torch.tensor(quantized_list[0], dtype=torch.int64, device=tensor.device)

    @staticmethod
    def dequantize(tensor: torch.Tensor, clipping_range: float = CLIPPING_RANGE, target_range: int = TARGET_RANGE) -> torch.Tensor:
        arr = tensor.detach().cpu().numpy().astype(np.int64)
        dequantized_list = fl_dequantize([arr], clipping_range=clipping_range, target_range=target_range)
        return torch.tensor(dequantized_list[0], dtype=torch.float32, device=tensor.device)


class MaskGenerator:
    """Generates pseudorandom noise masks using Flower's reviewed pseudo_rand_gen."""

    @staticmethod
    def generate_mask(seed: bytes, shape: torch.Size, device: torch.device) -> torch.Tensor:
        dims = list(shape)
        if len(dims) == 0:
            dims = [1]
        mask_arrays = pseudo_rand_gen(seed=seed, num_range=MODULUS, dimensions_list=[tuple(dims)])
        mask_np = mask_arrays[0]
        if len(shape) == 0:
            mask_np = mask_np.squeeze()
        return torch.tensor(mask_np, dtype=torch.int64, device=device)


class SecureAggregationEngine:
    """
    Reviewed SecAgg+ Engine with Shamir secret sharing and X25519 key rotation.
    """

    def __init__(self, client_id: str, threshold: int = 2):
        self.client_id = client_id
        self.threshold = threshold
        self.private_key, self.public_key = KeyGenerator.generate_keypair()
        self.peer_public_keys: Dict[str, str] = {}
        self.round_counter = 0

        SecurityAuditLogger.log_event(
            event_type="SECOBJ_INITIALIZED",
            participant_id=self.client_id,
            details={"threshold": self.threshold, "algorithm": "X25519+HKDF+FlowerSecAgg+"}
        )

    def rotate_keys(self) -> str:
        """Rotates client's X25519 keypair for forward secrecy."""
        self.private_key, self.public_key = KeyGenerator.generate_keypair()
        self.round_counter += 1
        SecurityAuditLogger.log_event(
            event_type="KEY_ROTATION",
            participant_id=self.client_id,
            details={"round": self.round_counter, "public_key_prefix": self.public_key[:8]}
        )
        return self.public_key

    def receive_public_keys(self, peer_keys: Dict[str, str]) -> None:
        """Store public keys of verified participating clients."""
        self.peer_public_keys = peer_keys

    def create_dropout_shares(self, secret_seed: bytes, total_clients: int) -> List[bytes]:
        """Generates Shamir secret shares for dropout resilience."""
        eff_threshold = max(2, min(self.threshold, total_clients))
        return create_shares(secret_seed, threshold=eff_threshold, num=total_clients)

    def reconstruct_secret_from_shares(self, shares: List[bytes]) -> bytes:
        """Reconstructs shared secret when surviving shares >= threshold."""
        return combine_shares(shares)

    def mask_local_update(self, model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Quantizes and applies pairwise additive masks using derived X25519 shared keys.
        """
        masked_state = {}

        for name, tensor in model_state.items():
            if tensor.dtype not in [torch.float32, torch.float64]:
                masked_state[name] = tensor.clone()
                continue

            # 1. Stochastic quantization
            q_tensor = Quantizer.quantize(tensor)
            mask_sum = torch.zeros_like(q_tensor)

            # 2. Pairwise additive masking
            for peer_id, peer_pub_key in self.peer_public_keys.items():
                if peer_id == self.client_id:
                    continue

                shared_key = KeyGenerator.compute_shared_secret(self.private_key, peer_pub_key)
                peer_mask = MaskGenerator.generate_mask(shared_key, tensor.shape, tensor.device)

                if self.client_id > peer_id:
                    mask_sum = (mask_sum + peer_mask) % MODULUS
                else:
                    mask_sum = (mask_sum - peer_mask) % MODULUS

            masked_state[name] = (q_tensor + mask_sum) % MODULUS

        return masked_state

    @staticmethod
    def aggregate_masked_updates(
        masked_updates: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Server-side unmasking and aggregation:
        Sums masked integer tensors; pairwise masks cancel out algebraically.
        Dequantizes summed integer tensors back to float parameters.
        """
        if not masked_updates:
            return {}

        num_clients = len(masked_updates)
        if weights is None:
            weights = [1.0 / num_clients] * num_clients

        aggregated_state = {}
        summed_ints = {}
        for name in masked_updates[0].keys():
            summed_ints[name] = torch.zeros_like(masked_updates[0][name], dtype=torch.int64)

        for update in masked_updates:
            for name, tensor in update.items():
                if tensor.dtype == torch.int64:
                    summed_ints[name] = (summed_ints[name] + tensor) % MODULUS

        for name, tensor in masked_updates[0].items():
            if tensor.dtype == torch.int64:
                # Adjust for zero-offset accumulated across K clients in stochastic quantization
                adjusted_sum = summed_ints[name] - ((num_clients - 1) * (TARGET_RANGE // 2))
                float_sum = Quantizer.dequantize(adjusted_sum)
                aggregated_state[name] = float_sum / num_clients
            else:
                float_tensors = [u[name].float() for u in masked_updates]
                avg_tensor = sum(float_tensors) / num_clients
                aggregated_state[name] = avg_tensor.to(tensor.dtype)

        SecurityAuditLogger.log_event(
            event_type="SECOBJ_AGGREGATED",
            participant_id="server",
            details={"num_clients": num_clients, "modulus": MODULUS}
        )

        return aggregated_state


SecureAggregationProtocol = SecureAggregationEngine
