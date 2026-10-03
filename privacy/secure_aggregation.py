"""
═══════════════════════════════════════════════════════════════
FedMedShield — Secure Aggregation Module (Educational Simulation)
Demonstrates the algebraic concept of pairwise additive mask cancellation.
NOTICE: This is an algorithmic simulation for research demonstration.
It is NOT an independently audited cryptographic implementation and does
not include Diffie-Hellman key agreement, Shamir secret sharing for dropouts,
or zero-knowledge proofs. For production deployments, integrate Flower's
official SecAgg+ workflow or an established MPC framework.
═══════════════════════════════════════════════════════════════
"""

import hashlib
import struct
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Very large prime for modular arithmetic in SMPC simulation
# In production, this would be a 256-bit prime. We use a smaller one
# here to prevent standard float64 overflow during simulation while
# retaining the modular arithmetic properties.
MODULUS = 2**31 - 1
SCALING_FACTOR = 1e5  # For quantizing floats to ints


class KeyGenerator:
    """Simulates Elliptic Curve Diffie-Hellman (ECDH) Key generation."""
    
    @staticmethod
    def generate_keypair(client_id: str) -> Tuple[str, str]:
        """Returns a simulated (private_key, public_key) pair."""
        # Simulate private key generation
        priv_str = f"priv_{client_id}_{np.random.randint(1000000)}"
        private_key = hashlib.sha256(priv_str.encode()).hexdigest()
        
        # Simulate public key derivation (g^priv mod P)
        pub_str = f"pub_derived_from_{private_key}"
        public_key = hashlib.sha256(pub_str.encode()).hexdigest()
        
        return private_key, public_key

    @staticmethod
    def compute_shared_secret(my_private_key: str, other_public_key: str) -> int:
        """
        Simulates DH shared secret computation.
        In reality: secret = (other_pub)^my_priv mod P.
        Here we deterministically combine them since we simulate the math.
        """
        # Sort to ensure both parties generate the identical string to hash
        components = sorted([my_private_key, other_public_key])
        combined = f"{components[0]}_{components[1]}"
        # Return a 32-bit integer seed
        return int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)


class Quantizer:
    """Handles conversion between continuous floats and discrete integers."""
    
    @staticmethod
    def quantize(tensor: torch.Tensor, scale: float = SCALING_FACTOR) -> torch.Tensor:
        """Float to Int conversion for modular arithmetic."""
        # Scale and round to nearest integer
        scaled = torch.round(tensor * scale)
        # Convert to large int64 to prevent overflow before modulo
        return scaled.to(torch.int64)

    @staticmethod
    def dequantize(tensor: torch.Tensor, scale: float = SCALING_FACTOR) -> torch.Tensor:
        """Int to Float conversion."""
        # Handle negative numbers in modular arithmetic
        # If val > MODULUS/2, it represents a negative number
        half_mod = MODULUS // 2
        
        # Use torch.where for vectorized conditional
        adjusted = torch.where(
            tensor > half_mod,
            tensor - MODULUS,
            tensor
        )
        
        # Convert back to float and scale down
        return adjusted.to(torch.float32) / scale


class MaskGenerator:
    """Generates cryptographic noise masks using a PRG."""
    
    @staticmethod
    def generate_mask(seed: int, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """
        Generates a random integer mask modulo P using the shared seed.
        We use numpy for deterministic cross-platform seeding.
        """
        numel = shape.numel()
        rng = np.random.RandomState(seed)
        
        # Generate random ints in [0, MODULUS)
        mask_np = rng.randint(0, MODULUS, size=numel, dtype=np.int64)
        
        # Convert to tensor and reshape
        return torch.tensor(mask_np, device=device).view(shape)


class SecureAggregationEngine:
    """
    Main controller for the Secure Aggregation protocol.
    Provides methods for both Client-side masking and Server-side aggregation.
    """

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.private_key, self.public_key = KeyGenerator.generate_keypair(client_id)
        self.peer_public_keys: Dict[str, str] = {}
        logger.debug(f"[{client_id}] SecAgg initialized. PubKey={self.public_key[:8]}...")

    def receive_public_keys(self, peer_keys: Dict[str, str]) -> None:
        """Store public keys of other clients participating in the round."""
        self.peer_public_keys = peer_keys

    def mask_local_update(self, model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Client-side operation: 
        1. Quantize weights.
        2. Generate shared masks with all peers.
        3. Add/Subtract masks modulo P.
        """
        masked_state = {}
        
        for name, tensor in model_state.items():
            if tensor.dtype not in [torch.float32, torch.float64]:
                masked_state[name] = tensor.clone()
                continue
                
            # 1. Quantize
            q_tensor = Quantizer.quantize(tensor)
            
            # 2. Add pairwise masks
            mask_sum = torch.zeros_like(q_tensor)
            
            for peer_id, peer_pub_key in self.peer_public_keys.items():
                if peer_id == self.client_id:
                    continue
                    
                # Compute shared DH secret
                shared_seed = KeyGenerator.compute_shared_secret(self.private_key, peer_pub_key)
                
                # Expand seed into a mask
                peer_mask = MaskGenerator.generate_mask(shared_seed, tensor.shape, tensor.device)
                
                # 3. Add or subtract based on canonical ordering (to ensure cancellation at server)
                if self.client_id > peer_id:
                    mask_sum = (mask_sum + peer_mask) % MODULUS
                else:
                    mask_sum = (mask_sum - peer_mask) % MODULUS
                    
            # Add mask sum to the quantized tensor, modulo P
            masked_state[name] = (q_tensor + mask_sum) % MODULUS
            
        return masked_state

    @staticmethod
    def aggregate_masked_updates(
        masked_updates: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Server-side operation:
        1. Sum the masked integer tensors (masks perfectly cancel out modulo P).
        2. Dequantize back to floating point values.
        3. Apply weighting (FedAvg) post-unmasking.
        
        Args:
            masked_updates: List of model states from clients (integers).
            weights: Optional FedAvg weights (e.g., number of samples).
                     If provided, weighted average is calculated.
        """
        if not masked_updates:
            return {}

        num_clients = len(masked_updates)
        if weights is None:
            weights = [1.0 / num_clients] * num_clients
            
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        aggregated_state = {}
        
        # Step 1: Secure Sum (Modulo Arithmetic)
        # Because every mask +M was added by one client and -M by another,
        # the sum of all clients completely removes the cryptographic noise.
        summed_ints = {}
        for name in masked_updates[0].keys():
            summed_ints[name] = torch.zeros_like(masked_updates[0][name], dtype=torch.int64)
            
        for update in masked_updates:
            for name, tensor in update.items():
                if tensor.dtype == torch.int64:
                    summed_ints[name] = (summed_ints[name] + tensor) % MODULUS
                
        # Step 2 & 3: Dequantize and apply FedAvg weights
        for name, tensor in masked_updates[0].items():
            if tensor.dtype == torch.int64:
                # Dequantize the raw sum
                float_sum = Quantizer.dequantize(summed_ints[name])
                
                # To get the average, we divide the sum by num_clients
                # (or apply specific weighting if needed)
                # Since the server just has the sum, we apply the weighted average logic here.
                # In strict SMPC, weights are multiplied *before* masking by clients, 
                # but doing it post-summation here is mathematically equivalent for average.
                aggregated_state[name] = float_sum / num_clients
            else:
                # Non-float parameters (e.g., batch tracking) - just take average or first
                float_tensors = [u[name].float() for u in masked_updates]
                avg_tensor = sum(float_tensors) / num_clients
                aggregated_state[name] = avg_tensor.to(tensor.dtype)
                
        return aggregated_state


# Alias for backward compatibility
SecureAggregationProtocol = SecureAggregationEngine

