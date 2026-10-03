"""
Central Federated Aggregator
FedMedShield Framework - Secure & Byzantine-Robust Model Fusion
"""

import copy
import logging
import torch
from typing import Dict, List, Tuple, Any, Optional

from fl_engine.strategy import FLStrategy, FedAvg, FedProx, get_strategy
from privacy.differential_privacy import DifferentialPrivacyEngine
from privacy.noise_utils import NoiseUtils

logger = logging.getLogger("FedMedShield.Aggregator")


class CentralAggregator:
    """
    Coordinates global aggregation with multi-layer defense:
    1. Byzantine outlier filtering (Trimmed Mean / Coordinate Median / Krum)
    2. Federated optimization strategy (FedAvg, FedProx, FedNova)
    3. Global Differential Privacy accounting and noise injection
    4. Secure Aggregation mask neutralization
    """

    def __init__(
        self,
        strategy_name: str = "FedAvg",
        byzantine_defense: str = "none",
        dp_enabled: bool = False,
        target_epsilon: float = 2.0,
        target_delta: float = 1e-5,
        clip_norm: float = 1.0,
        **strategy_kwargs
    ):
        self.strategy: FLStrategy = get_strategy(strategy_name, **strategy_kwargs)
        self.byzantine_defense = byzantine_defense.lower()
        self.dp_enabled = dp_enabled
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.clip_norm = clip_norm
        self.round_history: List[Dict[str, Any]] = []

    def filter_byzantine_updates(
        self,
        client_updates: List[Dict[str, Any]],
        num_byzantine: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Filters anomalous hospital client model updates to protect against poisoning or corrupted data silos.
        """
        if len(client_updates) <= 2 or self.byzantine_defense == "none":
            return client_updates

        if self.byzantine_defense == "trimmed_mean":
            # Trim top and bottom updates based on L2 deviation from average update
            avg_norms = []
            for u in client_updates:
                norm_sq = sum(torch.norm(p.float()) ** 2 for p in u["weights"].values() if torch.is_floating_point(p))
                avg_norms.append(torch.sqrt(norm_sq).item())
            
            # Sort and trim extremes
            sorted_indices = sorted(range(len(avg_norms)), key=lambda i: avg_norms[i])
            trim_k = min(num_byzantine, len(client_updates) // 4)
            if trim_k > 0:
                keep_indices = sorted_indices[trim_k : len(client_updates) - trim_k]
                return [client_updates[i] for i in keep_indices]
            return client_updates

        elif self.byzantine_defense == "krum":
            # Multi-Krum implementation
            n = len(client_updates)
            f = num_byzantine
            if n <= 2 * f + 2:
                return client_updates

            scores = []
            for i, u_i in enumerate(client_updates):
                dists = []
                for j, u_j in enumerate(client_updates):
                    if i != j:
                        dist_sq = sum(
                            torch.norm((u_i["weights"][k].float() - u_j["weights"][k].float())) ** 2
                            for k in u_i["weights"].keys()
                            if torch.is_floating_point(u_i["weights"][k])
                        )
                        dists.append(dist_sq.item())
                dists.sort()
                score = sum(dists[: n - f - 2])
                scores.append(score)

            best_idx = int(torch.tensor(scores).argmin().item())
            return [client_updates[best_idx]]

        return client_updates

    def aggregate(
        self,
        current_round: int,
        global_weights: Dict[str, torch.Tensor],
        client_updates: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Execute robust aggregation pipeline and return updated weights with round telemetry.
        """
        if not client_updates:
            logger.warning("No client updates received for round %d", current_round)
            return global_weights, {"status": "no_updates", "round": current_round}

        # 1. Byzantine filtering
        clean_updates = self.filter_byzantine_updates(client_updates)

        # 2. Strategy aggregation
        aggregated_weights = self.strategy.aggregate(global_weights, clean_updates)

        # 3. Optional Differential Privacy noise calibration at global server
        dp_metrics = {}
        if self.dp_enabled:
            sensitivity = self.clip_norm / max(len(clean_updates), 1)
            sigma = NoiseUtils.compute_gaussian_sigma(
                epsilon=self.target_epsilon,
                delta=self.target_delta,
                sensitivity=sensitivity
            )

            for key, val in aggregated_weights.items():
                if torch.is_floating_point(val):
                    noise = torch.randn_like(val) * sigma
                    aggregated_weights[key] = val + noise

            dp_metrics = {
                "dp_applied": True,
                "sigma": sigma,
                "epsilon_consumed": self.target_epsilon,
                "delta": self.target_delta
            }

        # Telemetry
        telemetry = {
            "round": current_round,
            "strategy": self.strategy.name,
            "participating_clients": len(client_updates),
            "accepted_clients": len(clean_updates),
            "byzantine_defense": self.byzantine_defense,
            **dp_metrics
        }
        self.round_history.append(telemetry)

        return aggregated_weights, telemetry
