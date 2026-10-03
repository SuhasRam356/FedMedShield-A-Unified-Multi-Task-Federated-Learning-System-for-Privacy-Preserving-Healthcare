"""
Federated Learning Strategies (FedAvg, FedProx, FedNova Reference Implementations)
FedMedShield Framework - Multi-Task Federated Healthcare

NOTE ON SERVER INTEGRATION:
The active Flower server (`fl_engine/server.py`) executes `SecureFedAvgStrategy`,
which subclasses Flower's native `FedAvg` to perform sample-weighted parameter aggregation
and optional mask unmasking. The strategy classes below (`FedProx`, `FedNova`) are
standalone PyTorch state-dict aggregation reference helpers for offline experimental simulation.
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod


class FLStrategy(ABC):
    """Abstract Base Strategy for Federated Learning aggregation and client weighting."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def aggregate(
        self,
        global_weights: Dict[str, torch.Tensor],
        client_updates: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client weights into an updated global model state dict.
        client_updates list of dicts containing:
          - 'weights': Dict[str, torch.Tensor]
          - 'num_samples': int
          - 'client_id': str
          - optional metrics or control variates
        """
        pass


class FedAvg(FLStrategy):
    """
    Classic Federated Averaging (McMahan et al., 2017).
    Weights each hospital client's model parameters proportional to their dataset sample size.
    """

    def __init__(self):
        super().__init__(name="FedAvg")

    def aggregate(
        self,
        global_weights: Dict[str, torch.Tensor],
        client_updates: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        if not client_updates:
            return global_weights

        total_samples = sum(u["num_samples"] for u in client_updates)
        if total_samples <= 0:
            total_samples = len(client_updates)
            for u in client_updates:
                u["num_samples"] = 1

        new_weights: Dict[str, torch.Tensor] = {}

        # Initialize with zeros in the same shape and dtype
        for key in global_weights.keys():
            # Only aggregate floating point parameters
            if torch.is_floating_point(global_weights[key]):
                new_weights[key] = torch.zeros_like(global_weights[key])
            else:
                new_weights[key] = global_weights[key].clone()

        for update in client_updates:
            weight_factor = float(update["num_samples"]) / float(total_samples)
            c_weights = update["weights"]

            for key in new_weights.keys():
                if torch.is_floating_point(new_weights[key]) and key in c_weights:
                    new_weights[key] += c_weights[key].to(new_weights[key].device) * weight_factor

        return new_weights


class FedProx(FLStrategy):
    """
    FedProx Strategy (Li et al., 2020) for Non-IID heterogeneous medical nodes.
    Features proximal term regularization mu to prevent client drift in distributed clinical silos.
    """

    def __init__(self, mu: float = 0.01):
        super().__init__(name="FedProx")
        self.mu = mu

    def get_proximal_loss(
        self,
        local_model: nn.Module,
        global_model: nn.Module
    ) -> torch.Tensor:
        """
        Computes proximal regularization penalty (mu / 2) * ||w - w_t||^2
        """
        prox_term = torch.tensor(0.0, device=next(local_model.parameters()).device)
        for (w, w_t) in zip(local_model.parameters(), global_model.parameters()):
            prox_term += torch.norm(w - w_t) ** 2
        return (self.mu / 2.0) * prox_term

    def aggregate(
        self,
        global_weights: Dict[str, torch.Tensor],
        client_updates: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        if not client_updates:
            return global_weights

        total_samples = sum(u["num_samples"] for u in client_updates)
        new_weights: Dict[str, torch.Tensor] = {}

        for key in global_weights.keys():
            if torch.is_floating_point(global_weights[key]):
                new_weights[key] = torch.zeros_like(global_weights[key])
            else:
                new_weights[key] = global_weights[key].clone()

        for update in client_updates:
            weight_factor = float(update["num_samples"]) / float(total_samples)
            c_weights = update["weights"]

            for key in new_weights.keys():
                if torch.is_floating_point(new_weights[key]) and key in c_weights:
                    new_weights[key] += c_weights[key].to(new_weights[key].device) * weight_factor

        return new_weights


class FedNova(FLStrategy):
    """
    FedNova (Wang et al., 2020) handles objective inconsistency under client computational heterogeneity.
    Normalizes local updates according to the number of local gradient steps taken.
    """

    def __init__(self):
        super().__init__(name="FedNova")

    def aggregate(
        self,
        global_weights: Dict[str, torch.Tensor],
        client_updates: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        if not client_updates:
            return global_weights

        total_samples = sum(u["num_samples"] for u in client_updates)
        
        # Calculate normalized tau steps
        taus = [u.get("local_steps", 10) for u in client_updates]
        p_ks = [float(u["num_samples"]) / float(total_samples) for u in client_updates]
        
        tau_eff = sum(p * t for p, t in zip(p_ks, taus))

        new_weights: Dict[str, torch.Tensor] = copy.deepcopy(global_weights)

        # Delta aggregation
        delta_acc: Dict[str, torch.Tensor] = {
            key: torch.zeros_like(val)
            for key, val in global_weights.items()
            if torch.is_floating_point(val)
        }

        for p_k, tau_k, update in zip(p_ks, taus, client_updates):
            c_weights = update["weights"]
            scale = p_k * (tau_eff / max(tau_k, 1e-5))

            for key in delta_acc.keys():
                if key in c_weights:
                    diff = c_weights[key].to(delta_acc[key].device) - global_weights[key].to(delta_acc[key].device)
                    delta_acc[key] += diff * scale

        for key in delta_acc.keys():
            new_weights[key] = global_weights[key] + delta_acc[key]

        return new_weights


def get_strategy(strategy_name: str, **kwargs) -> FLStrategy:
    """Factory helper to instantiate strategies dynamically."""
    name = strategy_name.lower().strip()
    if name == "fedprox":
        return FedProx(mu=kwargs.get("mu", 0.01))
    elif name == "fednova":
        return FedNova()
    else:
        return FedAvg()
