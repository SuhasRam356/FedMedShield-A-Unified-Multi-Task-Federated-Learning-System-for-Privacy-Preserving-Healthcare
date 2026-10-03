"""
Noise Utilities and Calibrations for Differential Privacy
FedMedShield Framework - Privacy-Preserving Healthcare FL
"""

import math
import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional

class NoiseUtils:
    """
    Advanced mathematical noise calibration and differential privacy accounting
    supporting Gaussian, Laplace, and analytical Moments Accountant tracking.
    """

    @staticmethod
    def compute_gaussian_sigma(
        epsilon: float,
        delta: float,
        sensitivity: float = 1.0
    ) -> float:
        """
        Compute standard deviation sigma for Gaussian mechanism using classical bound:
        sigma >= sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be strictly positive.")
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must satisfy 0 < delta < 1.")
        
        sigma = (sensitivity * math.sqrt(2 * math.log(1.25 / delta))) / epsilon
        return float(sigma)

    @staticmethod
    def compute_analytic_gaussian_sigma(
        epsilon: float,
        delta: float,
        sensitivity: float = 1.0,
        tol: float = 1e-6
    ) -> float:
        """
        Analytic Gaussian Mechanism (Balle and Wang, 2018) for optimal noise calibration.
        Provides strictly tighter privacy bounds compared to classical approximation.
        """
        delta_0 = 0.5 * math.erfc(epsilon / (2 * math.sqrt(2)))
        
        def phi(t: float) -> float:
            return 0.5 * math.erfc(-t / math.sqrt(2))

        def case_a(u: float, v: float) -> float:
            return phi(math.sqrt(epsilon * u)) - math.exp(epsilon) * phi(-math.sqrt(epsilon * (u + 2)))

        def case_b(u: float, v: float) -> float:
            return phi(-math.sqrt(epsilon * u)) - math.exp(epsilon) * phi(-math.sqrt(epsilon * (u + 2)))

        if delta >= delta_0:
            low, high = 0.0, 1.0
            while case_a(low, high) > delta:
                high *= 2.0
            for _ in range(50):
                mid = (low + high) / 2.0
                if case_a(mid, 1.0) < delta:
                    high = mid
                else:
                    low = mid
            sigma = (sensitivity * (math.sqrt(low + 2) + math.sqrt(low))) / (2 * math.sqrt(epsilon))
        else:
            low, high = 0.0, 1.0
            while case_b(low, high) < delta:
                high *= 2.0
            for _ in range(50):
                mid = (low + high) / 2.0
                if case_b(mid, 1.0) < delta:
                    low = mid
                else:
                    high = mid
            sigma = (sensitivity * (math.sqrt(low + 2) - math.sqrt(low))) / (2 * math.sqrt(epsilon))

        return float(sigma)

    @staticmethod
    def add_gaussian_noise(
        tensor: torch.Tensor,
        sensitivity: float,
        epsilon: float,
        delta: float,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Calibrate and add Gaussian noise to a PyTorch tensor.
        """
        sigma = NoiseUtils.compute_gaussian_sigma(epsilon, delta, sensitivity)
        if device is None:
            device = tensor.device
        noise = torch.randn_like(tensor, device=device) * sigma
        return tensor + noise, sigma

    @staticmethod
    def add_laplace_noise(
        tensor: torch.Tensor,
        sensitivity: float,
        epsilon: float,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Add Laplace noise scaled by sensitivity / epsilon for pure (eps, 0)-DP.
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be strictly positive.")
        scale = sensitivity / epsilon
        
        # PyTorch Laplace distribution
        if device is None:
            device = tensor.device
        dist = torch.distributions.Laplace(
            loc=torch.tensor([0.0], device=device),
            scale=torch.tensor([scale], device=device)
        )
        noise = dist.sample(tensor.shape).squeeze(-1)
        return tensor + noise, scale

    @staticmethod
    def clip_l2_norm(
        parameters: List[torch.Tensor],
        max_norm: float
    ) -> Tuple[List[torch.Tensor], float]:
        """
        Clip parameter updates to enforce bounded global L2 sensitivity.
        """
        total_norm = torch.norm(
            torch.stack([torch.norm(p.detach(), 2) for p in parameters]), 2
        ).item()

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            clipped = [p * clip_coef for p in parameters]
        else:
            clipped = [p.clone() for p in parameters]
            
        return clipped, total_norm

    @staticmethod
    def compute_rdp_step(q: float, sigma: float, alpha: float) -> float:
        """
        Compute Renyi Differential Privacy (RDP) guarantee at order alpha
        for a subsampled Gaussian mechanism with sampling ratio q.
        """
        if q == 0:
            return 0.0
        if q == 1.0:
            return alpha / (2 * (sigma ** 2))
        
        # Subsampled Renyi DP upper bound approximation (Mironov 2019)
        term1 = q ** 2 * alpha / (2 * (sigma ** 2))
        return float(term1)

    @staticmethod
    def rdp_to_dp(rdp_orders: np.ndarray, rdp_values: np.ndarray, delta: float) -> float:
        """
        Convert cumulative Renyi DP to classical (epsilon, delta)-DP via optimal infimum:
        eps(delta) = min_{alpha > 1} ( RDP(alpha) + log(1/delta) / (alpha - 1) )
        """
        eps_values = rdp_values + np.log(1.0 / delta) / (rdp_orders - 1.0)
        return float(np.min(eps_values))

    @staticmethod
    def estimate_privacy_budget_per_round(
        total_rounds: int,
        target_epsilon: float,
        target_delta: float,
        sample_rate: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute dynamic per-round noise multiplier to guarantee target (eps, delta)
        after `total_rounds` federated communication rounds.
        """
        orders = np.linspace(1.2, 64.0, 100)
        # Binary search for sigma
        low_sigma, high_sigma = 0.1, 50.0
        best_sigma = high_sigma

        for _ in range(40):
            mid_sigma = (low_sigma + high_sigma) / 2.0
            rdp_vals = np.array([
                total_rounds * NoiseUtils.compute_rdp_step(sample_rate, mid_sigma, a)
                for a in orders
            ])
            computed_eps = NoiseUtils.rdp_to_dp(orders, rdp_vals, target_delta)

            if computed_eps <= target_epsilon:
                best_sigma = mid_sigma
                high_sigma = mid_sigma
            else:
                low_sigma = mid_sigma

        return {
            "target_epsilon": target_epsilon,
            "target_delta": target_delta,
            "total_rounds": total_rounds,
            "noise_multiplier": round(float(best_sigma), 4),
            "estimated_round_epsilon": round(target_epsilon / total_rounds, 4)
        }
