"""
═══════════════════════════════════════════════════════════════
FedMedShield — COVID-19 Mortality Prediction Model
Specialized binary classifier for predicting COVID-19 patient
mortality risk. Uses an attention mechanism to identify the most
critical clinical features contributing to mortality.
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureAttention(nn.Module):
    """
    Soft attention mechanism over input features.
    Learns which clinical features (e.g., age, O2 saturation, comorbidities)
    are most predictive of COVID-19 mortality, producing interpretable
    attention weights that serve as "key factors" in the UI.
    """

    def __init__(self, input_dim: int, attention_dim: int = 32) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted features.

        Args:
            x: Input features (batch, input_dim).

        Returns:
            Tuple of (weighted features, attention weights).
        """
        attention_scores = self.attention(x)
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_features = x * attention_weights
        return weighted_features, attention_weights


class COVIDMortalityModel(nn.Module):
    """
    COVID-19 mortality risk prediction model with feature attention.

    Architecture:
      Input → FeatureAttention → Linear(128) → BatchNorm → GELU → Dropout
            → Linear(64) → BatchNorm → GELU → Dropout
            → Linear(32) → GELU → Linear(1) → Sigmoid

    The attention mechanism provides interpretability by highlighting
    which patient features most influence the mortality prediction,
    enabling clinicians to understand model decisions.
    """

    FEATURE_NAMES = [
        "age", "gender", "bmi", "bp_systolic", "bp_diastolic",
        "glucose", "cholesterol", "hba1c", "creatinine", "wbc",
        "heart_rate", "temperature", "respiratory_rate",
        "oxygen_saturation", "hemoglobin", "platelets", "albumin",
        "bun", "smoking", "comorbidities",
    ]

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.input_dim = input_dim

        # Feature attention layer
        self.feature_attention = FeatureAttention(input_dim, attention_dim=32)

        # Build classifier layers dynamically
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = h_dim
            dropout_rate *= 0.8  # Decay dropout in deeper layers

        layers.append(nn.Linear(prev_dim, 1))
        self.classifier = nn.Sequential(*layers)

        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"COVIDMortalityModel initialized: {total_params:,} params")

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with attention.

        Args:
            x: Patient features (batch, input_dim).

        Returns:
            Dict with 'logit' and 'attention_weights'.
        """
        weighted_x, attention_weights = self.feature_attention(x)
        logit = self.classifier(weighted_x).squeeze(-1)

        return {
            "logit": logit,
            "attention_weights": attention_weights,
        }

    def predict_with_factors(
        self,
        x: torch.Tensor,
        top_k: int = 5,
    ) -> Dict[str, object]:
        """
        Inference with interpretable key factors.

        Args:
            x: Patient features (batch, input_dim).
            top_k: Number of top contributing features to report.

        Returns:
            Dict with mortality_risk, risk_level, key_factors.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            mortality_risk = torch.sigmoid(outputs["logit"])
            attn = outputs["attention_weights"]

            # Get top-k features by attention weight for each sample
            batch_factors = []
            for i in range(x.size(0)):
                top_indices = torch.topk(attn[i], k=min(top_k, self.input_dim)).indices
                factors = [self.FEATURE_NAMES[idx.item()] for idx in top_indices]
                batch_factors.append(factors)

            # Classify risk level
            risk_levels = []
            for risk in mortality_risk:
                r = risk.item()
                if r < 0.25:
                    risk_levels.append("low")
                elif r < 0.50:
                    risk_levels.append("moderate")
                elif r < 0.75:
                    risk_levels.append("high")
                else:
                    risk_levels.append("critical")

            return {
                "mortality_risk": mortality_risk,
                "risk_level": risk_levels,
                "key_factors": batch_factors,
            }
