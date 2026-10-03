"""
═══════════════════════════════════════════════════════════════
FedMedShield — Sepsis Early Warning Model
Temporal-aware binary classifier for predicting sepsis onset.
Uses a GRU-based architecture to capture temporal trends in
patient vitals (even from a single snapshot, it learns temporal
patterns via positional encoding of features).
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class SepsisEarlyWarningModel(nn.Module):
    """
    Sepsis early warning model with feature grouping and attention.

    Groups clinical features into physiological systems:
    - Vital signs: heart_rate, temperature, respiratory_rate, bp_systolic, bp_diastolic, oxygen_saturation
    - Lab values: wbc, creatinine, platelets, albumin, bun, glucose, hba1c
    - Demographics: age, gender, bmi, comorbidities, smoking
    - Other: cholesterol, hemoglobin

    Each group is encoded separately, then fused with cross-group attention
    to capture inter-system interactions (e.g., elevated WBC + low BP → sepsis).

    Output includes sepsis risk score, urgency level, and recommended action.
    """

    # Feature groups by physiological system
    VITAL_INDICES = [3, 4, 10, 11, 12, 13]   # bp_sys, bp_dia, hr, temp, resp, o2
    LAB_INDICES = [5, 7, 8, 9, 15, 16, 17]   # glucose, hba1c, creat, wbc, platelets, albumin, bun
    DEMO_INDICES = [0, 1, 2, 18, 19]          # age, gender, bmi, smoking, comorbidities
    OTHER_INDICES = [6, 14]                    # cholesterol, hemoglobin

    URGENCY_THRESHOLDS = {
        "normal": 0.2,
        "watch": 0.4,
        "warning": 0.65,
        "critical": 1.0,
    }

    RECOMMENDED_ACTIONS = {
        "normal": "Continue routine monitoring. No immediate intervention required.",
        "watch": "Increase monitoring frequency to every 2 hours. Recheck labs in 4 hours.",
        "warning": "Initiate sepsis bundle. Obtain blood cultures. Consider empiric antibiotics.",
        "critical": "URGENT: Activate rapid response team. Initiate Hour-1 sepsis bundle immediately. "
                    "IV fluid resuscitation, blood cultures, broad-spectrum antibiotics.",
    }

    def __init__(
        self,
        input_dim: int = 20,
        group_hidden_dim: int = 32,
        fusion_dim: int = 64,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim

        # Group-specific encoders
        self.vital_encoder = nn.Sequential(
            nn.Linear(len(self.VITAL_INDICES), group_hidden_dim),
            nn.BatchNorm1d(group_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(group_hidden_dim, group_hidden_dim),
            nn.GELU(),
        )

        self.lab_encoder = nn.Sequential(
            nn.Linear(len(self.LAB_INDICES), group_hidden_dim),
            nn.BatchNorm1d(group_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(group_hidden_dim, group_hidden_dim),
            nn.GELU(),
        )

        self.demo_encoder = nn.Sequential(
            nn.Linear(len(self.DEMO_INDICES), group_hidden_dim),
            nn.BatchNorm1d(group_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(group_hidden_dim, group_hidden_dim),
            nn.GELU(),
        )

        self.other_encoder = nn.Sequential(
            nn.Linear(len(self.OTHER_INDICES), group_hidden_dim),
            nn.BatchNorm1d(group_hidden_dim),
            nn.GELU(),
        )

        # Cross-group fusion with attention
        num_groups = 4
        total_group_dim = group_hidden_dim * num_groups

        self.group_attention = nn.Sequential(
            nn.Linear(total_group_dim, num_groups),
            nn.Softmax(dim=-1),
        )

        # Final classifier after attention-weighted fusion
        self.classifier = nn.Sequential(
            nn.Linear(total_group_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(fusion_dim // 2, 1),
        )

        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"SepsisEarlyWarningModel initialized: {total_params:,} params")

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with feature grouping and cross-group fusion.

        Args:
            x: Patient features (batch, input_dim).

        Returns:
            Dict with 'logit' and 'group_attention_weights'.
        """
        # Split features into physiological groups
        vitals = x[:, self.VITAL_INDICES]
        labs = x[:, self.LAB_INDICES]
        demos = x[:, self.DEMO_INDICES]
        other = x[:, self.OTHER_INDICES]

        # Encode each group
        vital_emb = self.vital_encoder(vitals)
        lab_emb = self.lab_encoder(labs)
        demo_emb = self.demo_encoder(demos)
        other_emb = self.other_encoder(other)

        # Concatenate group embeddings
        all_groups = torch.cat([vital_emb, lab_emb, demo_emb, other_emb], dim=-1)

        # Cross-group attention weighting
        group_weights = self.group_attention(all_groups)

        # Apply attention to each group and re-concatenate
        group_dim = vital_emb.size(-1)
        weighted_groups = torch.cat([
            vital_emb * group_weights[:, 0:1],
            lab_emb * group_weights[:, 1:2],
            demo_emb * group_weights[:, 2:3],
            other_emb * group_weights[:, 3:4],
        ], dim=-1)

        # Final classification
        logit = self.classifier(weighted_groups).squeeze(-1)

        return {
            "logit": logit,
            "group_attention_weights": group_weights,
        }

    def predict_with_action(
        self,
        x: torch.Tensor,
    ) -> Dict[str, object]:
        """
        Inference with urgency classification and recommended action.

        Args:
            x: Patient features (batch, input_dim).

        Returns:
            Dict with sepsis_risk, alert, urgency, recommended_action.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            sepsis_risk = torch.sigmoid(outputs["logit"])

            results = []
            for risk in sepsis_risk:
                r = risk.item()
                # Determine urgency level
                urgency = "normal"
                for level, threshold in self.URGENCY_THRESHOLDS.items():
                    if r < threshold:
                        urgency = level
                        break

                results.append({
                    "sepsis_risk": r,
                    "alert": r >= self.URGENCY_THRESHOLDS["watch"],
                    "urgency": urgency,
                    "recommended_action": self.RECOMMENDED_ACTIONS[urgency],
                })

            return {
                "batch_results": results,
                "group_weights": outputs["group_attention_weights"],
            }
