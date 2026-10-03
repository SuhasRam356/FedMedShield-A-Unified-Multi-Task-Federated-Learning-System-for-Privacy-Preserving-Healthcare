"""
═══════════════════════════════════════════════════════════════
FedMedShield — EHR Multi-Task Prediction Model (Module 1)
A shared-backbone neural network with three task-specific heads:
  1. Disease Classification (4-class: healthy/diabetes/heart/cancer)
  2. COVID-19 Mortality Prediction (binary)
  3. Sepsis Early Warning (binary)

Architecture:
  SharedBackbone → TaskHead_Disease (4-class softmax)
                 → TaskHead_COVID   (binary sigmoid)
                 → TaskHead_Sepsis  (binary sigmoid)

The shared layers learn general EHR representations while
task heads specialize for each clinical prediction target.
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for deep tabular networks.
    Prevents gradient vanishing and allows the network to learn
    identity mappings when deeper layers aren't needed.

    Architecture: Linear → BatchNorm → GELU → Dropout → Linear → BatchNorm → + skip → GELU
    """

    def __init__(
        self,
        dim: int,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual skip connection."""
        residual = x
        out = self.block(x)
        out = out + residual  # Skip connection
        out = self.activation(out)
        return out


class SharedBackbone(nn.Module):
    """
    Shared feature extraction backbone for multi-task EHR prediction.
    Transforms raw patient features into a rich latent representation
    that all task heads can use.

    Architecture:
      Input (20 features) → Linear(128) → ResidualBlock(128) × 2
                          → Linear(64) → ResidualBlock(64) → Latent (64-dim)
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_residual_blocks: int = 2,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        # Input projection with batch normalization
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Residual blocks at hidden_dim
        self.residual_blocks_1 = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_residual_blocks)]
        )

        # Dimensionality reduction
        self.reduction = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        # Final residual refinement at latent_dim
        self.residual_block_2 = ResidualBlock(latent_dim, dropout_rate * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract shared features from raw patient data.

        Args:
            x: Patient features tensor (batch, input_dim).

        Returns:
            Latent representation (batch, latent_dim).
        """
        x = self.input_layer(x)
        x = self.residual_blocks_1(x)
        x = self.reduction(x)
        x = self.residual_block_2(x)
        return x


class TaskHead(nn.Module):
    """
    Task-specific prediction head that takes the shared backbone's
    latent representation and produces task-specific outputs.

    For classification: outputs class logits (no activation — use with CrossEntropyLoss).
    For binary tasks: outputs single logit (use with BCEWithLogitsLoss).
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 32,
        output_dim: int = 1,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce task-specific prediction from latent features.

        Args:
            x: Latent representation (batch, latent_dim).

        Returns:
            Raw logits (batch, output_dim).
        """
        return self.head(x)


class EHRMultiTaskModel(nn.Module):
    """
    Complete multi-task EHR prediction model combining a shared backbone
    with three task-specific heads.

    This architecture is beneficial for federated learning because:
    1. Shared layers transfer knowledge across tasks
    2. Task-specific heads allow specialization per clinical need
    3. Multi-task learning acts as implicit regularization,
       reducing overfitting on small per-hospital datasets

    Parameters:
        input_dim: Number of patient features (default: 20).
        hidden_dim: Width of hidden layers in backbone (default: 128).
        latent_dim: Dimension of shared representation (default: 64).
        num_disease_classes: Number of disease categories (default: 4).
        dropout_rate: Dropout probability (default: 0.3).
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_disease_classes: int = 4,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_disease_classes = num_disease_classes

        # Shared feature extraction backbone
        self.backbone = SharedBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_residual_blocks=2,
            dropout_rate=dropout_rate,
        )

        # Task 1: Disease classification head (4-class)
        self.disease_head = TaskHead(
            latent_dim=latent_dim,
            hidden_dim=32,
            output_dim=num_disease_classes,
            dropout_rate=dropout_rate * 0.7,
        )

        # Task 2: COVID-19 mortality prediction head (binary)
        self.covid_head = TaskHead(
            latent_dim=latent_dim,
            hidden_dim=32,
            output_dim=1,
            dropout_rate=dropout_rate * 0.7,
        )

        # Task 3: Sepsis early warning head (binary)
        self.sepsis_head = TaskHead(
            latent_dim=latent_dim,
            hidden_dim=32,
            output_dim=1,
            dropout_rate=dropout_rate * 0.7,
        )

        # Initialize weights properly
        self.apply(self._init_weights)

        # Log model size
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"EHRMultiTaskModel initialized: "
            f"{total_params:,} total params, {trainable_params:,} trainable"
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize weights using Kaiming Normal for linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all three task heads.

        Args:
            x: Patient feature tensor (batch, input_dim).

        Returns:
            Dictionary with raw logits for each task:
              - 'disease_logits': (batch, num_disease_classes)
              - 'covid_logit': (batch, 1)
              - 'sepsis_logit': (batch, 1)
        """
        # Shared backbone extracts latent features
        latent = self.backbone(x)

        # Each head produces task-specific predictions
        disease_logits = self.disease_head(latent)
        covid_logit = self.covid_head(latent)
        sepsis_logit = self.sepsis_head(latent)

        return {
            "disease_logits": disease_logits,
            "covid_logit": covid_logit.squeeze(-1),
            "sepsis_logit": sepsis_logit.squeeze(-1),
        }

    def predict(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference-time prediction with probabilities (not raw logits).

        Args:
            x: Patient feature tensor (batch, input_dim).

        Returns:
            Dictionary with:
              - 'disease_probs': (batch, num_disease_classes) softmax probabilities
              - 'covid_prob': (batch,) sigmoid probability
              - 'sepsis_prob': (batch,) sigmoid probability
              - 'disease_class': (batch,) predicted disease class index
              - 'disease_confidence': (batch,) max probability
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)

            disease_probs = F.softmax(outputs["disease_logits"], dim=-1)
            covid_prob = torch.sigmoid(outputs["covid_logit"])
            sepsis_prob = torch.sigmoid(outputs["sepsis_logit"])

            disease_class = torch.argmax(disease_probs, dim=-1)
            disease_confidence = torch.max(disease_probs, dim=-1).values

            return {
                "disease_probs": disease_probs,
                "covid_prob": covid_prob,
                "sepsis_prob": sepsis_prob,
                "disease_class": disease_class,
                "disease_confidence": disease_confidence,
            }


class EHRMultiTaskLoss(nn.Module):
    """
    Combined multi-task loss function with learnable task weights.

    Uses uncertainty-based automatic weighting (Kendall et al., 2018):
    Each task has a learnable log-variance parameter that automatically
    balances the loss contribution based on task difficulty.

    Total Loss = Σ (1/2σ²_i) * L_i + log(σ_i)

    This prevents any single task from dominating training, which is
    critical in federated settings where data distributions vary.
    """

    def __init__(self, num_disease_classes: int = 4) -> None:
        super().__init__()

        # Learnable log-variance parameters (one per task)
        self.log_var_disease = nn.Parameter(torch.zeros(1))
        self.log_var_covid = nn.Parameter(torch.zeros(1))
        self.log_var_sepsis = nn.Parameter(torch.zeros(1))

        # Task-specific loss functions
        self.disease_criterion = nn.CrossEntropyLoss()
        self.covid_criterion = nn.BCEWithLogitsLoss()
        self.sepsis_criterion = nn.BCEWithLogitsLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined multi-task loss.

        Args:
            predictions: Model output dict (disease_logits, covid_logit, sepsis_logit).
            targets: Ground truth dict (disease_label, covid_label, sepsis_label).

        Returns:
            Dictionary with:
              - 'total_loss': Combined weighted loss (for backward()).
              - 'disease_loss': Individual disease classification loss.
              - 'covid_loss': Individual COVID mortality loss.
              - 'sepsis_loss': Individual sepsis prediction loss.
              - 'task_weights': Current learned task weights.
        """
        # Compute individual task losses
        disease_loss = self.disease_criterion(
            predictions["disease_logits"],
            targets["disease_label"],
        )
        covid_loss = self.covid_criterion(
            predictions["covid_logit"],
            targets["covid_label"],
        )
        sepsis_loss = self.sepsis_criterion(
            predictions["sepsis_logit"],
            targets["sepsis_label"],
        )

        # Uncertainty-weighted combination
        # weight_i = 1 / (2 * σ²_i), regularization = log(σ_i)
        precision_disease = torch.exp(-self.log_var_disease)
        precision_covid = torch.exp(-self.log_var_covid)
        precision_sepsis = torch.exp(-self.log_var_sepsis)

        total_loss = (
            precision_disease * disease_loss + self.log_var_disease
            + precision_covid * covid_loss + self.log_var_covid
            + precision_sepsis * sepsis_loss + self.log_var_sepsis
        )

        return {
            "total_loss": total_loss.squeeze(),
            "disease_loss": disease_loss.detach(),
            "covid_loss": covid_loss.detach(),
            "sepsis_loss": sepsis_loss.detach(),
            "task_weights": {
                "disease": precision_disease.item(),
                "covid": precision_covid.item(),
                "sepsis": precision_sepsis.item(),
            },
        }
