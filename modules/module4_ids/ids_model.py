"""
═══════════════════════════════════════════════════════════════
FedMedShield — Intrusion Detection System Model (Module 4)
Deep neural network for classifying network traffic into
5 categories: Normal, DoS, Probe, R2L, U2R.

Architecture:
  Feature Embedding → Residual Blocks → Multi-Scale Feature Fusion
  → Attention-Weighted Classification → 5-class output

The model uses multi-scale feature extraction to capture both
fine-grained and coarse-grained traffic patterns, which is
important because different attack types manifest at different
feature scales (e.g., DoS in connection counts, U2R in binary flags).
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEmbeddingLayer(nn.Module):
    """
    Initial feature embedding with parallel processing paths:
    - Path A: Direct linear projection (captures linear relationships)
    - Path B: Two-layer MLP (captures nonlinear relationships)

    Outputs are concatenated and reduced, giving the model both
    linear and nonlinear views of the input features.
    """

    def __init__(
        self,
        input_dim: int = 41,
        embedding_dim: int = 128,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        # Path A: Linear projection
        self.linear_path = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # Path B: Nonlinear MLP
        self.nonlinear_path = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear_path(x)
        nonlinear_out = self.nonlinear_path(x)
        combined = torch.cat([linear_out, nonlinear_out], dim=-1)
        return self.fusion(combined)


class ResidualBlock(nn.Module):
    """Residual block with pre-activation batch normalization."""

    def __init__(self, dim: int, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MultiScaleFeatureExtractor(nn.Module):
    """
    Extracts features at multiple scales using parallel branches
    with different receptive fields (different hidden dimensions).

    - Fine scale (narrow): captures subtle feature patterns (e.g., flag values)
    - Medium scale: captures moderate feature interactions
    - Coarse scale (wide): captures global traffic behavior patterns

    All scales are concatenated for a comprehensive representation.
    """

    def __init__(
        self,
        input_dim: int = 128,
        fine_dim: int = 32,
        medium_dim: int = 64,
        coarse_dim: int = 128,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        # Fine-grained branch
        self.fine_branch = nn.Sequential(
            nn.Linear(input_dim, fine_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(fine_dim, fine_dim),
            nn.GELU(),
        )

        # Medium-grained branch
        self.medium_branch = nn.Sequential(
            nn.Linear(input_dim, medium_dim),
            nn.BatchNorm1d(medium_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(medium_dim, medium_dim),
            nn.GELU(),
        )

        # Coarse-grained branch
        self.coarse_branch = nn.Sequential(
            nn.Linear(input_dim, coarse_dim),
            nn.BatchNorm1d(coarse_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(coarse_dim, coarse_dim // 2),
            nn.GELU(),
            nn.Linear(coarse_dim // 2, coarse_dim),
            nn.GELU(),
        )

        self.output_dim = fine_dim + medium_dim + coarse_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fine = self.fine_branch(x)
        medium = self.medium_branch(x)
        coarse = self.coarse_branch(x)
        return torch.cat([fine, medium, coarse], dim=-1)


class AttentionClassifier(nn.Module):
    """
    Attention-weighted classifier that learns which feature
    dimensions are most important for each attack class.

    Uses a learnable attention vector per class, allowing the
    model to focus on different features for different attacks
    (e.g., connection count for DoS, root shell flags for U2R).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 5,
        hidden_dim: int = 128,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        # Shared feature refinement
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Per-class attention vectors
        self.class_attention = nn.Parameter(
            torch.randn(num_classes, hidden_dim) * 0.01
        )

        # Final projection per class
        self.class_projections = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_classes)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute class-specific logits with attention weights.

        Returns:
            Tuple of (logits, attention_weights).
        """
        shared_features = self.shared(x)

        # Compute attention-weighted features per class
        logits = []
        attention_weights = []
        for i in range(len(self.class_projections)):
            # Attention: softmax(shared_features ⊙ class_attention_i)
            attn = F.softmax(shared_features * self.class_attention[i], dim=-1)
            attention_weights.append(attn)

            # Attended features → class logit
            attended = shared_features * attn
            logit = self.class_projections[i](attended)
            logits.append(logit)

        logits = torch.cat(logits, dim=-1)
        attention_weights = torch.stack(attention_weights, dim=1)  # (batch, classes, hidden)

        return logits, attention_weights


class IntrusionDetectionModel(nn.Module):
    """
    Complete IDS model combining feature embedding, residual processing,
    multi-scale extraction, and attention-based classification.

    Designed for the NSL-KDD feature set (41 features, 5 attack classes).

    Args:
        input_dim: Number of input features (default: 41).
        embedding_dim: Internal embedding dimension.
        num_classes: Number of attack categories.
        num_residual_blocks: Depth of residual processing.
        dropout_rate: Dropout probability.
    """

    ATTACK_CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    SEVERITY_MAP = {
        "Normal": "low",
        "DoS": "high",
        "Probe": "medium",
        "R2L": "high",
        "U2R": "critical",
    }

    def __init__(
        self,
        input_dim: int = 41,
        embedding_dim: int = 128,
        num_classes: int = 5,
        num_residual_blocks: int = 3,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Stage 1: Feature embedding
        self.embedding = FeatureEmbeddingLayer(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
        )

        # Stage 2: Residual feature processing
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(embedding_dim, dropout_rate) for _ in range(num_residual_blocks)]
        )

        # Stage 3: Multi-scale feature extraction
        self.multi_scale = MultiScaleFeatureExtractor(
            input_dim=embedding_dim,
            fine_dim=32,
            medium_dim=64,
            coarse_dim=128,
            dropout_rate=dropout_rate,
        )

        # Stage 4: Attention-based classification
        self.classifier = AttentionClassifier(
            input_dim=self.multi_scale.output_dim,
            num_classes=num_classes,
            hidden_dim=128,
            dropout_rate=dropout_rate,
        )

        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"IntrusionDetectionModel initialized: "
            f"{total_params:,} total, {trainable:,} trainable, "
            f"{num_classes} classes"
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all stages.

        Args:
            x: Network traffic features (batch, input_dim).

        Returns:
            Dict with 'logits' and 'attention_weights'.
        """
        # Embed
        embedded = self.embedding(x)

        # Residual processing
        processed = self.residual_blocks(embedded)

        # Multi-scale features
        multi_scale_features = self.multi_scale(processed)

        # Classify with attention
        logits, attention_weights = self.classifier(multi_scale_features)

        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "features": multi_scale_features,
        }

    def predict(self, x: torch.Tensor) -> Dict[str, object]:
        """
        Inference with attack type classification and severity.

        Args:
            x: Network traffic features (batch, input_dim).

        Returns:
            Dict with predicted attack type, severity, confidence per sample.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probs = F.softmax(outputs["logits"], dim=-1)
            confidence, predicted_class = torch.max(probs, dim=-1)

            results = []
            for i in range(x.size(0)):
                cls_idx = predicted_class[i].item()
                attack_type = self.ATTACK_CLASSES[cls_idx]
                results.append({
                    "attack_type": attack_type,
                    "severity": self.SEVERITY_MAP[attack_type],
                    "confidence": confidence[i].item(),
                    "probabilities": {
                        name: probs[i][j].item()
                        for j, name in enumerate(self.ATTACK_CLASSES)
                    },
                })

            return {"results": results}


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in IDS datasets.
    Down-weights easy examples (Normal traffic) and focuses on
    hard, minority attack classes (R2L, U2R).

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: Per-class weight tensor. If None, uses uniform weights.
        gamma: Focusing parameter. Higher γ = more focus on hard examples.
        reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model output (batch, num_classes).
            targets: Ground truth class indices (batch,).

        Returns:
            Scalar focal loss.
        """
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()

        # p_t = probability of true class
        p_t = (probs * targets_one_hot).sum(dim=-1)

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Cross-entropy: -log(p_t)
        ce_loss = -torch.log(p_t.clamp(min=1e-8))

        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = (alpha * targets_one_hot).sum(dim=-1)
            loss = alpha_t * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
