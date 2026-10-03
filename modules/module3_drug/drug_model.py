"""
═══════════════════════════════════════════════════════════════
FedMedShield — Drug-Protein Binding Affinity Model (Module 3)
Predicts binding affinity between drug compounds and protein targets
using a dual-encoder architecture with cross-attention fusion.

Architecture:
  DrugEncoder(512-dim fingerprint) → drug embedding (128-dim)
  ProteinEncoder(256-dim sequence)  → protein embedding (128-dim)
  CrossAttentionFusion(drug_emb, protein_emb) → interaction features
  AffinityHead → binding affinity score [0, 1]
  ViabilityHead → viable / not viable (binary)
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DrugEncoder(nn.Module):
    """
    Encodes drug molecular fingerprint vectors into a dense embedding.
    Handles the sparse binary + continuous Morgan fingerprint representation.

    Architecture: Input(512) → Linear(256) → BN → GELU → Dropout
                            → Linear(128) → BN → GELU → Embedding(128)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode drug fingerprint → dense embedding."""
        return self.encoder(x)


class ProteinEncoder(nn.Module):
    """
    Encodes protein sequence embedding vectors into a dense representation.

    Architecture: Input(256) → Linear(192) → BN → GELU → Dropout
                            → Linear(128) → BN → Embedding(128)
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 192,
        embedding_dim: int = 128,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode protein sequence → dense embedding."""
        return self.encoder(x)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention mechanism that models the interaction between
    drug and protein embeddings. The drug "attends" to protein features
    and vice versa, capturing binding-relevant interactions.

    This is more expressive than simple concatenation or element-wise
    product, allowing the model to learn complex binding patterns.
    """

    def __init__(self, embedding_dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        # Drug attends to protein
        self.drug_to_protein = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Protein attends to drug
        self.protein_to_drug = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Layer norm for stability
        self.norm_drug = nn.LayerNorm(embedding_dim)
        self.norm_protein = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        drug_emb: torch.Tensor,
        protein_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-attention interaction features.

        Args:
            drug_emb: Drug embedding (batch, embedding_dim).
            protein_emb: Protein embedding (batch, embedding_dim).

        Returns:
            Interaction feature vector (batch, embedding_dim * 4).
        """
        # Reshape for attention: (batch, 1, dim) — treating each embedding as a single token
        drug_seq = drug_emb.unsqueeze(1)
        protein_seq = protein_emb.unsqueeze(1)

        # Cross-attention
        drug_attended, _ = self.drug_to_protein(
            query=drug_seq, key=protein_seq, value=protein_seq
        )
        protein_attended, _ = self.protein_to_drug(
            query=protein_seq, key=drug_seq, value=drug_seq
        )

        # Squeeze back
        drug_attended = self.norm_drug(drug_attended.squeeze(1) + drug_emb)
        protein_attended = self.norm_protein(protein_attended.squeeze(1) + protein_emb)

        # Combine: attended embeddings + element-wise product + absolute difference
        interaction = torch.cat([
            drug_attended,
            protein_attended,
            drug_attended * protein_attended,          # Element-wise interaction
            torch.abs(drug_attended - protein_attended),  # Distance features
        ], dim=-1)

        return interaction


class DrugProteinBindingModel(nn.Module):
    """
    Complete drug-protein binding affinity prediction model.

    Uses dual encoders with cross-attention fusion to predict:
    1. Binding affinity score (continuous, 0-1)
    2. Compound viability (binary: viable if affinity > threshold)

    This dual-head approach is useful because:
    - Affinity provides granular ranking of compounds
    - Viability gives a clear go/no-go decision
    - Joint training improves both through shared representations

    Args:
        drug_dim: Dimensionality of drug fingerprint input.
        protein_dim: Dimensionality of protein embedding input.
        embedding_dim: Size of internal embeddings.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        drug_dim: int = 512,
        protein_dim: int = 256,
        embedding_dim: int = 128,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.drug_dim = drug_dim
        self.protein_dim = protein_dim

        # Dual encoders
        self.drug_encoder = DrugEncoder(
            input_dim=drug_dim,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
        )
        self.protein_encoder = ProteinEncoder(
            input_dim=protein_dim,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
        )

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            embedding_dim=embedding_dim,
            num_heads=4,
        )

        # Binding affinity regression head
        fusion_dim = embedding_dim * 4  # From CrossAttentionFusion output
        self.affinity_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Viability classification head (shares early fusion features)
        self.viability_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"DrugProteinBindingModel initialized: {total_params:,} params")

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        drug_features: torch.Tensor,
        protein_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: encode drug & protein, fuse, predict affinity + viability.

        Args:
            drug_features: Drug fingerprint (batch, drug_dim).
            protein_features: Protein embedding (batch, protein_dim).

        Returns:
            Dict with 'binding_affinity' and 'viability_logit'.
        """
        # Encode
        drug_emb = self.drug_encoder(drug_features)
        protein_emb = self.protein_encoder(protein_features)

        # Fuse with cross-attention
        interaction = self.fusion(drug_emb, protein_emb)

        # Predict
        binding_affinity = self.affinity_head(interaction).squeeze(-1)
        viability_logit = self.viability_head(interaction).squeeze(-1)

        return {
            "binding_affinity": binding_affinity,
            "viability_logit": viability_logit,
            "drug_embedding": drug_emb,
            "protein_embedding": protein_emb,
        }

    def predict(
        self,
        drug_features: torch.Tensor,
        protein_features: torch.Tensor,
        viability_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference-time prediction with viability classification.

        Args:
            drug_features: Drug fingerprint (batch, drug_dim).
            protein_features: Protein embedding (batch, protein_dim).
            viability_threshold: Threshold for viable classification.

        Returns:
            Dict with affinity, viability probability, viable flag, efficacy.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(drug_features, protein_features)

            viability_prob = torch.sigmoid(outputs["viability_logit"])
            viable = viability_prob > viability_threshold

            # Predicted efficacy: weighted combination of affinity and viability
            predicted_efficacy = (
                outputs["binding_affinity"] * 0.7 + viability_prob * 0.3
            ).clamp(0, 1)

            return {
                "binding_affinity": outputs["binding_affinity"],
                "viability_prob": viability_prob,
                "viable": viable,
                "predicted_efficacy": predicted_efficacy,
            }


class DrugBindingLoss(nn.Module):
    """
    Combined loss for drug-protein binding prediction.

    Combines:
    1. MSE loss for binding affinity regression
    2. BCE loss for viability classification
    3. Concordance regularization: ensures affinity and viability agree

    The concordance term penalizes cases where the model predicts
    high affinity but low viability (or vice versa), enforcing
    internal consistency.
    """

    def __init__(
        self,
        affinity_weight: float = 1.0,
        viability_weight: float = 0.5,
        concordance_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.affinity_weight = affinity_weight
        self.viability_weight = viability_weight
        self.concordance_weight = concordance_weight

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined drug binding loss.

        Args:
            predictions: Model outputs (binding_affinity, viability_logit).
            targets: Ground truth (binding_affinity, viability_label).

        Returns:
            Dict with total_loss and individual loss components.
        """
        affinity_loss = self.mse_loss(
            predictions["binding_affinity"],
            targets["binding_affinity"],
        )

        viability_loss = self.bce_loss(
            predictions["viability_logit"],
            targets["viability_label"],
        )

        # Concordance: affinity and viability should agree
        viability_prob = torch.sigmoid(predictions["viability_logit"])
        concordance_loss = self.mse_loss(
            predictions["binding_affinity"],
            viability_prob.detach(),  # Detach to prevent gradient interference
        )

        total_loss = (
            self.affinity_weight * affinity_loss
            + self.viability_weight * viability_loss
            + self.concordance_weight * concordance_loss
        )

        return {
            "total_loss": total_loss,
            "affinity_loss": affinity_loss.detach(),
            "viability_loss": viability_loss.detach(),
            "concordance_loss": concordance_loss.detach(),
        }
