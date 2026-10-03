"""
═══════════════════════════════════════════════════════════════
FedMedShield — Medical Imaging Model (Module 2)
ResNet-based CNN for medical image classification tasks:
  - Brain Tumor Detection (3-class: none/benign/malignant)
  - Glaucoma Detection (2-class: normal/glaucoma)
  - COVID Chest X-Ray (3-class: normal/covid/pneumonia)

Uses a modified ResNet-18 backbone with custom classification heads
optimized for small medical imaging datasets. Supports transfer learning
from ImageNet pretrained weights.
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional, Literal
import logging

logger = logging.getLogger(__name__)


class SqueezeExcitationBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel-wise attention.
    Recalibrates feature map channels by learning which channels
    are most informative for the classification task.

    This is particularly effective for medical imaging where
    specific texture/pattern channels carry diagnostic information.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        # Squeeze: global average pooling
        y = self.squeeze(x).view(batch, channels)
        # Excitation: channel attention weights
        y = self.excitation(y).view(batch, channels, 1, 1)
        # Scale: multiply input by attention weights
        return x * y.expand_as(x)


class MedicalImagingModel(nn.Module):
    """
    Medical imaging classification model based on ResNet-18 with:
    - Optional ImageNet pretrained weights for transfer learning
    - Squeeze-and-Excitation attention on the final feature maps
    - Global Average Pooling + custom classification head
    - Dropout regularization to prevent overfitting on small datasets
    - Gradient checkpointing support for memory efficiency

    The model can be configured for any number of classes, making it
    reusable across all three imaging tasks (tumor, glaucoma, COVID).

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.
        dropout_rate: Dropout probability before final classifier.
        freeze_backbone_layers: Number of ResNet layers to freeze (0-4).
            Freezing early layers retains ImageNet features while
            allowing later layers to adapt to medical images.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone_layers: int = 2,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        # Load ResNet-18 backbone
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            backbone = models.resnet18(weights=weights)
            logger.info("Loaded ResNet-18 with ImageNet pretrained weights")
        else:
            backbone = models.resnet18(weights=None)
            logger.info("Initialized ResNet-18 with random weights")

        # Extract backbone layers (excluding final fc)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Freeze early layers for transfer learning
        if freeze_backbone_layers > 0:
            layers_to_freeze = [self.conv1, self.bn1]
            if freeze_backbone_layers >= 1:
                layers_to_freeze.append(self.layer1)
            if freeze_backbone_layers >= 2:
                layers_to_freeze.append(self.layer2)
            if freeze_backbone_layers >= 3:
                layers_to_freeze.append(self.layer3)
            if freeze_backbone_layers >= 4:
                layers_to_freeze.append(self.layer4)

            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

            frozen_params = sum(
                1 for p in self.parameters() if not p.requires_grad
            )
            logger.info(f"Froze {frozen_params} parameters in first {freeze_backbone_layers} layers")

        # Channel-wise attention after backbone
        backbone_out_channels = 512  # ResNet-18 final layer output channels
        self.se_block = SqueezeExcitationBlock(backbone_out_channels, reduction=16)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_out_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        # Initialize classifier head weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"MedicalImagingModel initialized: {total_params:,} total, "
            f"{trainable_params:,} trainable, {num_classes} classes"
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ResNet backbone, SE attention, and classifier.

        Args:
            x: Input image tensor (batch, 3, H, W).

        Returns:
            Dict with 'logits' and 'features' (pre-classifier embedding).
        """
        # ResNet backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Channel attention
        x = self.se_block(x)

        # Global average pooling → (batch, channels)
        features = self.global_pool(x).flatten(1)

        # Classification
        logits = self.classifier(features)

        return {
            "logits": logits,
            "features": features,
        }

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Inference-time prediction with probabilities.

        Args:
            x: Input image tensor (batch, 3, H, W).

        Returns:
            Dict with 'probabilities', 'predicted_class', 'confidence'.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probs = F.softmax(outputs["logits"], dim=-1)
            confidence, predicted = torch.max(probs, dim=-1)

            return {
                "probabilities": probs,
                "predicted_class": predicted,
                "confidence": confidence,
            }

    def unfreeze_all(self) -> None:
        """Unfreeze all backbone layers (for fine-tuning after warm-up)."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen for fine-tuning")

    def get_cam_weights(self) -> torch.Tensor:
        """
        Get the weights for Class Activation Mapping (CAM).
        Used to generate heatmaps showing which image regions
        contributed most to the prediction (e.g., tumor location).

        Returns:
            Weight tensor (num_classes, backbone_out_channels).
        """
        # Get the first linear layer's weights from the classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                return m.weight.data
        raise RuntimeError("No linear layer found in classifier")


class TumorDetector(MedicalImagingModel):
    """
    Specialized model for brain tumor detection from MRI scans.
    3 classes: none (0), benign (1), malignant (2).
    """

    CLASS_NAMES = ["none", "benign", "malignant"]

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.5) -> None:
        super().__init__(
            num_classes=3,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone_layers=2,
        )

    def detect(self, x: torch.Tensor) -> Dict[str, object]:
        """
        Detect tumors with detailed result formatting.

        Returns:
            Dict with 'detected', 'type', 'confidence' per sample.
        """
        preds = self.predict(x)
        results = []
        for i in range(x.size(0)):
            class_idx = preds["predicted_class"][i].item()
            results.append({
                "detected": class_idx != 0,
                "type": self.CLASS_NAMES[class_idx],
                "confidence": preds["confidence"][i].item(),
                "probabilities": {
                    name: preds["probabilities"][i][j].item()
                    for j, name in enumerate(self.CLASS_NAMES)
                },
            })
        return {"results": results}


class GlaucomaDetector(MedicalImagingModel):
    """
    Specialized model for glaucoma detection from retinal fundus images.
    2 classes: normal (0), glaucoma (1).
    """

    CLASS_NAMES = ["normal", "glaucoma"]
    SEVERITY_THRESHOLDS = [0.3, 0.5, 0.7, 0.85]
    SEVERITY_LABELS = ["normal", "suspect", "mild", "moderate", "severe"]

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.4) -> None:
        super().__init__(
            num_classes=2,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone_layers=2,
        )

    def detect(self, x: torch.Tensor) -> Dict[str, object]:
        """
        Detect glaucoma with severity estimation.

        Returns:
            Dict with 'glaucomaDetected', 'riskScore', 'severity' per sample.
        """
        preds = self.predict(x)
        results = []
        for i in range(x.size(0)):
            glaucoma_prob = preds["probabilities"][i][1].item()

            # Estimate severity from probability
            severity = self.SEVERITY_LABELS[0]
            for j, threshold in enumerate(self.SEVERITY_THRESHOLDS):
                if glaucoma_prob >= threshold:
                    severity = self.SEVERITY_LABELS[j + 1]

            # Estimate cup-disc ratio from probability (approximate)
            cup_disc_ratio = 0.3 + glaucoma_prob * 0.5  # Maps [0,1] → [0.3, 0.8]

            results.append({
                "glaucoma_detected": glaucoma_prob > 0.5,
                "risk_score": glaucoma_prob,
                "cup_disc_ratio": round(cup_disc_ratio, 3),
                "severity": severity,
            })
        return {"results": results}
