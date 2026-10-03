"""
═══════════════════════════════════════════════════════════════
FedMedShield — IDS Training Script (Module 4)
Training loop for the intrusion detection model with
Focal Loss for class imbalance, FedProx support, and
per-class metric tracking.
═══════════════════════════════════════════════════════════════
"""

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import logging

from modules.module4_ids.ids_model import IntrusionDetectionModel, FocalLoss

logger = logging.getLogger(__name__)

ATTACK_CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# Class weights: inverse frequency weighting for imbalanced classes
# Normal(0.45) DoS(0.30) Probe(0.12) R2L(0.08) U2R(0.05)
DEFAULT_CLASS_WEIGHTS = torch.tensor([1.0, 1.5, 3.0, 5.0, 8.0])


def compute_fedprox_term(
    model: nn.Module, global_model: nn.Module, mu: float
) -> torch.Tensor:
    """FedProx proximal regularization."""
    term = torch.tensor(0.0, device=next(model.parameters()).device)
    for lp, gp in zip(model.parameters(), global_model.parameters()):
        term += torch.sum((lp - gp.detach()) ** 2)
    return (mu / 2.0) * term


def train_one_epoch(
    model: IntrusionDetectionModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    global_model: Optional[nn.Module] = None,
    fedprox_mu: float = 0.0,
) -> Dict[str, float]:
    """Train IDS model for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in train_loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs["logits"], labels)

        if global_model is not None and fedprox_mu > 0:
            loss = loss + compute_fedprox_term(model, global_model, fedprox_mu)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = features.size(0)
        total_loss += loss.item() * bs
        total += bs

        preds = torch.argmax(outputs["logits"], dim=-1)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Macro F1 (treats all classes equally — important for imbalanced data)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


@torch.no_grad()
def evaluate(
    model: IntrusionDetectionModel,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate IDS model with detailed per-class metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in data_loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)

        outputs = model(features)
        loss = criterion(outputs["logits"], labels)

        bs = features.size(0)
        total_loss += loss.item() * bs
        total += bs

        preds = torch.argmax(outputs["logits"], dim=-1)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Overall metrics
    accuracy = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    # Per-class metrics
    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, labels=list(range(5)), zero_division=0
    )
    per_class_precision = precision_score(
        all_labels, all_preds, average=None, labels=list(range(5)), zero_division=0
    )
    per_class_recall = recall_score(
        all_labels, all_preds, average=None, labels=list(range(5)), zero_division=0
    )

    per_class_metrics = {}
    for i, cls_name in enumerate(ATTACK_CLASSES):
        per_class_metrics[f"{cls_name}_f1"] = float(per_class_f1[i])
        per_class_metrics[f"{cls_name}_precision"] = float(per_class_precision[i])
        per_class_metrics[f"{cls_name}_recall"] = float(per_class_recall[i])

    # Detection rate: percentage of attacks correctly identified
    attack_preds = [p for p, l in zip(all_preds, all_labels) if l != 0]
    attack_labels = [l for l in all_labels if l != 0]
    detection_rate = (
        sum(1 for p, l in zip(attack_preds, attack_labels) if p == l) / max(len(attack_labels), 1)
    )

    # False positive rate: normal traffic incorrectly classified as attack
    normal_preds = [p for p, l in zip(all_preds, all_labels) if l == 0]
    false_positive_rate = (
        sum(1 for p in normal_preds if p != 0) / max(len(normal_preds), 1)
    )

    return {
        "loss": total_loss / total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        **per_class_metrics,
    }


def train_ids_model(
    model: IntrusionDetectionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    global_model: Optional[nn.Module] = None,
    fedprox_mu: float = 0.0,
    patience: int = 3,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
) -> Dict[str, object]:
    """
    Complete training loop for the IDS model.

    Args:
        model: IDS model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Torch device.
        num_epochs: Local epochs.
        learning_rate: LR.
        weight_decay: L2 reg.
        global_model: For FedProx.
        fedprox_mu: FedProx mu.
        patience: Early stopping patience.
        use_focal_loss: Whether to use Focal Loss (recommended for imbalanced data).
        focal_gamma: Focal Loss focusing parameter.

    Returns:
        Training results with history, best metrics, and model state.
    """
    model = model.to(device)

    # Loss function: Focal Loss for class imbalance or standard CE
    if use_focal_loss:
        criterion = FocalLoss(
            alpha=DEFAULT_CLASS_WEIGHTS.to(device),
            gamma=focal_gamma,
        )
        logger.info(f"Using Focal Loss (gamma={focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=DEFAULT_CLASS_WEIGHTS.to(device)
        )

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.3,
    )

    if global_model is not None:
        global_model = global_model.to(device)

    best_macro_f1 = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train (note: OneCycleLR steps per batch, handled within train_one_epoch)
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds_train: List[int] = []
        all_labels_train: List[int] = []

        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs["logits"], labels)

            if global_model is not None and fedprox_mu > 0:
                loss = loss + compute_fedprox_term(model, global_model, fedprox_mu)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            bs = features.size(0)
            total_loss += loss.item() * bs
            total += bs
            preds = torch.argmax(outputs["logits"], dim=-1)
            correct += (preds == labels).sum().item()
            all_preds_train.extend(preds.cpu().tolist())
            all_labels_train.extend(labels.cpu().tolist())

        train_accuracy = correct / total
        train_f1 = f1_score(all_labels_train, all_preds_train, average="macro", zero_division=0)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": total_loss / total,
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_f1,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_detection_rate": val_metrics["detection_rate"],
            "val_fpr": val_metrics["false_positive_rate"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
        }
        # Add per-class metrics
        for cls_name in ATTACK_CLASSES:
            for metric in ["f1", "precision", "recall"]:
                key = f"{cls_name}_{metric}"
                if key in val_metrics:
                    epoch_result[f"val_{key}"] = val_metrics[key]

        history.append(epoch_result)

        logger.info(
            f"  Epoch {epoch + 1}/{num_epochs} "
            f"| Train Loss: {total_loss / total:.4f} "
            f"| Val Acc: {val_metrics['accuracy']:.4f} "
            f"| Val F1: {val_metrics['macro_f1']:.4f} "
            f"| Det Rate: {val_metrics['detection_rate']:.4f} "
            f"| FPR: {val_metrics['false_positive_rate']:.4f} "
            f"| {epoch_time:.1f}s"
        )

        # Best model + early stopping (on macro F1, not accuracy — better for imbalanced)
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"  Early stopping at epoch {epoch + 1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        "history": history,
        "best_macro_f1": best_macro_f1,
        "total_training_time": time.time() - start_time,
        "model_state_dict": best_model_state or model.state_dict(),
    }
