"""
═══════════════════════════════════════════════════════════════
FedMedShield — Medical Imaging Training Script (Module 2)
Handles local training loop for all imaging tasks (tumor, glaucoma, COVID).
Supports standard training and FedProx with learning rate warm-up
and progressive unfreezing for transfer learning.
═══════════════════════════════════════════════════════════════
"""

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import logging

from modules.module2_imaging.imaging_model import MedicalImagingModel

logger = logging.getLogger(__name__)


def compute_fedprox_term(
    model: nn.Module,
    global_model: nn.Module,
    mu: float = 0.1,
) -> torch.Tensor:
    """Compute FedProx proximal regularization term."""
    proximal_term = torch.tensor(0.0, device=next(model.parameters()).device)
    for local_param, global_param in zip(model.parameters(), global_model.parameters()):
        if local_param.requires_grad:
            proximal_term += torch.sum((local_param - global_param.detach()) ** 2)
    return (mu / 2.0) * proximal_term


def train_one_epoch(
    model: MedicalImagingModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    global_model: Optional[nn.Module] = None,
    fedprox_mu: float = 0.0,
) -> Dict[str, float]:
    """
    Train the imaging model for one epoch.

    Args:
        model: Medical imaging model.
        train_loader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function (CrossEntropyLoss).
        device: Torch device.
        global_model: For FedProx regularization.
        fedprox_mu: FedProx coefficient.

    Returns:
        Dict with training loss and accuracy.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs["logits"], labels)

        # FedProx proximal term
        if global_model is not None and fedprox_mu > 0:
            loss = loss + compute_fedprox_term(model, global_model, fedprox_mu)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs["logits"], dim=-1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(
    model: MedicalImagingModel,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the imaging model.

    Args:
        model: Trained model.
        data_loader: Validation/test DataLoader.
        criterion: Loss function.
        device: Torch device.

    Returns:
        Dict with evaluation loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in data_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = criterion(outputs["logits"], labels)

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs["logits"], dim=-1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Per-class accuracy
    num_classes = model.num_classes
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    per_class_acc = {
        f"class_{i}_accuracy": class_correct[i] / max(class_total[i], 1)
        for i in range(num_classes)
    }

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        **per_class_acc,
    }


def train_imaging_model(
    model: MedicalImagingModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 5,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    global_model: Optional[nn.Module] = None,
    fedprox_mu: float = 0.0,
    patience: int = 3,
    warmup_epochs: int = 1,
    unfreeze_at_epoch: Optional[int] = None,
) -> Dict[str, object]:
    """
    Complete training loop for medical imaging model.

    Features:
    - Learning rate warm-up for stable transfer learning
    - Progressive unfreezing of backbone layers
    - Early stopping with best model tracking
    - Label smoothing for regularization

    Args:
        model: Medical imaging model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Torch device.
        num_epochs: Number of local epochs.
        learning_rate: Peak learning rate.
        weight_decay: L2 regularization.
        global_model: For FedProx.
        fedprox_mu: FedProx coefficient.
        patience: Early stopping patience.
        warmup_epochs: Number of warmup epochs.
        unfreeze_at_epoch: Epoch at which to unfreeze all layers.

    Returns:
        Dict with training history and best model state.
    """
    model = model.to(device)

    # Label smoothing for better generalization on small datasets
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Use different learning rates for backbone vs classifier
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "classifier" in name or "se_block" in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": learning_rate * 0.1},
        {"params": classifier_params, "lr": learning_rate},
    ], weight_decay=weight_decay)

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(num_epochs, 2), T_mult=1,
    )

    if global_model is not None:
        global_model = global_model.to(device)

    best_val_accuracy = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Progressive unfreezing
        if unfreeze_at_epoch is not None and epoch == unfreeze_at_epoch:
            model.unfreeze_all()
            logger.info(f"  Unfreezing all layers at epoch {epoch + 1}")

        # Learning rate warmup
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] *= warmup_factor

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            global_model=global_model, fedprox_mu=fedprox_mu,
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        epoch_time = time.time() - epoch_start

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": optimizer.param_groups[-1]["lr"],
            "epoch_time": epoch_time,
            **{k: v for k, v in val_metrics.items() if k.startswith("class_")},
        }
        history.append(epoch_result)

        logger.info(
            f"  Epoch {epoch + 1}/{num_epochs} "
            f"| Train Loss: {train_metrics['loss']:.4f} "
            f"| Train Acc: {train_metrics['accuracy']:.4f} "
            f"| Val Acc: {val_metrics['accuracy']:.4f} "
            f"| {epoch_time:.1f}s"
        )

        # Best model + early stopping
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"  Early stopping at epoch {epoch + 1}")
                break

    total_time = time.time() - start_time

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "total_training_time": total_time,
        "model_state_dict": best_model_state or model.state_dict(),
    }
