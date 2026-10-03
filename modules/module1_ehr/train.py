"""
═══════════════════════════════════════════════════════════════
FedMedShield — EHR Module Training Script
Handles local training loop for the EHR multi-task model.
Supports standard training, FedProx proximal term, and
per-epoch validation with early stopping.
═══════════════════════════════════════════════════════════════
"""

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple
import logging

from modules.module1_ehr.ehr_model import EHRMultiTaskModel, EHRMultiTaskLoss

logger = logging.getLogger(__name__)


def compute_fedprox_term(
    model: nn.Module,
    global_model: nn.Module,
    mu: float = 0.1,
) -> torch.Tensor:
    """
    Compute FedProx proximal regularization term.

    FedProx adds a penalty proportional to the L2 distance between
    local model weights and the global model weights. This prevents
    local models from drifting too far from the global model during
    training on non-IID data.

    Proximal term = (mu / 2) * Σ ||w_local - w_global||²

    Args:
        model: Current local model being trained.
        global_model: Global model weights from the server.
        mu: Proximal coefficient (higher = stronger regularization).

    Returns:
        Scalar tensor representing the proximal penalty.
    """
    proximal_term = torch.tensor(0.0, device=next(model.parameters()).device)
    for local_param, global_param in zip(model.parameters(), global_model.parameters()):
        proximal_term += torch.sum((local_param - global_param.detach()) ** 2)
    return (mu / 2.0) * proximal_term


def train_one_epoch(
    model: EHRMultiTaskModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: EHRMultiTaskLoss,
    device: torch.device,
    global_model: Optional[nn.Module] = None,
    fedprox_mu: float = 0.0,
) -> Dict[str, float]:
    """
    Train the EHR model for one epoch.

    Args:
        model: Multi-task EHR model.
        train_loader: Training data DataLoader.
        optimizer: Optimizer instance.
        criterion: Multi-task loss function.
        device: Torch device (cpu/cuda).
        global_model: Global model for FedProx (None = standard FedAvg).
        fedprox_mu: FedProx mu coefficient (0 = disabled).

    Returns:
        Dict with epoch-level training metrics.
    """
    model.train()
    epoch_metrics = {
        "total_loss": 0.0,
        "disease_loss": 0.0,
        "covid_loss": 0.0,
        "sepsis_loss": 0.0,
        "num_batches": 0,
        "disease_correct": 0,
        "covid_correct": 0,
        "sepsis_correct": 0,
        "total_samples": 0,
    }

    for batch in train_loader:
        features = batch["features"].to(device)
        targets = {
            "disease_label": batch["disease_label"].to(device),
            "covid_label": batch["covid_label"].to(device),
            "sepsis_label": batch["sepsis_label"].to(device),
        }

        optimizer.zero_grad()

        # Forward pass
        predictions = model(features)
        losses = criterion(predictions, targets)
        loss = losses["total_loss"]

        # Add FedProx proximal term if enabled
        if global_model is not None and fedprox_mu > 0:
            proximal_loss = compute_fedprox_term(model, global_model, fedprox_mu)
            loss = loss + proximal_loss

        # Backward pass
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track metrics
        batch_size = features.size(0)
        epoch_metrics["total_loss"] += loss.item() * batch_size
        epoch_metrics["disease_loss"] += losses["disease_loss"].item() * batch_size
        epoch_metrics["covid_loss"] += losses["covid_loss"].item() * batch_size
        epoch_metrics["sepsis_loss"] += losses["sepsis_loss"].item() * batch_size
        epoch_metrics["num_batches"] += 1
        epoch_metrics["total_samples"] += batch_size

        # Accuracy tracking
        disease_preds = torch.argmax(predictions["disease_logits"], dim=-1)
        epoch_metrics["disease_correct"] += (disease_preds == targets["disease_label"]).sum().item()

        covid_preds = (torch.sigmoid(predictions["covid_logit"]) > 0.5).float()
        epoch_metrics["covid_correct"] += (covid_preds == targets["covid_label"]).sum().item()

        sepsis_preds = (torch.sigmoid(predictions["sepsis_logit"]) > 0.5).float()
        epoch_metrics["sepsis_correct"] += (sepsis_preds == targets["sepsis_label"]).sum().item()

    # Compute averages
    n = epoch_metrics["total_samples"]
    return {
        "total_loss": epoch_metrics["total_loss"] / n,
        "disease_loss": epoch_metrics["disease_loss"] / n,
        "covid_loss": epoch_metrics["covid_loss"] / n,
        "sepsis_loss": epoch_metrics["sepsis_loss"] / n,
        "disease_accuracy": epoch_metrics["disease_correct"] / n,
        "covid_accuracy": epoch_metrics["covid_correct"] / n,
        "sepsis_accuracy": epoch_metrics["sepsis_correct"] / n,
    }


@torch.no_grad()
def evaluate(
    model: EHRMultiTaskModel,
    data_loader: DataLoader,
    criterion: EHRMultiTaskLoss,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the EHR model on validation/test data.

    Args:
        model: Trained multi-task model.
        data_loader: Validation or test DataLoader.
        criterion: Loss function.
        device: Torch device.

    Returns:
        Dict with evaluation metrics.
    """
    model.eval()
    metrics = {
        "total_loss": 0.0,
        "disease_correct": 0,
        "covid_correct": 0,
        "sepsis_correct": 0,
        "total_samples": 0,
    }

    all_disease_preds: List[int] = []
    all_disease_labels: List[int] = []
    all_covid_preds: List[float] = []
    all_covid_labels: List[float] = []
    all_sepsis_preds: List[float] = []
    all_sepsis_labels: List[float] = []

    for batch in data_loader:
        features = batch["features"].to(device)
        targets = {
            "disease_label": batch["disease_label"].to(device),
            "covid_label": batch["covid_label"].to(device),
            "sepsis_label": batch["sepsis_label"].to(device),
        }

        predictions = model(features)
        losses = criterion(predictions, targets)

        batch_size = features.size(0)
        metrics["total_loss"] += losses["total_loss"].item() * batch_size
        metrics["total_samples"] += batch_size

        # Disease accuracy
        disease_preds = torch.argmax(predictions["disease_logits"], dim=-1)
        metrics["disease_correct"] += (disease_preds == targets["disease_label"]).sum().item()
        all_disease_preds.extend(disease_preds.cpu().tolist())
        all_disease_labels.extend(targets["disease_label"].cpu().tolist())

        # COVID accuracy
        covid_probs = torch.sigmoid(predictions["covid_logit"])
        covid_preds = (covid_probs > 0.5).float()
        metrics["covid_correct"] += (covid_preds == targets["covid_label"]).sum().item()
        all_covid_preds.extend(covid_probs.cpu().tolist())
        all_covid_labels.extend(targets["covid_label"].cpu().tolist())

        # Sepsis accuracy
        sepsis_probs = torch.sigmoid(predictions["sepsis_logit"])
        sepsis_preds = (sepsis_probs > 0.5).float()
        metrics["sepsis_correct"] += (sepsis_preds == targets["sepsis_label"]).sum().item()
        all_sepsis_preds.extend(sepsis_probs.cpu().tolist())
        all_sepsis_labels.extend(targets["sepsis_label"].cpu().tolist())

    n = metrics["total_samples"]
    return {
        "total_loss": metrics["total_loss"] / n,
        "disease_accuracy": metrics["disease_correct"] / n,
        "covid_accuracy": metrics["covid_correct"] / n,
        "sepsis_accuracy": metrics["sepsis_correct"] / n,
        "avg_accuracy": (
            metrics["disease_correct"] + metrics["covid_correct"] + metrics["sepsis_correct"]
        ) / (3 * n),
    }


def train_ehr_model(
    model: EHRMultiTaskModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    global_model: Optional[nn.Module] = None,
    fedprox_mu: float = 0.0,
    patience: int = 3,
) -> Dict[str, object]:
    """
    Complete training loop for the EHR multi-task model.

    Includes learning rate scheduling (cosine annealing), gradient clipping,
    early stopping, and best model checkpointing.

    Args:
        model: Multi-task EHR model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Torch device.
        num_epochs: Number of local epochs.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization.
        global_model: Global model for FedProx.
        fedprox_mu: FedProx mu.
        patience: Early stopping patience.

    Returns:
        Dict with training history and best model state dict.
    """
    model = model.to(device)
    criterion = EHRMultiTaskLoss().to(device)

    # Optimizer: AdamW with weight decay
    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
    )

    # Move global model to device if provided
    if global_model is not None:
        global_model = global_model.to(device)

    best_val_accuracy = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            global_model=global_model, fedprox_mu=fedprox_mu,
        )

        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Combine metrics
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["total_loss"],
            "train_disease_acc": train_metrics["disease_accuracy"],
            "train_covid_acc": train_metrics["covid_accuracy"],
            "train_sepsis_acc": train_metrics["sepsis_accuracy"],
            "val_loss": val_metrics["total_loss"],
            "val_disease_acc": val_metrics["disease_accuracy"],
            "val_covid_acc": val_metrics["covid_accuracy"],
            "val_sepsis_acc": val_metrics["sepsis_accuracy"],
            "val_avg_acc": val_metrics["avg_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
        }
        history.append(epoch_result)

        logger.info(
            f"  Epoch {epoch + 1}/{num_epochs} "
            f"| Train Loss: {train_metrics['total_loss']:.4f} "
            f"| Val Acc: {val_metrics['avg_accuracy']:.4f} "
            f"(disease={val_metrics['disease_accuracy']:.3f}, "
            f"covid={val_metrics['covid_accuracy']:.3f}, "
            f"sepsis={val_metrics['sepsis_accuracy']:.3f}) "
            f"| {epoch_time:.1f}s"
        )

        # Best model tracking + early stopping
        if val_metrics["avg_accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["avg_accuracy"]
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
                break

    total_time = time.time() - start_time

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "total_training_time": total_time,
        "model_state_dict": best_model_state or model.state_dict(),
    }
