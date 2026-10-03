"""
═══════════════════════════════════════════════════════════════
FedMedShield — Drug Discovery Training Script (Module 3)
Training loop for the drug-protein binding model with
FedProx support and dual-head optimization.
═══════════════════════════════════════════════════════════════
"""

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import numpy as np
import logging

from modules.module3_drug.drug_model import DrugProteinBindingModel, DrugBindingLoss

logger = logging.getLogger(__name__)


def compute_fedprox_term(
    model: nn.Module, global_model: nn.Module, mu: float
) -> torch.Tensor:
    """FedProx proximal regularization."""
    term = torch.tensor(0.0, device=next(model.parameters()).device)
    for lp, gp in zip(model.parameters(), global_model.parameters()):
        term += torch.sum((lp - gp.detach()) ** 2)
    return (mu / 2.0) * term


def train_one_epoch(
    model: DrugProteinBindingModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: DrugBindingLoss,
    device: torch.device,
    global_model: Optional[nn.Module] = None,
    fedprox_mu: float = 0.0,
) -> Dict[str, float]:
    """Train drug model for one epoch."""
    model.train()
    total_loss = 0.0
    total_affinity_loss = 0.0
    total_viability_loss = 0.0
    affinity_errors = []
    viability_correct = 0
    total = 0

    for batch in train_loader:
        drug_feat = batch["drug_features"].to(device)
        protein_feat = batch["protein_features"].to(device)
        targets = {
            "binding_affinity": batch["binding_affinity"].to(device),
            "viability_label": batch["viability_label"].to(device),
        }

        optimizer.zero_grad()

        predictions = model(drug_feat, protein_feat)
        losses = criterion(predictions, targets)
        loss = losses["total_loss"]

        if global_model is not None and fedprox_mu > 0:
            loss = loss + compute_fedprox_term(model, global_model, fedprox_mu)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = drug_feat.size(0)
        total_loss += loss.item() * bs
        total_affinity_loss += losses["affinity_loss"].item() * bs
        total_viability_loss += losses["viability_loss"].item() * bs
        total += bs

        # Track affinity prediction error (MAE)
        with torch.no_grad():
            mae = torch.abs(
                predictions["binding_affinity"] - targets["binding_affinity"]
            ).mean().item()
            affinity_errors.append(mae)

            # Viability accuracy
            via_preds = (torch.sigmoid(predictions["viability_logit"]) > 0.5).float()
            viability_correct += (via_preds == targets["viability_label"]).sum().item()

    return {
        "total_loss": total_loss / total,
        "affinity_loss": total_affinity_loss / total,
        "viability_loss": total_viability_loss / total,
        "affinity_mae": np.mean(affinity_errors),
        "viability_accuracy": viability_correct / total,
    }


@torch.no_grad()
def evaluate(
    model: DrugProteinBindingModel,
    data_loader: DataLoader,
    criterion: DrugBindingLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate drug model on validation/test data."""
    model.eval()
    total_loss = 0.0
    affinity_errors = []
    viability_correct = 0
    total = 0

    all_affinities_pred = []
    all_affinities_true = []

    for batch in data_loader:
        drug_feat = batch["drug_features"].to(device)
        protein_feat = batch["protein_features"].to(device)
        targets = {
            "binding_affinity": batch["binding_affinity"].to(device),
            "viability_label": batch["viability_label"].to(device),
        }

        predictions = model(drug_feat, protein_feat)
        losses = criterion(predictions, targets)

        bs = drug_feat.size(0)
        total_loss += losses["total_loss"].item() * bs
        total += bs

        mae = torch.abs(
            predictions["binding_affinity"] - targets["binding_affinity"]
        ).mean().item()
        affinity_errors.append(mae)

        via_preds = (torch.sigmoid(predictions["viability_logit"]) > 0.5).float()
        viability_correct += (via_preds == targets["viability_label"]).sum().item()

        all_affinities_pred.extend(predictions["binding_affinity"].cpu().tolist())
        all_affinities_true.extend(targets["binding_affinity"].cpu().tolist())

    # Pearson correlation between predicted and true affinities
    pred_arr = np.array(all_affinities_pred)
    true_arr = np.array(all_affinities_true)
    if len(pred_arr) > 1:
        correlation = np.corrcoef(pred_arr, true_arr)[0, 1]
    else:
        correlation = 0.0

    # R² score
    ss_res = np.sum((true_arr - pred_arr) ** 2)
    ss_tot = np.sum((true_arr - np.mean(true_arr)) ** 2)
    r_squared = 1 - (ss_res / max(ss_tot, 1e-8))

    return {
        "total_loss": total_loss / total,
        "affinity_mae": np.mean(affinity_errors),
        "viability_accuracy": viability_correct / total,
        "pearson_correlation": float(correlation),
        "r_squared": float(r_squared),
    }


def train_drug_model(
    model: DrugProteinBindingModel,
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
    Complete training loop for the drug-protein binding model.

    Args:
        model: Drug binding model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Torch device.
        num_epochs: Local epochs.
        learning_rate: LR.
        weight_decay: L2 reg.
        global_model: For FedProx.
        fedprox_mu: FedProx mu.
        patience: Early stopping patience.

    Returns:
        Training results with history, best metrics, and model state.
    """
    model = model.to(device)
    criterion = DrugBindingLoss(
        affinity_weight=1.0,
        viability_weight=0.5,
        concordance_weight=0.1,
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
    )

    if global_model is not None:
        global_model = global_model.to(device)

    best_correlation = -1.0
    best_model_state = None
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            global_model=global_model, fedprox_mu=fedprox_mu,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["total_loss"],
            "train_mae": train_metrics["affinity_mae"],
            "train_via_acc": train_metrics["viability_accuracy"],
            "val_loss": val_metrics["total_loss"],
            "val_mae": val_metrics["affinity_mae"],
            "val_via_acc": val_metrics["viability_accuracy"],
            "val_correlation": val_metrics["pearson_correlation"],
            "val_r_squared": val_metrics["r_squared"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
        }
        history.append(epoch_result)

        logger.info(
            f"  Epoch {epoch + 1}/{num_epochs} "
            f"| Train Loss: {train_metrics['total_loss']:.4f} "
            f"| Val MAE: {val_metrics['affinity_mae']:.4f} "
            f"| Val Corr: {val_metrics['pearson_correlation']:.4f} "
            f"| Via Acc: {val_metrics['viability_accuracy']:.4f} "
            f"| {epoch_time:.1f}s"
        )

        if val_metrics["pearson_correlation"] > best_correlation:
            best_correlation = val_metrics["pearson_correlation"]
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
        "best_correlation": best_correlation,
        "total_training_time": time.time() - start_time,
        "model_state_dict": best_model_state or model.state_dict(),
    }
