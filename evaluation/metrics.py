"""
═══════════════════════════════════════════════════════════════
FedMedShield — IEEE Metrics Calculator
Standardized evaluation metrics for multi-class and binary
medical predictions. Uses scikit-learn to compute:
- Accuracy, Precision, Recall, F1-Score (Macro/Micro)
- AUROC (Area Under Receiver Operating Characteristic)
- AUPRC (Area Under Precision-Recall Curve)
═══════════════════════════════════════════════════════════════
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from typing import Dict, Any, List

def compute_classification_metrics(
    y_true: np.ndarray, 
    y_pred_probs: np.ndarray, 
    is_multiclass: bool = False
) -> Dict[str, Any]:
    """
    Computes standard IEEE metrics.
    
    Args:
        y_true: Array of true integer labels.
        y_pred_probs: Array of predicted probabilities. 
                      Shape (N,) for binary, (N, C) for multiclass.
        is_multiclass: True if C > 2.
    """
    
    # 1. Get hard predictions
    if is_multiclass:
        y_pred = np.argmax(y_pred_probs, axis=1)
    else:
        y_pred = (y_pred_probs >= 0.5).astype(int)

    # 2. Base Metrics
    acc = accuracy_score(y_true, y_pred)
    
    # Macro-averaged metrics (treats all classes equally, good for imbalanced medical data)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # 3. AUROC & AUPRC
    try:
        if is_multiclass:
            # One-vs-Rest strategy for multiclass AUROC
            auroc = roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="macro")
            # Multiclass AUPRC via one-hot binarization
            from sklearn.preprocessing import label_binarize
            classes = np.unique(y_true)
            if len(classes) > 1:
                y_bin = label_binarize(y_true, classes=classes)
                if y_pred_probs.shape[1] == y_bin.shape[1]:
                    auprc = average_precision_score(y_bin, y_pred_probs, average="macro")
                else:
                    auprc = None
            else:
                auprc = None
        else:
            auroc = roc_auc_score(y_true, y_pred_probs)
            auprc = average_precision_score(y_true, y_pred_probs)
    except ValueError:
        # Happens if only one class is present in y_true during a batch eval
        auroc = 0.5
        auprc = 0.0

    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "auroc": float(auroc) if auroc is not None else None,
        "auprc": float(auprc) if auprc is not None else None,
        "confusion_matrix": cm.tolist()
    }
    
    return metrics

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Metrics for Drug Binding Affinity (MSE, RMSE, MAE, Pearson Correlation, Concordance Index)."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr
    
    y_true_flat = np.asarray(y_true).ravel()
    y_pred_flat = np.asarray(y_pred).ravel()

    mse = mean_squared_error(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = float(np.sqrt(mse))
    
    # Pearson Correlation
    try:
        pearson_corr, _ = pearsonr(y_true_flat, y_pred_flat)
    except Exception:
        pearson_corr = 0.0

    # Concordance Index (C-Index)
    n = len(y_true_flat)
    if n > 1000:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n, size=1000, replace=False)
        y_t, y_p = y_true_flat[sample_idx], y_pred_flat[sample_idx]
        n_sample = 1000
    else:
        y_t, y_p = y_true_flat, y_pred_flat
        n_sample = n

    concordant = 0.0
    total = 0.0
    for i in range(n_sample):
        for j in range(i + 1, n_sample):
            if y_t[i] != y_t[j]:
                total += 1.0
                if (y_t[i] > y_t[j] and y_p[i] > y_p[j]) or (y_t[i] < y_t[j] and y_p[i] < y_p[j]):
                    concordant += 1.0
                elif y_p[i] == y_p[j]:
                    concordant += 0.5
    c_index = float(concordant / total) if total > 0 else 0.5
        
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "pearson_r": float(pearson_corr),
        "concordance_index": c_index
    }
