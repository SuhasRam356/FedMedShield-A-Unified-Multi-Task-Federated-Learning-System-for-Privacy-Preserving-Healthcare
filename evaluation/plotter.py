"""
═══════════════════════════════════════════════════════════════
FedMedShield — IEEE Publication Plotter
Generates high-quality, publication-ready (IEEE formatted)
graphs using Matplotlib and Seaborn.
- ROC Curves
- Confusion Matrices
- Federated Learning Convergence Curves
═══════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Configure Matplotlib for IEEE publication standards
# IEEE typically requires Times New Roman, specific font sizes, and high DPI
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

SAVE_DIR = "evaluation/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_roc_curve(y_true: np.ndarray, y_pred_probs: np.ndarray, filename: str = "roc_curve.pdf", title: str = "ROC Curve"):
    """Plots a standard ROC curve for binary classification."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(3.5, 3.5)) # IEEE single column width is typically 3.5 inches
    plt.plot(fpr, tpr, color='darkorange', lw=1.5, label=f'AUROC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved ROC plot to {save_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, filename: str = "confusion_matrix.pdf"):
    """Plots a heatmap confusion matrix."""
    plt.figure(figsize=(4, 3.5))
    
    # Normalize CM for better color mapping
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, annot_kws={"size": 10})
                
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Confusion Matrix to {save_path}")


def plot_fl_convergence(rounds: list, losses: list, accuracies: list, filename: str = "fl_convergence.pdf"):
    """Plots dual-axis learning curves over FL rounds."""
    fig, ax1 = plt.subplots(figsize=(4.5, 3.5))

    # Loss axis
    color = 'tab:red'
    ax1.set_xlabel('Federated Round')
    ax1.set_ylabel('Global Loss', color=color)
    ax1.plot(rounds, losses, color=color, marker='o', markersize=4, linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Accuracy axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Global Accuracy / F1', color=color)  
    ax2.plot(rounds, accuracies, color=color, marker='s', markersize=4, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Limits
    ax2.set_ylim([0, 1.05])
    
    fig.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved FL Convergence plot to {save_path}")
