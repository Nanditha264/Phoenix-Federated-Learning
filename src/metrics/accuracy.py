"""
Accuracy and Classification Metrics for Federated Learning Evaluation.
Computes top-1 accuracy, per-class accuracy, and Macro-F1.
"""

from typing import Dict, Any, List
import numpy as np
import torch
from sklearn.metrics import f1_score


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Compute top-1 classification accuracy.
    """
    if logits.numel() == 0:
        return 0.0
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).sum().item()
    return float(correct / targets.numel())


def compute_classification_metrics(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    num_classes: int = 9,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics:
    - Overall Top-1 Accuracy
    - Macro-F1 score
    - Per-class accuracy
    """
    all_preds = np.asarray(all_preds).flatten()
    all_targets = np.asarray(all_targets).flatten()
    n_total = len(all_targets)

    if n_total == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class_accuracy": {}}

    overall_acc = float(np.mean(all_preds == all_targets))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))

    per_class_acc = {}
    for c in range(num_classes):
        mask_c = (all_targets == c)
        if np.sum(mask_c) > 0:
            per_class_acc[c] = float(np.mean(all_preds[mask_c] == all_targets[mask_c]))
        else:
            per_class_acc[c] = None

    return {
        "accuracy": round(overall_acc, 6),
        "macro_f1": round(macro_f1, 6),
        "per_class_accuracy": per_class_acc,
    }
