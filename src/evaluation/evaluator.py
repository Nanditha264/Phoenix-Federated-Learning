"""
Independent Global Evaluator for Validation and Test Splits.
Evaluates model accuracy, cross-entropy loss, Macro-F1, and per-class performance.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.metrics.accuracy import compute_classification_metrics
from src.metrics.loss import RunningAverage


class GlobalEvaluator:
    """
    Evaluates global models against clean validation and test splits.
    """

    def __init__(
        self,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        batch_size: int = 64,
    ):
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.device = device if device is not None else torch.device("cpu")
        self.batch_size = batch_size

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_classes: int = 9,
    ) -> Dict[str, Any]:
        """
        Run complete evaluation over a dataset split.
        Returns accuracy, average loss, macro-F1, and per-class accuracy.
        """
        model.eval()
        loss_meter = RunningAverage()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = model(images)
                loss = self.criterion(logits, targets)

                loss_meter.update(loss.item(), n=targets.size(0))
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())

        class_metrics = compute_classification_metrics(
            all_preds=np.array(all_preds),
            all_targets=np.array(all_targets),
            num_classes=num_classes,
        )

        return {
            "loss": round(loss_meter.avg, 6),
            "accuracy": class_metrics["accuracy"],
            "macro_f1": class_metrics["macro_f1"],
            "per_class_accuracy": class_metrics["per_class_accuracy"],
            "total_samples": len(all_targets),
        }
