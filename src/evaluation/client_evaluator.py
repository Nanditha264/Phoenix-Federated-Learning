"""
Per-Client Validation Evaluator.
Evaluates the global model across the 5 disjoint clean validation subsets.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.evaluation.evaluator import GlobalEvaluator


class ClientValidationEvaluator:
    """
    Evaluates global models independently on each client's dedicated validation partition.
    """

    def __init__(
        self,
        val_loaders: List[DataLoader],
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        self.val_loaders = val_loaders
        self.num_clients = len(val_loaders)
        self.evaluator = GlobalEvaluator(criterion=criterion, device=device)

    def evaluate_clients(self, model: nn.Module) -> Dict[str, Any]:
        """
        Evaluate model on each client's disjoint validation split.
        Returns:
            Dictionary mapping client_id -> metrics, plus client-averaged accuracy and loss.
        """
        client_metrics = {}
        total_samples = 0
        weighted_acc = 0.0
        weighted_loss = 0.0

        for cid in range(self.num_clients):
            loader = self.val_loaders[cid]
            res = self.evaluator.evaluate(model, loader)
            client_metrics[cid] = res

            n = res["total_samples"]
            total_samples += n
            weighted_acc += res["accuracy"] * n
            weighted_loss += res["loss"] * n

        macro_avg_acc = sum(m["accuracy"] for m in client_metrics.values()) / self.num_clients
        macro_avg_loss = sum(m["loss"] for m in client_metrics.values()) / self.num_clients

        return {
            "per_client": client_metrics,
            "mean_val_accuracy": round(macro_avg_acc, 6),
            "mean_val_loss": round(macro_avg_loss, 6),
            "weighted_val_accuracy": round(weighted_acc / total_samples, 6) if total_samples > 0 else 0.0,
            "weighted_val_loss": round(weighted_loss / total_samples, 6) if total_samples > 0 else 0.0,
        }
