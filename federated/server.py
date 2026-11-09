"""
Federated Server Orchestrator.
Coordinates communication rounds, sequential client training, resource checks,
parameter aggregation, update geometry tracking, independent evaluation, and checkpointing.
"""

import time
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fl_benchmark.federated.client import FederatedClient
from fl_benchmark.federated.aggregation import aggregate_models
from fl_benchmark.metrics.update_geometry import analyze_update_geometry
from fl_benchmark.evaluation.evaluator import GlobalEvaluator
from fl_benchmark.evaluation.client_evaluator import ClientValidationEvaluator
from fl_benchmark.tracking.local_logger import LocalLogger
from fl_benchmark.tracking.wandb_logger import WandbLogger
from fl_benchmark.utils.resource_monitor import FederatedResourceMonitor
from fl_benchmark.utils.checkpoint import CheckpointManager


class FederatedServer:
    """
    Central coordinator for federated learning benchmark.
    Executes sequential client training to strictly conserve physical memory on Windows.
    """

    def __init__(
        self,
        global_model: nn.Module,
        clients: List[FederatedClient],
        val_loader: DataLoader,
        client_val_loaders: List[DataLoader],
        total_rounds: int = 50,
        checkpoint_manager: Optional[CheckpointManager] = None,
        local_logger: Optional[LocalLogger] = None,
        wandb_logger: Optional[WandbLogger] = None,
        resource_monitor: Optional[FederatedResourceMonitor] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.global_model = global_model
        self.clients = clients
        self.num_clients = len(clients)
        self.total_rounds = total_rounds
        self.device = device if device is not None else torch.device("cpu")
        self.config = config or {}

        self.global_evaluator = GlobalEvaluator(device=self.device)
        self.client_evaluator = ClientValidationEvaluator(
            val_loaders=client_val_loaders, device=self.device
        )
        self.checkpoint_manager = checkpoint_manager
        self.local_logger = local_logger
        self.wandb_logger = wandb_logger
        self.resource_monitor = resource_monitor or FederatedResourceMonitor()
        self.val_loader = val_loader

        self.best_val_acc = 0.0
        self.current_round = 0

    def fit_round(self, round_idx: int) -> Dict[str, Any]:
        """
        Execute one communication round across all federated clients sequentially.
        """
        start_time = time.time()
        self.current_round = round_idx

        # 1. Resource pre-flight check
        action, msg = self.resource_monitor.evaluate_state()
        if action == "GRACEFUL_STOP":
            print(f"\n[SERVER HALT] {msg}")
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save_round(
                    model=self.global_model,
                    round_idx=round_idx,
                    metrics={"halt_reason": msg},
                    config=self.config,
                )
            raise MemoryError(f"Graceful halt triggered by resource monitor: {msg}")

        # 2. Sequential Client Execution
        client_models = []
        client_updates = []
        client_sample_counts = []
        client_losses = []

        for cid, client in enumerate(self.clients):
            # Check pause threshold
            action_c, _ = self.resource_monitor.evaluate_state()
            if action_c == "DO_NOT_START_CLIENT":
                print(f"[PAUSE] Low memory detected before Client {cid}. Triggering garbage collection...")
                gc.collect()
                time.sleep(1.0)

            c_start = time.time()
            trained_model, c_update, c_loss = client.train(self.global_model)
            c_elapsed = time.time() - c_start

            client_models.append(trained_model)
            client_updates.append(c_update)
            client_sample_counts.append(client.num_samples)
            client_losses.append(c_loss)

        # 3. Model Aggregation (Sample-weighted)
        self.global_model = aggregate_models(
            models=client_models,
            sample_counts=client_sample_counts,
        )

        # Explicit memory cleanup of client models
        del client_models
        gc.collect()

        # 4. Update Geometry Analysis
        geom_metrics = analyze_update_geometry(
            client_updates=client_updates,
            client_sample_counts=client_sample_counts,
        )
        del client_updates

        # 5. Global Validation Evaluation
        val_metrics = self.global_evaluator.evaluate(self.global_model, self.val_loader)
        is_best = val_metrics["accuracy"] > self.best_val_acc
        if is_best:
            self.best_val_acc = val_metrics["accuracy"]

        # 6. Per-Client Validation Evaluation
        client_val_res = self.client_evaluator.evaluate_clients(self.global_model)

        round_elapsed = time.time() - start_time
        telemetry = self.resource_monitor.get_system_telemetry()

        # Average training loss across participating clients
        avg_train_loss = float(sum(client_losses) / len(client_losses))

        # 7. Compile Round Metrics
        metrics = {
            "round": round_idx,
            "train/avg_loss": round(avg_train_loss, 6),
            "eval/val_loss": val_metrics["loss"],
            "eval/val_acc": val_metrics["accuracy"],
            "eval/val_macro_f1": val_metrics["macro_f1"],
            "eval/best_val_acc": round(self.best_val_acc, 6),
            "eval/client_mean_val_acc": client_val_res["mean_val_accuracy"],
            "eval/client_mean_val_loss": client_val_res["mean_val_loss"],
            "geometry/global_update_norm": geom_metrics["global_update_norm"],
            "geometry/client_update_norms": geom_metrics["update_l2_norms"],
            "geometry/alignment_with_global": geom_metrics["alignment_with_global"],
            "geometry/pairwise_cosine_matrix": geom_metrics["pairwise_cosine_matrix"],
            "system/process_rss_mb": telemetry["process_rss_mb"],
            "system/ram_available_mb": telemetry["ram_available_mb"],
            "system/cpu_percent": telemetry["cpu_utilization_pct"],
            "time/round_seconds": round(round_elapsed, 2),
        }

        # 8. Checkpoint Serialization
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.save_round(
                model=self.global_model,
                round_idx=round_idx,
                metrics=metrics,
                config=self.config,
                is_best=is_best,
            )

        # 9. Logging
        if self.local_logger is not None:
            self.local_logger.log_metrics(step=round_idx, metrics=metrics)
            self.local_logger.print_round_progress(
                round_idx=round_idx,
                total_rounds=self.total_rounds,
                train_loss=avg_train_loss,
                val_acc=val_metrics["accuracy"],
                val_loss=val_metrics["loss"],
                elapsed_sec=round_elapsed,
                ram_avail_mb=telemetry["ram_available_mb"],
            )

        if self.wandb_logger is not None:
            # Flatten metrics for W&B
            flat_wandb_metrics = {
                "round": round_idx,
                "train/avg_loss": avg_train_loss,
                "eval/val_loss": val_metrics["loss"],
                "eval/val_acc": val_metrics["accuracy"],
                "eval/val_macro_f1": val_metrics["macro_f1"],
                "eval/best_val_acc": self.best_val_acc,
                "geometry/global_update_norm": geom_metrics["global_update_norm"],
                "system/ram_available_mb": telemetry["ram_available_mb"],
                "system/process_rss_mb": telemetry["process_rss_mb"],
            }
            for k in range(self.num_clients):
                flat_wandb_metrics[f"geometry/client_{k}_update_norm"] = geom_metrics["update_l2_norms"][k]
                flat_wandb_metrics[f"geometry/client_{k}_alignment"] = geom_metrics["alignment_with_global"][k]
                flat_wandb_metrics[f"eval/client_{k}_val_acc"] = client_val_res["per_client"][k]["accuracy"]
            self.wandb_logger.log(flat_wandb_metrics, step=round_idx)

        return metrics

    def run(self) -> Dict[str, Any]:
        """
        Run complete federated training loop across all rounds.
        """
        print(f"\n{'='*70}")
        print(f"Starting Federated Training: {self.total_rounds} Rounds, {self.num_clients} Clients")
        print(f"Algorithm: {self.config.get('federated', {}).get('algorithm', 'unknown').upper()}")
        print(f"{'='*70}\n")

        all_round_metrics = []
        for r in range(1, self.total_rounds + 1):
            r_metrics = self.fit_round(round_idx=r)
            all_round_metrics.append(r_metrics)

        print(f"\n{'='*70}")
        print(f"Federated Training Complete. Best Val Acc: {self.best_val_acc * 100:.2f}%")
        print(f"{'='*70}\n")

        return {
            "best_val_acc": self.best_val_acc,
            "rounds_completed": len(all_round_metrics),
            "final_metrics": all_round_metrics[-1] if all_round_metrics else {},
        }
