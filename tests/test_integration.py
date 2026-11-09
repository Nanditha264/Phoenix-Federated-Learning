"""
Integration Tests for Federated Learning Framework.
Runs complete 1-round communication loops with 5 clients, aggregation, evaluation, and checkpointing.
"""

from pathlib import Path
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytest

from fl_benchmark.models.cnn import FedAvgCNN
from fl_benchmark.federated.fedavg import FedAvgClient
from fl_benchmark.federated.fedprox import FedProxClient
from fl_benchmark.federated.server import FederatedServer
from fl_benchmark.tracking.local_logger import LocalLogger
from fl_benchmark.utils.checkpoint import CheckpointManager
from fl_benchmark.utils.resource_monitor import FederatedResourceMonitor


def create_mock_synthetic_client_datasets(num_clients=5, samples_per_client=32):
    """Generate small synthetic datasets with 3x28x28 inputs and 9 classes."""
    datasets = []
    for c in range(num_clients):
        x = torch.randn(samples_per_client, 3, 28, 28)
        y = torch.randint(0, 9, (samples_per_client,))
        datasets.append(TensorDataset(x, y))
    return datasets


def test_fedavg_one_round_integration():
    """Verify 1 complete round of FedAvg with 5 clients, evaluation, and checkpointing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        client_datasets = create_mock_synthetic_client_datasets(num_clients=5, samples_per_client=32)
        val_dataset = TensorDataset(torch.randn(50, 3, 28, 28), torch.randint(0, 9, (50,)))
        val_loader = DataLoader(val_dataset, batch_size=16)

        client_val_loaders = [
            DataLoader(TensorDataset(torch.randn(10, 3, 28, 28), torch.randint(0, 9, (10,))), batch_size=10)
            for _ in range(5)
        ]

        global_model = FedAvgCNN(in_channels=3, num_classes=9)
        clients = [
            FedAvgClient(cid=i, dataset=client_datasets[i], batch_size=16, learning_rate=0.01, local_epochs=1)
            for i in range(5)
        ]

        ckpt_mgr = CheckpointManager(checkpoint_dir=str(Path(tmp_dir) / "checkpoints"))
        local_logger = LocalLogger(output_dir=tmp_dir, run_name="test_fedavg_smoke")

        server = FederatedServer(
            global_model=global_model,
            clients=clients,
            val_loader=val_loader,
            client_val_loaders=client_val_loaders,
            total_rounds=1,
            checkpoint_manager=ckpt_mgr,
            local_logger=local_logger,
            wandb_logger=None,
            resource_monitor=FederatedResourceMonitor(),
        )

        results = server.run()

        assert results["rounds_completed"] == 1
        assert "eval/val_acc" in results["final_metrics"]
        assert (Path(tmp_dir) / "checkpoints" / "last_checkpoint.pt").exists()


def test_fedprox_one_round_integration():
    """Verify 1 complete round of FedProx (mu=0.01) with 5 clients, evaluation, and checkpointing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        client_datasets = create_mock_synthetic_client_datasets(num_clients=5, samples_per_client=32)
        val_dataset = TensorDataset(torch.randn(50, 3, 28, 28), torch.randint(0, 9, (50,)))
        val_loader = DataLoader(val_dataset, batch_size=16)

        client_val_loaders = [
            DataLoader(TensorDataset(torch.randn(10, 3, 28, 28), torch.randint(0, 9, (10,))), batch_size=10)
            for _ in range(5)
        ]

        global_model = FedAvgCNN(in_channels=3, num_classes=9)
        clients = [
            FedProxClient(
                cid=i, dataset=client_datasets[i], mu=0.01, batch_size=16, learning_rate=0.01, local_epochs=1
            )
            for i in range(5)
        ]

        ckpt_mgr = CheckpointManager(checkpoint_dir=str(Path(tmp_dir) / "checkpoints"))
        local_logger = LocalLogger(output_dir=tmp_dir, run_name="test_fedprox_smoke")

        server = FederatedServer(
            global_model=global_model,
            clients=clients,
            val_loader=val_loader,
            client_val_loaders=client_val_loaders,
            total_rounds=1,
            checkpoint_manager=ckpt_mgr,
            local_logger=local_logger,
            wandb_logger=None,
            resource_monitor=FederatedResourceMonitor(),
        )

        results = server.run()

        assert results["rounds_completed"] == 1
        assert "eval/val_acc" in results["final_metrics"]
        assert (Path(tmp_dir) / "checkpoints" / "last_checkpoint.pt").exists()
