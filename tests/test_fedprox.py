"""
Unit Tests for FedProx Client and Proximal Regularizer Mathematics.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import pytest

from fl_benchmark.models.cnn import FedAvgCNN
from fl_benchmark.federated.fedavg import FedAvgClient
from fl_benchmark.federated.fedprox import FedProxClient


def test_fedprox_mu_zero_equivalence():
    """Verify FedProx with mu=0 is mathematically equivalent to FedAvg."""
    torch.manual_seed(42)
    x = torch.randn(64, 3, 28, 28)
    y = torch.randint(0, 9, (64,))

    dataset_avg = TensorDataset(x.clone(), y.clone())
    dataset_prox = TensorDataset(x.clone(), y.clone())

    torch.manual_seed(100)
    client_avg = FedAvgClient(
        cid=0, dataset=dataset_avg, batch_size=16, learning_rate=0.01, local_epochs=1
    )
    torch.manual_seed(100)
    client_prox = FedProxClient(
        cid=0, dataset=dataset_prox, mu=0.0, batch_size=16, learning_rate=0.01, local_epochs=1
    )

    global_model = FedAvgCNN(in_channels=3, num_classes=9)
    torch.manual_seed(999)
    model_avg, update_avg, loss_avg = client_avg.train(global_model)

    torch.manual_seed(999)
    model_prox, update_prox, loss_prox = client_prox.train(global_model)

    assert torch.allclose(update_avg, update_prox, atol=1e-5), (
        "FedProx (mu=0) update differs from FedAvg update"
    )
    assert abs(loss_avg - loss_prox) < 1e-5


def test_fedprox_reference_model_immutability():
    """Verify that global reference model is never mutated during FedProx training."""
    x = torch.randn(32, 3, 28, 28)
    y = torch.randint(0, 9, (32,))
    dataset = TensorDataset(x, y)

    client = FedProxClient(
        cid=0, dataset=dataset, mu=0.1, batch_size=16, learning_rate=0.01, local_epochs=2
    )

    global_model = FedAvgCNN(in_channels=3, num_classes=9)
    initial_weights = [p.clone() for p in global_model.parameters()]

    local_model, update, _ = client.train(global_model)

    # Reference model remains untouched
    for p_init, p_curr in zip(initial_weights, global_model.parameters()):
        assert torch.equal(p_init, p_curr), "Global model parameters mutated during FedProx"


def test_fedprox_drift_reduction():
    """Verify that proximal penalty constrains parameter drift (||Delta||_prox < ||Delta||_fedavg)."""
    torch.manual_seed(42)
    x = torch.randn(64, 3, 28, 28)
    y = torch.randint(0, 9, (64,))

    dataset_avg = TensorDataset(x.clone(), y.clone())
    dataset_prox = TensorDataset(x.clone(), y.clone())

    torch.manual_seed(42)
    global_model = FedAvgCNN(in_channels=3, num_classes=9)

    torch.manual_seed(123)
    client_avg = FedAvgClient(
        cid=0, dataset=dataset_avg, batch_size=16, learning_rate=0.05, local_epochs=3
    )
    _, update_avg, _ = client_avg.train(global_model)
    norm_avg = torch.linalg.norm(update_avg).item()

    torch.manual_seed(123)
    # High proximal regularizer
    client_prox = FedProxClient(
        cid=0, dataset=dataset_prox, mu=1.0, batch_size=16, learning_rate=0.05, local_epochs=3
    )
    _, update_prox, _ = client_prox.train(global_model)
    norm_prox = torch.linalg.norm(update_prox).item()

    assert norm_prox < norm_avg, f"Expected norm_prox ({norm_prox}) < norm_avg ({norm_avg})"
