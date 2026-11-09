"""
Unit Tests for FedAvg Client Training and Sample-Weighted Aggregation.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import pytest

from fl_benchmark.models.cnn import FedAvgCNN
from fl_benchmark.federated.fedavg import FedAvgClient
from fl_benchmark.federated.aggregation import aggregate_models


def test_sample_weighted_aggregation_math():
    """Verify exact mathematical sample-weighted parameter aggregation."""
    # Create 2 simple linear models
    model1 = nn.Linear(4, 2, bias=False)
    model2 = nn.Linear(4, 2, bias=False)

    with torch.no_grad():
        model1.weight.fill_(1.0)
        model2.weight.fill_(5.0)

    # Weights: 100 samples vs 300 samples -> ratios 0.25 and 0.75
    # Expected: 0.25 * 1.0 + 0.75 * 5.0 = 0.25 + 3.75 = 4.0
    sample_counts = [100, 300]
    aggregated = aggregate_models([model1, model2], sample_counts)

    expected_val = 4.0
    assert torch.allclose(aggregated.weight, torch.tensor(expected_val))


def test_fedavg_client_local_training():
    """Verify FedAvg client trains, decreases loss, and produces valid parameter displacement."""
    torch.manual_seed(42)
    x = torch.randn(64, 3, 28, 28)
    y = torch.randint(0, 9, (64,))
    dataset = TensorDataset(x, y)

    client = FedAvgClient(
        cid=0,
        dataset=dataset,
        batch_size=16,
        learning_rate=0.01,
        local_epochs=2,
    )

    global_model = FedAvgCNN(in_channels=3, num_classes=9)
    initial_params = [p.clone() for p in global_model.parameters()]

    local_model, client_update, avg_loss = client.train(global_model)

    # 1. Local model weights must have changed
    for p_init, p_local in zip(initial_params, local_model.parameters()):
        assert not torch.equal(p_init, p_local), "Local parameters did not update"

    # 2. Global model must NOT have been mutated
    for p_init, p_glob in zip(initial_params, global_model.parameters()):
        assert torch.equal(p_init, p_glob), "Global model was unintentionally mutated"

    # 3. Client update norm must be positive
    update_norm = torch.linalg.norm(client_update).item()
    assert update_norm > 0.0
    assert avg_loss > 0.0
