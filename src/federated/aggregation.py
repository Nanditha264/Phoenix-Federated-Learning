"""
Federated Model Aggregation Engine.
Implements mathematically rigorous sample-count-weighted parameter aggregation (FedAvg).
"""

from typing import List, Dict
import copy
import torch
import torch.nn as nn


def aggregate_models(
    models: List[nn.Module],
    sample_counts: List[int],
) -> nn.Module:
    """
    Compute sample-weighted parameter average across client models:
      w_global = sum_{k=0}^{K-1} (n_k / N) * w_k

    Args:
        models: List of client models trained in current round.
        sample_counts: List of local training sample counts [n_0, ..., n_{K-1}].

    Returns:
        New nn.Module instance holding the aggregated global weights.
    """
    assert len(models) > 0, "Cannot aggregate an empty list of models"
    assert len(models) == len(sample_counts), (
        f"Length mismatch: {len(models)} models vs {len(sample_counts)} sample counts"
    )

    total_samples = sum(sample_counts)
    assert total_samples > 0, f"Total sample count must be positive, got {total_samples}"

    weights = [n / total_samples for n in sample_counts]
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"

    # Clone base architecture from first model
    global_model = copy.deepcopy(models[0])
    global_state = global_model.state_dict()

    # Initialize float tensors to zeros for accumulation
    for key in global_state.keys():
        if global_state[key].is_floating_point():
            global_state[key] = torch.zeros_like(global_state[key])

    # Accumulate weighted parameters
    for client_model, weight in zip(models, weights):
        client_state = client_model.state_dict()
        for key in global_state.keys():
            if global_state[key].is_floating_point():
                global_state[key] += weight * client_state[key].to(global_state[key].device)
            else:
                # Non-floating buffers (e.g., num_batches_tracked) take value from client 0
                global_state[key] = client_state[key]

    global_model.load_state_dict(global_state)
    return global_model
