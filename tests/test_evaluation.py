"""
Unit Tests for Independent Evaluation Engine.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

from fl_benchmark.evaluation.evaluator import GlobalEvaluator
from fl_benchmark.evaluation.client_evaluator import ClientValidationEvaluator
from fl_benchmark.data.validation import partition_validation_set


def test_global_evaluator():
    """Verify GlobalEvaluator calculates accuracy and loss over DataLoader."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(12, 3)
    )
    x = torch.randn(30, 12)
    y = torch.randint(0, 3, (30,))
    loader = DataLoader(TensorDataset(x, y), batch_size=10)

    evaluator = GlobalEvaluator()
    res = evaluator.evaluate(model, loader, num_classes=3)

    assert res["total_samples"] == 30
    assert 0.0 <= res["accuracy"] <= 1.0
    assert res["loss"] > 0.0


def test_client_validation_evaluator():
    """Verify ClientValidationEvaluator across multiple disjoint client partitions."""
    model = nn.Linear(8, 2)
    val_loaders = [
        DataLoader(TensorDataset(torch.randn(20, 8), torch.randint(0, 2, (20,))), batch_size=10)
        for _ in range(5)
    ]

    c_eval = ClientValidationEvaluator(val_loaders=val_loaders)
    results = c_eval.evaluate_clients(model)

    assert len(results["per_client"]) == 5
    for cid in range(5):
        assert results["per_client"][cid]["total_samples"] == 20
    assert 0.0 <= results["mean_val_accuracy"] <= 1.0


def test_validation_split_disjointness():
    """Verify validation split is completely partitioned with zero overlap."""
    n_val = 10004
    labels = torch.randint(0, 9, (n_val,)).numpy()
    subsets = partition_validation_set(labels, num_clients=5, seed=42)

    assert len(subsets) == 5
    assert sum(len(s) for s in subsets) == n_val

    all_indices = set()
    for s in subsets:
        idx_set = set(s.tolist())
        assert len(idx_set.intersection(all_indices)) == 0, "Overlap found between validation subsets"
        all_indices.update(idx_set)

    assert len(all_indices) == n_val
