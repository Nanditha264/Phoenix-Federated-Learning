"""
Unit Tests for Metrics and Update Geometry Engine.
"""

import torch
import numpy as np
import pytest

from src.metrics.accuracy import compute_accuracy, compute_classification_metrics
from src.metrics.update_geometry import (
    compute_vector_norm,
    compute_cosine_similarity,
    analyze_update_geometry,
)


def test_classification_metrics():
    """Verify top-1 accuracy, macro-F1, and per-class metrics."""
    targets = np.array([0, 0, 1, 1, 2, 2])
    preds   = np.array([0, 1, 1, 1, 2, 0])
    # Class 0: 1/2 correct = 0.5
    # Class 1: 2/2 correct = 1.0
    # Class 2: 1/2 correct = 0.5
    # Overall: 4/6 = 0.666667
    metrics = compute_classification_metrics(preds, targets, num_classes=3)

    assert abs(metrics["accuracy"] - 4 / 6) < 1e-5
    assert metrics["per_class_accuracy"][0] == 0.5
    assert metrics["per_class_accuracy"][1] == 1.0
    assert metrics["per_class_accuracy"][2] == 0.5
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_cosine_similarity_properties():
    """Verify standard geometric identities of cosine similarity."""
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v2 = torch.tensor([0.0, 1.0, 0.0])  # Orthogonal
    v3 = torch.tensor([-1.0, 0.0, 0.0]) # Opposite
    v_zero = torch.tensor([0.0, 0.0, 0.0])

    assert abs(compute_cosine_similarity(v1, v1) - 1.0) < 1e-6
    assert abs(compute_cosine_similarity(v1, v2)) < 1e-6
    assert abs(compute_cosine_similarity(v1, v3) - (-1.0)) < 1e-6
    # Zero vector handling
    assert compute_cosine_similarity(v1, v_zero) == 0.0


def test_update_geometry_analysis():
    """Verify update geometry outputs 5x5 symmetric matrix and valid norms."""
    num_clients = 5
    vec_len = 100
    updates = [torch.randn(vec_len) for _ in range(num_clients)]
    sample_counts = [1000, 2000, 1500, 3000, 2500]

    analysis = analyze_update_geometry(updates, sample_counts)

    assert len(analysis["update_l2_norms"]) == 5
    matrix = analysis["pairwise_cosine_matrix"]
    assert len(matrix) == 5
    assert len(matrix[0]) == 5

    for i in range(5):
        assert abs(matrix[i][i] - 1.0) < 1e-5
        for j in range(5):
            assert abs(matrix[i][j] - matrix[j][i]) < 1e-6

    assert analysis["global_update_norm"] > 0.0
    assert len(analysis["alignment_with_global"]) == 5
