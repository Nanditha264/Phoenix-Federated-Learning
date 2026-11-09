"""
Unit Tests for Dirichlet Non-IID Partitioning.
"""

from pathlib import Path
import numpy as np
import pytest

from src.data.partition import dirichlet_partition, compute_partition_stats
from src.data.pathmnist_dataset import load_pathmnist_raw


def test_synthetic_dirichlet_partition():
    """Verify Dirichlet partition properties on synthetic multi-class labels."""
    num_samples = 1000
    num_classes = 9
    num_clients = 5
    alpha = 0.3
    seed = 42

    labels = np.random.RandomState(seed).randint(0, num_classes, size=num_samples)
    client_indices = dirichlet_partition(
        labels=labels,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
        min_samples_per_client=10,
    )

    assert len(client_indices) == num_clients

    # Conservation
    total_assigned = sum(len(idx) for idx in client_indices)
    assert total_assigned == num_samples

    # No duplicates and no missing
    all_assigned = np.concatenate(client_indices)
    assert len(all_assigned) == num_samples
    assert len(np.unique(all_assigned)) == num_samples

    # Stats
    stats = compute_partition_stats(labels, client_indices, num_classes=num_classes)
    assert sum(stats["client_sample_counts"]) == num_samples


def test_dirichlet_reproducibility():
    """Verify identical allocations under the same random seed."""
    labels = np.random.randint(0, 9, size=500)
    part_a = dirichlet_partition(labels, num_clients=5, alpha=0.3, seed=123)
    part_b = dirichlet_partition(labels, num_clients=5, alpha=0.3, seed=123)

    for k in range(5):
        assert np.array_equal(part_a[k], part_b[k])


def test_dirichlet_seed_independence():
    """Verify distinct allocations under different seeds."""
    labels = np.random.randint(0, 9, size=500)
    part_a = dirichlet_partition(labels, num_clients=5, alpha=0.3, seed=101)
    part_b = dirichlet_partition(labels, num_clients=5, alpha=0.3, seed=202)

    counts_a = [len(p) for p in part_a]
    counts_b = [len(p) for p in part_b]
    assert counts_a != counts_b


def test_real_pathmnist_partition():
    """Verify Dirichlet partition on actual 89,996 PathMNIST training labels."""
    npz_path = Path("D:/Reasearch/dataset_cache/pathmnist.npz")
    if not npz_path.exists():
        pytest.skip("PathMNIST archive not found")

    raw_data = load_pathmnist_raw(npz_path)
    train_labels = raw_data["train_labels"]
    assert len(train_labels) == 89996

    client_indices = dirichlet_partition(
        labels=train_labels,
        num_clients=5,
        alpha=0.3,
        seed=42,
        min_samples_per_client=50,
    )

    assert len(client_indices) == 5
    counts = [len(idx) for idx in client_indices]
    assert sum(counts) == 89996
    all_assigned = np.concatenate(client_indices)
    assert len(np.unique(all_assigned)) == 89996
