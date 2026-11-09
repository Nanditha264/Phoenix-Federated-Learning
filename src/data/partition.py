"""
Dirichlet Non-IID Partitioning for Federated Learning Clients.
Implements class-wise Dirichlet allocation with exact sample conservation.
"""

from typing import List, Dict, Any
import numpy as np


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int = 5,
    alpha: float = 0.3,
    seed: int = 42,
    min_samples_per_client: int = 50,
) -> List[np.ndarray]:
    """
    Partition training sample indices among federated clients using Dirichlet distribution.

    Args:
        labels: 1D numpy array of sample class labels (e.g. 89,996 for PathMNIST train).
        num_clients: Total number of clients (default: 5).
        alpha: Dirichlet concentration parameter (default: 0.3 for Non-IID).
        seed: Random seed for partition reproducibility (default: 42).
        min_samples_per_client: Minimum number of samples required per client.

    Returns:
        List of 1D numpy int64 arrays, each containing indices assigned to that client.
    """
    labels = np.asarray(labels).flatten()
    num_samples = len(labels)
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)

    max_attempts = 100
    for attempt in range(max_attempts):
        rng = np.random.RandomState(seed + attempt)
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]

        for c in unique_classes:
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)

            # Sample Dirichlet proportions for this class across clients
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            
            # Compute cumulative split points
            split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            splits = np.split(idx_c, split_points)

            for j in range(num_clients):
                client_indices[j].extend(splits[j].tolist())

        # Check minimum sample safeguard
        counts = [len(idx) for idx in client_indices]
        if all(cnt >= min_samples_per_client for cnt in counts):
            break
    else:
        raise RuntimeError(
            f"Failed to find valid partition meeting min_samples={min_samples_per_client} "
            f"after {max_attempts} attempts."
        )

    # Convert to sorted int64 numpy arrays
    partitioned_arrays = [np.array(sorted(idx), dtype=np.int64) for idx in client_indices]

    # Mathematical Verification
    total_allocated = sum(len(idx) for idx in partitioned_arrays)
    assert total_allocated == num_samples, (
        f"Sample conservation violated: allocated {total_allocated} != {num_samples}"
    )

    all_indices = np.concatenate(partitioned_arrays)
    assert len(np.unique(all_indices)) == num_samples, (
        "Duplicate or missing indices detected in Dirichlet partition"
    )

    return partitioned_arrays


def compute_partition_stats(
    labels: np.ndarray,
    client_indices: List[np.ndarray],
    num_classes: int = 9,
) -> Dict[str, Any]:
    """
    Calculate summary statistics and class distributions across client partitions.
    """
    num_clients = len(client_indices)
    client_counts = [len(idx) for idx in client_indices]
    total_samples = sum(client_counts)

    class_distribution = np.zeros((num_clients, num_classes), dtype=np.int64)
    for j, idxs in enumerate(client_indices):
        c_labels = labels[idxs]
        for c in range(num_classes):
            class_distribution[j, c] = np.sum(c_labels == c)

    counts_arr = np.array(client_counts, dtype=np.float64)[:, None]
    class_proportions = class_distribution.astype(np.float64) / np.maximum(counts_arr, 1.0)

    return {
        "num_clients": num_clients,
        "total_samples": total_samples,
        "client_sample_counts": client_counts,
        "class_distribution_matrix": class_distribution.tolist(),
        "class_proportions_matrix": class_proportions.tolist(),
    }
