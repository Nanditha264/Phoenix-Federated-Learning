"""
Data Pipeline Audit and Validation Utilities.
Verifies complete dataset integrity, partition conservation, and noise isolation.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from fl_benchmark.data.download import ensure_pathmnist_downloaded
from fl_benchmark.data.pathmnist_dataset import load_pathmnist_raw
from fl_benchmark.data.partition import dirichlet_partition, compute_partition_stats
from fl_benchmark.data.noise import inject_symmetric_noise


def partition_validation_set(
    val_labels: np.ndarray,
    num_clients: int = 5,
    seed: int = 42
) -> List[np.ndarray]:
    """
    Partition official validation split into disjoint evaluation subsets for each client.
    Guarantees exact sample conservation (10,004 total) and zero label contamination.
    """
    n_val = len(val_labels)
    rng = np.random.RandomState(seed + 999)
    shuffled_indices = rng.permutation(n_val)
    splits = np.array_split(shuffled_indices, num_clients)
    
    # Verify conservation
    total_assigned = sum(len(s) for s in splits)
    assert total_assigned == n_val, f"Validation split conservation failed: {total_assigned} != {n_val}"
    all_assigned = np.concatenate(splits)
    assert len(np.unique(all_assigned)) == n_val, "Validation subsets have duplicate indices"
    
    return [np.array(sorted(s), dtype=np.int64) for s in splits]


def audit_data_pipeline(
    cache_dir: str = "D:/Reasearch/dataset_cache",
    num_clients: int = 5,
    alpha: float = 0.3,
    noise_rate: float = 0.40,
    noisy_client_id: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Perform a complete integrity check on PathMNIST data loading, Dirichlet partitioning,
    and symmetric noise injection.
    """
    npz_path, md5_hash = ensure_pathmnist_downloaded(cache_dir)
    raw_data = load_pathmnist_raw(npz_path)

    train_labels = raw_data["train_labels"]
    val_labels = raw_data["val_labels"]
    test_labels = raw_data["test_labels"]

    # Verify official MedMNIST counts
    assert len(train_labels) == 89996, f"Expected 89,996 train samples, got {len(train_labels)}"
    assert len(val_labels) == 10004, f"Expected 10,004 val samples, got {len(val_labels)}"
    assert len(test_labels) == 7180, f"Expected 7,180 test samples, got {len(test_labels)}"

    # Partition training set
    client_indices = dirichlet_partition(
        labels=train_labels,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )
    partition_stats = compute_partition_stats(train_labels, client_indices)

    # Inject noise into noisy_client_id
    noise_audits = {}
    client_clean_labels = []
    client_training_labels = []
    corruption_masks = []

    for cid in range(num_clients):
        c_idxs = client_indices[cid]
        c_labels = train_labels[c_idxs]
        client_clean_labels.append(c_labels)

        if cid == noisy_client_id:
            c_noisy, c_clean, mask, audit = inject_symmetric_noise(
                c_labels, noise_rate=noise_rate, num_classes=9, seed=seed + cid
            )
            client_training_labels.append(c_noisy)
            corruption_masks.append(mask)
            noise_audits[cid] = audit
        else:
            client_training_labels.append(c_labels)
            corruption_masks.append(np.zeros(len(c_labels), dtype=bool))
            noise_audits[cid] = {"corrupted_samples": 0, "actual_noise_rate": 0.0}

    # Partition validation set
    val_subsets = partition_validation_set(val_labels, num_clients=num_clients, seed=seed)

    return {
        "md5_checksum": md5_hash,
        "train_samples": len(train_labels),
        "val_samples": len(val_labels),
        "test_samples": len(test_labels),
        "client_sample_counts": partition_stats["client_sample_counts"],
        "class_distribution_matrix": partition_stats["class_distribution_matrix"],
        "noise_audits": noise_audits,
        "val_subset_counts": [len(s) for s in val_subsets],
    }
