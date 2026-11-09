"""
Controlled Label Noise Injection for Federated Learning Benchmark.
Applies exact symmetric label noise exclusively to Client 0 while preserving ground truth.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def inject_symmetric_noise(
    labels: np.ndarray,
    noise_rate: float = 0.40,
    num_classes: int = 9,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Inject symmetric label noise into a label array.

    Args:
        labels: 1D numpy array of integer class labels.
        noise_rate: Fraction of labels to corrupt (default: 0.40).
        num_classes: Total number of classes (default: 9 for PathMNIST).
        seed: Random seed for reproducible corruption selection.

    Returns:
        Tuple of (noisy_labels, clean_labels, is_corrupted_mask, audit_report)
    """
    clean_labels = np.asarray(labels, dtype=np.int64).copy()
    noisy_labels = clean_labels.copy()
    n_samples = len(clean_labels)
    is_corrupted = np.zeros(n_samples, dtype=bool)

    if noise_rate <= 0.0 or n_samples == 0:
        audit_report = {
            "total_samples": n_samples,
            "corrupted_samples": 0,
            "actual_noise_rate": 0.0,
            "per_class_corruption": {c: 0 for c in range(num_classes)},
        }
        return noisy_labels, clean_labels, is_corrupted, audit_report

    # Discrete rounding convention: round(noise_rate * n_samples)
    num_corrupted = int(round(noise_rate * n_samples))
    rng = np.random.RandomState(seed)

    # Randomly select exact indices to corrupt
    corrupted_indices = rng.choice(n_samples, size=num_corrupted, replace=False)
    is_corrupted[corrupted_indices] = True

    per_class_corrupt_count = {c: 0 for c in range(num_classes)}

    for idx in corrupted_indices:
        original_class = int(clean_labels[idx])
        per_class_corrupt_count[original_class] += 1
        
        # Sample uniformly from {0..num_classes-1} \ {original_class}
        candidates = [c for c in range(num_classes) if c != original_class]
        replacement = int(rng.choice(candidates))
        noisy_labels[idx] = replacement

    # Strict Mathematical Verifications
    assert np.all(noisy_labels[is_corrupted] != clean_labels[is_corrupted]), (
        "Violation: detected corrupted sample where y' == y"
    )
    assert np.all(noisy_labels[~is_corrupted] == clean_labels[~is_corrupted]), (
        "Violation: uncorrupted label was inadvertently modified"
    )

    actual_rate = num_corrupted / n_samples
    assert abs(actual_rate - noise_rate) <= 0.5 / n_samples, (
        f"Actual noise rate {actual_rate} deviates from discrete bounds around {noise_rate}"
    )

    audit_report = {
        "total_samples": n_samples,
        "corrupted_samples": num_corrupted,
        "actual_noise_rate": round(actual_rate, 6),
        "actual_noise_pct": round(actual_rate * 100, 4),
        "per_class_corruption": per_class_corrupt_count,
    }

    return noisy_labels, clean_labels, is_corrupted, audit_report
