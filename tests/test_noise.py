"""
Unit Tests for Controlled Symmetric Label Noise.
"""

import numpy as np
import pytest
from fl_benchmark.data.noise import inject_symmetric_noise


def test_symmetric_noise_exact_count_and_rate():
    """Verify exact discrete count and rounding bounds."""
    n_samples = 1000
    labels = np.random.randint(0, 9, size=n_samples)
    noise_rate = 0.40

    noisy, clean, mask, audit = inject_symmetric_noise(
        labels, noise_rate=noise_rate, num_classes=9, seed=42
    )

    assert audit["total_samples"] == n_samples
    assert audit["corrupted_samples"] == 400
    assert audit["actual_noise_rate"] == 0.40
    assert np.sum(mask) == 400


def test_guaranteed_label_change():
    """Verify every corrupted sample receives y' != y from remaining 8 classes."""
    n_samples = 500
    labels = np.random.randint(0, 9, size=n_samples)

    noisy, clean, mask, _ = inject_symmetric_noise(
        labels, noise_rate=0.40, num_classes=9, seed=42
    )

    # Corrupted samples must change
    assert np.all(noisy[mask] != clean[mask])
    # Uncorrupted samples must remain identical
    assert np.all(noisy[~mask] == clean[~mask])
    # Range of replacement classes
    assert np.all(noisy >= 0) and np.all(noisy < 9)


def test_zero_noise_edge_case():
    """Verify noise_rate=0.0 leaves all labels untouched."""
    labels = np.random.randint(0, 9, size=200)
    noisy, clean, mask, audit = inject_symmetric_noise(
        labels, noise_rate=0.0, num_classes=9, seed=42
    )

    assert audit["corrupted_samples"] == 0
    assert audit["actual_noise_rate"] == 0.0
    assert np.sum(mask) == 0
    assert np.array_equal(noisy, clean)


def test_discrete_rounding_odd_count():
    """Verify rounding convention for odd sample sizes."""
    n_samples = 11
    labels = np.random.randint(0, 9, size=n_samples)
    # round(11 * 0.40) = round(4.4) = 4
    noisy, clean, mask, audit = inject_symmetric_noise(
        labels, noise_rate=0.40, num_classes=9, seed=42
    )
    assert audit["corrupted_samples"] == 4
