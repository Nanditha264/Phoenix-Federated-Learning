"""
Metrics Package for Federated Learning Benchmark.
"""

from src.metrics.accuracy import compute_accuracy, compute_classification_metrics
from src.metrics.loss import RunningAverage
from src.metrics.update_geometry import (
    extract_flattened_params,
    compute_client_update,
    compute_vector_norm,
    compute_cosine_similarity,
    analyze_update_geometry,
)

__all__ = [
    "compute_accuracy",
    "compute_classification_metrics",
    "RunningAverage",
    "extract_flattened_params",
    "compute_client_update",
    "compute_vector_norm",
    "compute_cosine_similarity",
    "analyze_update_geometry",
]
