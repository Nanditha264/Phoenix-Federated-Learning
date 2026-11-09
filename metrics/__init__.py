"""
Metrics Package for Federated Learning Benchmark.
"""

from fl_benchmark.metrics.accuracy import compute_accuracy, compute_classification_metrics
from fl_benchmark.metrics.loss import RunningAverage
from fl_benchmark.metrics.update_geometry import (
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
