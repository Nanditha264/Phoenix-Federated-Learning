"""
Update Geometry Engine for Federated Learning.
Computes client parameter displacement vectors (Delta_k = w_local - w_global),
their L2 norms, pairwise cosine similarity matrix, alignment with the global update,
and weighted global update norms.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn


def extract_flattened_params(model: nn.Module) -> torch.Tensor:
    """
    Extract all trainable parameters of a model into a single 1D flattened contiguous tensor.
    Guarantees consistent ordering across calls.
    """
    params = [p.detach().flatten() for p in model.parameters() if p.requires_grad]
    return torch.cat(params)


def compute_client_update(
    local_model: nn.Module,
    global_model: nn.Module
) -> torch.Tensor:
    """
    Compute parameter displacement vector: Delta_k = w_local_k - w_global.
    """
    w_local = extract_flattened_params(local_model)
    w_global = extract_flattened_params(global_model)
    return w_local - w_global


def compute_vector_norm(vec: torch.Tensor) -> float:
    """
    Compute Euclidean (L2) norm of a 1D tensor: ||vec||_2.
    """
    return float(torch.linalg.norm(vec).item())


def compute_cosine_similarity(
    vec_a: torch.Tensor,
    vec_b: torch.Tensor,
    eps: float = 1e-12
) -> float:
    """
    Compute cosine similarity between two 1D tensors with zero-norm guard.
    cos(a, b) = (a . b) / (||a||_2 * ||b||_2)
    """
    norm_a = torch.linalg.norm(vec_a).item()
    norm_b = torch.linalg.norm(vec_b).item()

    if norm_a < eps or norm_b < eps:
        return 0.0

    dot_product = torch.dot(vec_a, vec_b).item()
    similarity = dot_product / (norm_a * norm_b)
    # Clamp to [-1.0, 1.0] for numerical stability
    return float(np.clip(similarity, -1.0, 1.0))


def analyze_update_geometry(
    client_updates: List[torch.Tensor],
    client_sample_counts: List[int],
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Analyze geometric properties of client model updates for a communication round.

    Args:
        client_updates: List of 1D tensors [Delta_0, Delta_1, ..., Delta_{K-1}].
        client_sample_counts: Number of training samples per client [n_0, ..., n_{K-1}].
        eps: Small constant for numerical stability.

    Returns:
        Dictionary containing:
        - update_l2_norms: List of float L2 norms for each client update.
        - pairwise_cosine_matrix: K x K matrix (list of lists) of pairwise cosine similarities.
        - global_update_norm: Float L2 norm of the sample-weighted global update.
        - alignment_with_global: List of cosine similarities between each Delta_k and Delta_global.
    """
    num_clients = len(client_updates)
    assert num_clients == len(client_sample_counts), (
        f"Length mismatch: {num_clients} updates vs {len(client_sample_counts)} counts"
    )

    # 1. Compute individual L2 norms
    l2_norms = [compute_vector_norm(u) for u in client_updates]

    # 2. Compute K x K Pairwise Cosine Similarity Matrix
    cosine_matrix = np.zeros((num_clients, num_clients), dtype=np.float64)
    for i in range(num_clients):
        cosine_matrix[i, i] = 1.0
        for j in range(i + 1, num_clients):
            sim = compute_cosine_similarity(client_updates[i], client_updates[j], eps=eps)
            cosine_matrix[i, j] = sim
            cosine_matrix[j, i] = sim

    # 3. Compute Sample-Weighted Global Update: Delta_global = sum_k (n_k / N) * Delta_k
    total_samples = sum(client_sample_counts)
    assert total_samples > 0, "Total sample count must be positive"

    delta_global = torch.zeros_like(client_updates[0])
    for k in range(num_clients):
        weight = client_sample_counts[k] / total_samples
        delta_global += weight * client_updates[k]

    global_update_norm = compute_vector_norm(delta_global)

    # 4. Compute Alignment of each client update with the global update
    alignment_with_global = [
        compute_cosine_similarity(u, delta_global, eps=eps) for u in client_updates
    ]

    return {
        "update_l2_norms": [round(n, 6) for n in l2_norms],
        "pairwise_cosine_matrix": [[round(val, 6) for val in row] for row in cosine_matrix.tolist()],
        "global_update_norm": round(global_update_norm, 6),
        "alignment_with_global": [round(a, 6) for a in alignment_with_global],
    }
