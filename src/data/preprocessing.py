"""
Memory-Efficient Preprocessing Transforms for PathMNIST.
Converts in-memory uint8 HWC images to normalized float32 CHW tensors on-the-fly.
"""

from typing import Callable, Optional
import numpy as np
import torch
import torchvision.transforms as transforms


def get_default_transform(normalize_range_minus1_1: bool = True) -> Callable[[np.ndarray], torch.Tensor]:
    """
    Construct image transformation pipeline.
    
    Args:
        normalize_range_minus1_1: If True, normalize to [-1, 1] using mean=0.5, std=0.5.
                                  If False, scale to [0, 1].

    Returns:
        Callable transform operating on a single uint8 (28, 28, 3) numpy array.
    """
    transform_list = [
        transforms.ToTensor(),  # Converts (H, W, C) [0, 255] to (C, H, W) [0.0, 1.0]
    ]
    if normalize_range_minus1_1:
        transform_list.append(
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )

    composed = transforms.Compose(transform_list)

    def _transform(image_array: np.ndarray) -> torch.Tensor:
        # Expects uint8 array with shape (28, 28, 3)
        return composed(image_array)

    return _transform
