"""
PathMNIST Dataset Implementation.
Memory-efficient PyTorch Dataset utilizing zero-copy indices and lazy uint8-to-float32 tensor conversion.
"""

from pathlib import Path
from typing import Optional, Callable, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.preprocessing import get_default_transform


class PathMNISTDataset(Dataset):
    """
    In-memory PyTorch Dataset for PathMNIST.
    Stores images in compact uint8 representation (28x28x3) to conserve system memory,
    performing float32 conversion and normalization lazily inside __getitem__.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: Optional[Sequence[int]] = None,
        transform: Optional[Callable] = None,
        custom_labels: Optional[np.ndarray] = None,
    ):
        """
        Args:
            images: Contiguous numpy array of shape (N, 28, 28, 3) and dtype uint8.
            labels: 1D numpy array of class labels with shape (N,).
            indices: Optional integer indices pointing to a client partition subset.
            transform: Transformation callable applied to (28, 28, 3) uint8 image.
            custom_labels: Optional alternative label array (e.g. noisy labels for Client 0).
        """
        assert images.ndim == 4 and images.shape[1:] == (28, 28, 3), (
            f"Expected images shape (N, 28, 28, 3), got {images.shape}"
        )
        assert images.dtype == np.uint8, f"Expected uint8 dtype, got {images.dtype}"

        self.images = images
        self.labels = custom_labels if custom_labels is not None else labels
        self.original_labels = labels
        self.indices = np.asarray(indices, dtype=np.int64) if indices is not None else np.arange(len(self.labels))
        self.transform = transform if transform is not None else get_default_transform()

        # Check whether labels are local to the subset or global to the image array
        self.labels_are_subset = (len(self.labels) == len(self.indices))
        self.orig_labels_are_subset = (len(self.original_labels) == len(self.indices))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img_arr = self.images[real_idx]

        if self.labels_are_subset:
            label = int(self.labels[idx])
        else:
            label = int(self.labels[real_idx])

        img_tensor = self.transform(img_arr)
        return img_tensor, label

    def get_labels(self) -> np.ndarray:
        """Return the active labels for this subset."""
        if self.labels_are_subset:
            return self.labels
        return self.labels[self.indices]

    def get_original_labels(self) -> np.ndarray:
        """Return the pristine ground-truth labels for this subset."""
        if self.orig_labels_are_subset:
            return self.original_labels
        return self.original_labels[self.indices]
        """Return the pristine ground-truth labels for this subset."""
        return self.original_labels[self.indices]


def load_pathmnist_raw(npz_path: Path):
    """
    Load raw uncompressed arrays from PathMNIST NPZ archive into RAM.
    Note: NPZ files are compressed ZIP archives and are uncompressed into
    in-memory contiguous numpy uint8 arrays (240.5 MB total RAM).
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"PathMNIST NPZ file not found at: {npz_path}")

    with np.load(npz_path) as data:
        train_images = data["train_images"]  # (89996, 28, 28, 3) uint8
        train_labels = data["train_labels"].flatten().astype(np.int64)  # (89996,)
        val_images = data["val_images"]      # (10004, 28, 28, 3) uint8
        val_labels = data["val_labels"].flatten().astype(np.int64)      # (10004,)
        test_images = data["test_images"]    # (7180, 28, 28, 3) uint8
        test_labels = data["test_labels"].flatten().astype(np.int64)    # (7180,)

    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "val_images": val_images,
        "val_labels": val_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }
