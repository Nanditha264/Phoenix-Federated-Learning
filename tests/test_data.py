"""
Unit Tests for PathMNIST Data Pipeline.
"""

from pathlib import Path
import numpy as np
import torch
import pytest

from src.data.preprocessing import get_default_transform
from src.data.pathmnist_dataset import PathMNISTDataset, load_pathmnist_raw
from src.data.download import compute_file_md5, ensure_pathmnist_downloaded


def test_preprocessing_transform():
    """Verify transform converts uint8 HWC to normalized float32 CHW tensor."""
    dummy_img = np.random.randint(0, 256, (28, 28, 3), dtype=np.uint8)
    transform = get_default_transform(normalize_range_minus1_1=True)
    out_tensor = transform(dummy_img)

    assert isinstance(out_tensor, torch.Tensor)
    assert out_tensor.shape == (3, 28, 28)
    assert out_tensor.dtype == torch.float32
    assert -1.0 <= out_tensor.min() <= 1.0
    assert -1.0 <= out_tensor.max() <= 1.0


def test_dataset_indexing_and_custom_labels():
    """Verify dataset indexing, len, and custom label overriding."""
    n_samples = 100
    images = np.random.randint(0, 256, (n_samples, 28, 28, 3), dtype=np.uint8)
    labels = np.random.randint(0, 9, (n_samples,), dtype=np.int64)

    # Subset indexing
    subset_indices = [5, 12, 19, 42]
    ds = PathMNISTDataset(images=images, labels=labels, indices=subset_indices)
    assert len(ds) == 4

    img, lbl = ds[0]
    assert img.shape == (3, 28, 28)
    assert lbl == labels[5]

    # Custom labels (e.g. noise)
    custom = labels.copy()
    custom[5] = 8
    ds_noisy = PathMNISTDataset(
        images=images, labels=labels, indices=subset_indices, custom_labels=custom
    )
    img_n, lbl_n = ds_noisy[0]
    assert lbl_n == 8
    assert ds_noisy.get_original_labels()[0] == labels[5]


def test_official_pathmnist_split_counts():
    """Verify official PathMNIST dataset split counts and shape."""
    cache_dir = Path("D:/Reasearch/dataset_cache")
    npz_path = cache_dir / "pathmnist.npz"
    if not npz_path.exists():
        pytest.skip("PathMNIST archive not found in cache")

    raw_data = load_pathmnist_raw(npz_path)
    assert raw_data["train_images"].shape == (89996, 28, 28, 3)
    assert raw_data["train_labels"].shape == (89996,)
    assert raw_data["val_images"].shape == (10004, 28, 28, 3)
    assert raw_data["val_labels"].shape == (10004,)
    assert raw_data["test_images"].shape == (7180, 28, 28, 3)
    assert raw_data["test_labels"].shape == (7180,)
    assert raw_data["train_images"].dtype == np.uint8
