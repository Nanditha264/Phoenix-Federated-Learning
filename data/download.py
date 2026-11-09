"""
Dataset Download and Cryptographic Verification for PathMNIST.
Checks local cache and validates MD5 checksum against official MedMNIST metadata.
"""

import os
import hashlib
from pathlib import Path
from typing import Tuple

EXPECTED_MD5 = "a8b06965200029087d5bd730944a56c1"
FILENAME = "pathmnist.npz"


def compute_file_md5(file_path: Path) -> str:
    """
    Compute MD5 hash of a local file using 64KB chunking.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def ensure_pathmnist_downloaded(cache_dir: str = "D:/Reasearch/dataset_cache") -> Tuple[Path, str]:
    """
    Ensure PathMNIST archive is present and cryptographically verified.
    Downloads only if missing or if checksum fails.

    Returns:
        Tuple of (path_to_npz, computed_md5)
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    target_file = cache_path / FILENAME

    if target_file.exists():
        actual_md5 = compute_file_md5(target_file)
        if actual_md5 == EXPECTED_MD5:
            return target_file, actual_md5
        else:
            print(f"[WARN] Checksum mismatch for existing file ({actual_md5} != {EXPECTED_MD5}). Re-downloading...")
            target_file.unlink()

    # If missing or corrupted, download using medmnist package
    import medmnist
    from medmnist import PathMNIST
    print(f"Downloading PathMNIST to {cache_path} via official MedMNIST API...")
    dataset = PathMNIST(split="train", root=str(cache_path), download=True)
    
    assert target_file.exists(), f"Download finished but {target_file} not found"
    actual_md5 = compute_file_md5(target_file)
    if actual_md5 != EXPECTED_MD5:
        raise ValueError(
            f"Cryptographic verification failed: expected {EXPECTED_MD5}, got {actual_md5}"
        )

    return target_file, actual_md5
