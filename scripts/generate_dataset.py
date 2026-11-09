"""
Dataset Preparation & Partition Verification Script.
Checks PathMNIST (MD5 verified) and ensures Dirichlet partitions are generated.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import ensure_pathmnist_downloaded
from src.data.pathmnist_dataset import load_pathmnist_raw
from src.data.partition import dirichlet_partition, compute_partition_stats

def main():
    print("=" * 60)
    print("PathMNIST Dataset & Dirichlet Partition Setup")
    print("=" * 60)
    
    # 1. Locate / Download PathMNIST
    print("\n[1] Verifying PathMNIST dataset...")
    # Check potential cache locations
    candidate_dirs = [
        PROJECT_ROOT.parent / "dataset_cache",
        PROJECT_ROOT / "data" / "cache",
        Path("dataset_cache"),
        Path("data/cache")
    ]
    cache_dir = candidate_dirs[0]
    for cd in candidate_dirs:
        if (cd / "pathmnist.npz").exists():
            cache_dir = cd
            break
            
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path, md5 = ensure_pathmnist_downloaded(cache_dir)
    print(f"    PathMNIST NPZ: {npz_path} (MD5: {md5})")
    
    # 2. Check canonical partition files
    print("\n[2] Checking canonical partition files in src/data/partitions/...")
    partitions_dir = PROJECT_ROOT / "src" / "data" / "partitions"
    for seed in [42, 101, 2024]:
        p_path = partitions_dir / f"partition_dir0.3_clients5_seed{seed}.npz"
        if p_path.exists():
            print(f"    Seed {seed}: Found canonical {p_path.name}")
        else:
            print(f"    Seed {seed}: Generating partition...")
            train_images, train_labels, _, _ = load_pathmnist_raw(npz_path)
            splits = dirichlet_partition(train_labels, num_clients=5, alpha=0.3, seed=seed)
            import numpy as np
            np.savez(p_path, **{f"client_{i}": splits[i] for i in range(5)})
            print(f"    Seed {seed}: Saved {p_path.name}")
            
    print("\nDataset preparation verified successfully.")

if __name__ == "__main__":
    main()
