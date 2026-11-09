"""
Register canonical initial model weights for Seeds 101 and 2024.
Ensures 100% paired bit-level initial state across FedAvg and FedProx.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
for p in [str(PROJECT_ROOT), str(WORKSPACE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import hashlib
import torch
from src.utils.seed import set_seed
from src.models.model_factory import create_model

INIT_DIR = Path("D:/Reasearch/fl_benchmark/data/initial_models")
INIT_DIR.mkdir(parents=True, exist_ok=True)

for seed in [42, 101, 2024]:
    model_path = INIT_DIR / f"initial_model_seed{seed}.pt"
    if not model_path.exists():
        set_seed(seed=seed, deterministic_algorithms=False)
        model = create_model("FedAvgCNN", in_channels=3, num_classes=9, fc_hidden=1024)
        torch.save(model.state_dict(), model_path)
        print(f"Generated canonical initial model for Seed {seed}")
    else:
        print(f"Found existing canonical initial model for Seed {seed}")

    with open(model_path, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    print(f"  Seed {seed} file: {model_path.name} | SHA-256: {h}")
