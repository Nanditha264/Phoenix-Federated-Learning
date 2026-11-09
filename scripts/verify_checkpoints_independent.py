"""
Independent Re-evaluation of Best Checkpoints on the PathMNIST Test Split.
Verifies all 6 primary benchmark runs against actual checkpoint weights.
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import ensure_pathmnist_downloaded
from src.data.pathmnist_dataset import PathMNISTDataset
from src.models.cnn import FedAvgCNN
from src.evaluation.evaluator import GlobalEvaluator

def main():
    device = torch.device("cpu")
    torch.set_num_threads(4)
    
    # 1. Load clean test split
    print("[1] Loading raw PathMNIST test split...")
    cache_dir = "D:/Reasearch/dataset_cache"
    npz_path, md5_hash = ensure_pathmnist_downloaded(cache_dir)
    raw_data = np.load(npz_path)
    test_images = raw_data["test_images"]
    test_labels = raw_data["test_labels"].squeeze()
    
    test_dataset = PathMNISTDataset(images=test_images, labels=test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f"    Loaded {len(test_dataset)} test samples across 9 classes.")
    
    # Print class distribution in test split
    unique, counts = np.unique(test_labels, return_counts=True)
    print(f"    Test split class counts: {dict(zip(unique.tolist(), counts.tolist()))}")
    
    evaluator = GlobalEvaluator(device=device)
    
    primary_dir = PROJECT_ROOT / "results" / "primary_benchmark"
    runs = [
        ("fedavg", 42, "fedavg_dir0.3_noise0.4_seed42"),
        ("fedprox", 42, "fedprox_mu0.01_dir0.3_noise0.4_seed42"),
        ("fedavg", 101, "fedavg_dir0.3_noise0.4_seed101"),
        ("fedprox", 101, "fedprox_mu0.01_dir0.3_noise0.4_seed101"),
        ("fedavg", 2024, "fedavg_dir0.3_noise0.4_seed2024"),
        ("fedprox", 2024, "fedprox_mu0.01_dir0.3_noise0.4_seed2024"),
    ]
    
    print("\n[2] Re-evaluating saved best checkpoints...")
    all_verified = True
    
    for algo, seed, r_name in runs:
        ckpt_path = primary_dir / r_name / "checkpoints" / "best_checkpoint.pt"
        summary_path = primary_dir / r_name / "summary.json"
        
        with open(summary_path, "r") as f:
            saved_summary = json.load(f)
            
        saved_te = saved_summary["test_evaluation"]
        
        # Load model
        model = FedAvgCNN(num_classes=9)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict"))
        model.load_state_dict(state_dict)
        model.eval()
        
        # Evaluate
        recomputed = evaluator.evaluate(model, test_loader)
        
        # Compare
        acc_diff = abs(recomputed["accuracy"] - saved_te["accuracy"])
        loss_diff = abs(recomputed["loss"] - saved_te["loss"])
        f1_diff = abs(recomputed["macro_f1"] - saved_te["macro_f1"])
        
        # Compare Class 8
        c8_saved = saved_te["per_class_accuracy"]["8"]
        c8_recomp = recomputed["per_class_accuracy"][8]
        c8_diff = abs(c8_saved - c8_recomp)
        
        match = (acc_diff < 1e-4 and loss_diff < 1e-3 and f1_diff < 1e-4 and c8_diff < 1e-4)
        if not match:
            all_verified = False
            
        print(f"\n--- {r_name} ---")
        print(f"  Checkpoint Round: {ckpt.get('round', 'unknown')}")
        print(f"  Saved Accuracy:      {saved_te['accuracy']*100:.2f}% | Recomputed: {recomputed['accuracy']*100:.2f}% (Diff: {acc_diff:.6f})")
        print(f"  Saved Loss:          {saved_te['loss']:.4f}   | Recomputed: {recomputed['loss']:.4f} (Diff: {loss_diff:.6f})")
        print(f"  Saved Macro-F1:      {saved_te['macro_f1']:.4f}   | Recomputed: {recomputed['macro_f1']:.4f} (Diff: {f1_diff:.6f})")
        print(f"  Saved Class 8 Acc:   {c8_saved*100:.2f}% | Recomputed: {c8_recomp*100:.2f}% (Diff: {c8_diff:.6f})")
        print(f"  Status: {'MATCH (100% VERIFIED)' if match else 'MISMATCH'}")
        
    print("\n" + "=" * 80)
    if all_verified:
        print("ALL 6 BEST CHECKPOINTS RE-EVALUATED AND INDEPENDENTLY CONFIRMED!")
        print("Class 8 accuracy and overall test metrics are 100% authentic and reproducible from raw model weights.")
    else:
        print("WARNING: Some recomputed metrics diverged from saved summaries!")
    print("=" * 80)

if __name__ == "__main__":
    main()
