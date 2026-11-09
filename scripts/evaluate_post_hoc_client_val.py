"""
Post-Hoc Client Validation Evaluator.
Computes post-hoc per-client validation accuracy, loss, and macro-F1
for each of the 6 primary benchmark runs using their retained best_checkpoint.pt
and exact canonical client validation partitions.
"""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import ensure_pathmnist_downloaded
from src.data.pathmnist_dataset import PathMNISTDataset
from src.data.validation import partition_validation_set
from src.models.cnn import FedAvgCNN
from src.evaluation.client_evaluator import ClientValidationEvaluator

def run_evaluation():
    device = torch.device("cpu")
    torch.set_num_threads(4)

    cache_dir = "D:/Reasearch/dataset_cache"
    npz_path, md5 = ensure_pathmnist_downloaded(cache_dir)
    raw_data = np.load(npz_path)
    val_images = raw_data["val_images"]
    val_labels = raw_data["val_labels"].squeeze()

    runs = [
        ("fedavg", 42, "fedavg_dir0.3_noise0.4_seed42"),
        ("fedprox", 42, "fedprox_mu0.01_dir0.3_noise0.4_seed42"),
        ("fedavg", 101, "fedavg_dir0.3_noise0.4_seed101"),
        ("fedprox", 101, "fedprox_mu0.01_dir0.3_noise0.4_seed101"),
        ("fedavg", 2024, "fedavg_dir0.3_noise0.4_seed2024"),
        ("fedprox", 2024, "fedprox_mu0.01_dir0.3_noise0.4_seed2024"),
    ]

    results_all = {}
    primary_dir = PROJECT_ROOT / "results" / "primary_benchmark"

    for algo, seed, r_name in runs:
        val_client_indices = partition_validation_set(val_labels, num_clients=5, seed=seed)
        client_val_loaders = [
            DataLoader(PathMNISTDataset(images=val_images, labels=val_labels, indices=idxs), batch_size=64, shuffle=False)
            for idxs in val_client_indices
        ]
        client_evaluator = ClientValidationEvaluator(val_loaders=client_val_loaders, device=device)

        ckpt_path = primary_dir / r_name / "checkpoints" / "best_checkpoint.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict"))
        model = FedAvgCNN(num_classes=9)
        model.load_state_dict(state_dict)
        model.eval()

        res = client_evaluator.evaluate_clients(model)
        
        # Format metrics cleanly
        run_res = {
            "evaluation_type": "post_hoc_best_checkpoint_client_validation",
            "model_checkpoint": "best_checkpoint.pt",
            "seed": seed,
            "algorithm": algo,
            "mean_val_accuracy": res["mean_val_accuracy"],
            "mean_val_loss": res["mean_val_loss"],
            "weighted_val_accuracy": res["weighted_val_accuracy"],
            "weighted_val_loss": res["weighted_val_loss"],
            "per_client": {}
        }
        for cid in range(5):
            c_res = res["per_client"][cid]
            run_res["per_client"][str(cid)] = {
                "client_id": cid,
                "is_noisy": (cid == 0),
                "accuracy": round(c_res["accuracy"], 6),
                "loss": round(c_res["loss"], 6),
                "macro_f1": round(c_res["macro_f1"], 6),
                "total_samples": c_res["total_samples"],
            }

        # Save to run directory
        out_file = primary_dir / r_name / "post_hoc_client_val.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(run_res, f, indent=2)

        results_all[r_name] = run_res

        print(f"[{r_name}] Mean Val Acc = {res['mean_val_accuracy']*100:.2f}%, Weighted = {res['weighted_val_accuracy']*100:.2f}%")
        for cid in range(5):
            c = run_res["per_client"][str(cid)]
            tag = "(40% Noise)" if cid == 0 else "(Clean)"
            print(f"  Client {cid} {tag:11s}: Acc = {c['accuracy']*100:.2f}%, Loss = {c['loss']:.4f}, Macro-F1 = {c['macro_f1']:.4f}")

    # Summary table
    print("\n" + "=" * 80)
    print("POST-HOC PER-CLIENT VALIDATION SUMMARY (BEST CHECKPOINTS)")
    print("=" * 80)
    return results_all

if __name__ == "__main__":
    run_evaluation()
