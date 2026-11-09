import json
from pathlib import Path
import numpy as np

def verify_all_claims():
    p = Path(r"D:\Reasearch\fl_benchmark\results\primary_benchmark")
    seeds = [42, 101, 2024]
    algos = ["fedavg", "fedprox"]
    
    results = {}
    
    for algo in algos:
        results[algo] = {
            "test_acc": [],
            "test_loss": [],
            "test_f1": [],
            "class_mat": np.zeros((3, 9)),
            "global_update_norm": [],
            "client0_norm": [],
            "clean_clients_norm": [],
            "clean_cosine_alignment": [],
            "pairwise_cosine": []
        }
        
        for i, s in enumerate(seeds):
            folder = f"{algo}_dir0.3_noise0.4_seed{s}" if algo == "fedavg" else f"fedprox_mu0.01_dir0.3_noise0.4_seed{s}"
            summary_path = p / folder / "summary.json"
            metrics_path = p / folder / "metrics.jsonl"
            
            with open(summary_path, "r") as f:
                s_data = json.load(f)
            
            te = s_data["test_evaluation"]
            results[algo]["test_acc"].append(te["accuracy"] * 100)
            results[algo]["test_loss"].append(te["loss"])
            results[algo]["test_f1"].append(te["macro_f1"])
            
            pc = te["per_class_accuracy"]
            for c in range(9):
                results[algo]["class_mat"][i, c] = pc[str(c)] * 100
                
            with open(metrics_path, "r") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            
            rounds = [r for r in lines if "round" in r]
            if i == 0 and algo == "fedavg":
                print("SAMPLE ROUND KEYS:")
                print([k for k in rounds[-1].keys() if "geometry" in k or "norm" in k or "cosine" in k or "update" in k])

            
            # Geometry metrics across all rounds or last round
            # Let's inspect how norms were computed
            r_global_norms = [r.get("geometry/global_update_norm", 0) for r in rounds]
            r_c0_norms = [r.get("geometry/client_update_norm/client_0", 0) for r in rounds]
            r_clean_norms = [
                np.mean([r.get(f"geometry/client_update_norm/client_{c}", 0) for c in range(1, 5)])
                for r in rounds
            ]
            r_clean_cos = [
                np.mean([r.get(f"geometry/client_cosine_to_global/client_{c}", 0) for c in range(1, 5)])
                for r in rounds
            ]
            
            results[algo]["global_update_norm"].append(np.mean(r_global_norms))
            results[algo]["client0_norm"].append(np.mean(r_c0_norms))
            results[algo]["clean_clients_norm"].append(np.mean(r_clean_norms))
            results[algo]["clean_cosine_alignment"].append(np.mean(r_clean_cos))
            
    print("=" * 80)
    print("VERIFICATION OF SCIENTIFIC CLAIMS")
    print("=" * 80)
    
    for algo in algos:
        print(f"\n--- {algo.upper()} ---")
        acc = results[algo]["test_acc"]
        loss = results[algo]["test_loss"]
        f1 = results[algo]["test_f1"]
        print(f"Test Accuracy: {acc} -> Mean: {np.mean(acc):.2f}% +/- {np.std(acc):.2f}% (ddof=0) | sample std: {np.std(acc, ddof=1):.2f}%")
        print(f"Test Loss:     {loss} -> Mean: {np.mean(loss):.4f} +/- {np.std(loss):.4f}")
        print(f"Test Macro-F1: {f1} -> Mean: {np.mean(f1):.4f} +/- {np.std(f1):.4f}")
        print(f"Global Norm:   {results[algo]['global_update_norm']} -> Mean: {np.mean(results[algo]['global_update_norm']):.4f} +/- {np.std(results[algo]['global_update_norm']):.4f}")
        print(f"Client 0 Norm: {results[algo]['client0_norm']} -> Mean: {np.mean(results[algo]['client0_norm']):.4f} +/- {np.std(results[algo]['client0_norm']):.4f}")
        print(f"Clean Cosine:  {results[algo]['clean_cosine_alignment']} -> Mean: {np.mean(results[algo]['clean_cosine_alignment']):.4f} +/- {np.std(results[algo]['clean_cosine_alignment']):.4f}")
        
        print("\nPer-Class Accuracies (%):")
        for c in range(9):
            vals = results[algo]["class_mat"][:, c]
            print(f"  Class {c}: seeds={vals} -> Mean: {np.mean(vals):.2f}% +/- {np.std(vals):.2f}%")

    print("\n" + "=" * 80)
    print("SPECIFIC CLAIM CHECK: Class 8 (Colorectal Adenocarcinoma Epithelium)")
    avg_c8 = np.mean(results["fedavg"]["class_mat"][:, 8])
    prox_c8 = np.mean(results["fedprox"]["class_mat"][:, 8])
    delta_c8 = prox_c8 - avg_c8
    print(f"FedAvg Class 8 Mean:  {avg_c8:.2f}% (seeds: {results['fedavg']['class_mat'][:, 8]})")
    print(f"FedProx Class 8 Mean: {prox_c8:.2f}% (seeds: {results['fedprox']['class_mat'][:, 8]})")
    print(f"Difference (FedProx - FedAvg): {delta_c8:+.2f} percentage points")
    print("=" * 80)

if __name__ == "__main__":
    verify_all_claims()
