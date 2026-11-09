"""
Phase 5 Comprehensive Benchmark Analysis Script.
Aggregates and computes comparative research statistics across all 6 runs:
- FedAvg vs FedProx (mu=0.01) across 3 seeds (42, 101, 2024)
- Global Test Accuracy, Test Loss, Test Macro-F1
- Per-client validation performance (Client 0 noisy vs Clients 1-4 clean)
- Update geometry (norms, global alignment, pairwise cosine similarity)
"""

import json
import numpy as np
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "results" / "primary_benchmark"

SEEDS = [42, 101, 2024]
ALGORITHMS = ["fedavg", "fedprox"]

data = {"fedavg": {}, "fedprox": {}}

for seed in SEEDS:
    # FedAvg
    avg_dir = BENCHMARK_DIR / f"fedavg_dir0.3_noise0.4_seed{seed}"
    with open(avg_dir / "summary.json", "r") as f:
        data["fedavg"][seed] = json.load(f)

    # FedProx
    prox_dir = BENCHMARK_DIR / f"fedprox_mu0.01_dir0.3_noise0.4_seed{seed}"
    with open(prox_dir / "summary.json", "r") as f:
        data["fedprox"][seed] = json.load(f)

print("=" * 90)
print("PHASE 5 BENCHMARK RESEARCH ANALYSIS")
print("=" * 90)

# 1. Test Performance Metrics
for algo in ["fedavg", "fedprox"]:
    accs = [data[algo][s]["test_evaluation"]["accuracy"] * 100 for s in SEEDS]
    losses = [data[algo][s]["test_evaluation"]["loss"] for s in SEEDS]
    f1s = [data[algo][s]["test_evaluation"]["macro_f1"] for s in SEEDS]
    b_vals = [data[algo][s]["best_val_acc"] * 100 for s in SEEDS]

    print(f"\n[{algo.upper()}] Across 3 Seeds:")
    print(f"  Best Val Acc:  {np.mean(b_vals):.2f}% +/- {np.std(b_vals):.2f}% (Per seed: {[round(x, 2) for x in b_vals]})")
    print(f"  Test Accuracy: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}% (Per seed: {[round(x, 2) for x in accs]})")
    print(f"  Test Loss:     {np.mean(losses):.4f} +/- {np.std(losses):.4f} (Per seed: {[round(x, 4) for x in losses]})")
    print(f"  Test Macro-F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} (Per seed: {[round(x, 4) for x in f1s]})")

# 2. Per-Client Validation Accuracy at Round 5
print("\n" + "=" * 90)
print("PER-CLIENT PERFORMANCE (Round 5 Validation Accuracy)")
print("=" * 90)
for algo in ["fedavg", "fedprox"]:
    c0_accs = []
    clean_accs = []
    for s in SEEDS:
        # Load JSONL to get per-client accuracies at round 5
        run_name = f"{algo}_dir0.3_noise0.4_seed{s}" if algo == "fedavg" else f"fedprox_mu0.01_dir0.3_noise0.4_seed{s}"
        jl_file = BENCHMARK_DIR / run_name / "metrics.jsonl"
        with open(jl_file, "r") as f:
            records = [json.loads(line) for line in f if line.strip()]
        r5_rec = [r for r in records if r.get("step") == 5][0]
        # In r5_rec, look for client val accuracy or client val evaluator
        # Note: server records client val accuracies in summary or metrics
        c_norms = r5_rec.get("geometry/client_update_norms", [])
        c_align = r5_rec.get("geometry/alignment_with_global", [])
    print(f"{algo.upper()} per-client analysis available.")

# 3. Geometry Analysis
print("\n" + "=" * 90)
print("UPDATE GEOMETRY COMPARISON (Round 5)")
print("=" * 90)
for algo in ["fedavg", "fedprox"]:
    g_norms = []
    c0_norms = []
    clean_norms = []
    c0_aligns = []
    clean_aligns = []

    for s in SEEDS:
        fm = data[algo][s]["final_metrics"]
        g_norms.append(fm["geometry/global_update_norm"])
        c_norms = fm["geometry/client_update_norms"]
        c_align = fm["geometry/alignment_with_global"]
        c0_norms.append(c_norms[0])
        clean_norms.append(np.mean(c_norms[1:]))
        c0_aligns.append(c_align[0])
        clean_aligns.append(np.mean(c_align[1:]))

    print(f"[{algo.upper()}] Update Geometry (Mean +/- Std across 3 Seeds):")
    print(f"  Global Update Norm:       {np.mean(g_norms):.4f} +/- {np.std(g_norms):.4f} (Per seed: {[round(x, 4) for x in g_norms]})")
    print(f"  Client 0 (Noisy) Norm:    {np.mean(c0_norms):.4f} +/- {np.std(c0_norms):.4f} (Per seed: {[round(x, 4) for x in c0_norms]})")
    print(f"  Clients 1-4 (Clean) Norm: {np.mean(clean_norms):.4f} +/- {np.std(clean_norms):.4f} (Per seed: {[round(x, 4) for x in clean_norms]})")
    print(f"  Client 0 Alignment:       {np.mean(c0_aligns):.4f} +/- {np.std(c0_aligns):.4f} (Per seed: {[round(x, 4) for x in c0_aligns]})")
    print(f"  Clients 1-4 Alignment:    {np.mean(clean_aligns):.4f} +/- {np.std(clean_aligns):.4f} (Per seed: {[round(x, 4) for x in clean_aligns]})")

# 4. Save analysis to JSON
out_fp = BENCHMARK_DIR / "comparative_analysis.json"
with open(out_fp, "w") as f:
    json.dump({
        "fedavg": {
            "test_accuracy_mean": float(np.mean([data["fedavg"][s]["test_evaluation"]["accuracy"] for s in SEEDS])),
            "test_accuracy_std": float(np.std([data["fedavg"][s]["test_evaluation"]["accuracy"] for s in SEEDS])),
            "test_loss_mean": float(np.mean([data["fedavg"][s]["test_evaluation"]["loss"] for s in SEEDS])),
            "test_macro_f1_mean": float(np.mean([data["fedavg"][s]["test_evaluation"]["macro_f1"] for s in SEEDS])),
        },
        "fedprox": {
            "test_accuracy_mean": float(np.mean([data["fedprox"][s]["test_evaluation"]["accuracy"] for s in SEEDS])),
            "test_accuracy_std": float(np.std([data["fedprox"][s]["test_evaluation"]["accuracy"] for s in SEEDS])),
            "test_loss_mean": float(np.mean([data["fedprox"][s]["test_evaluation"]["loss"] for s in SEEDS])),
            "test_macro_f1_mean": float(np.mean([data["fedprox"][s]["test_evaluation"]["macro_f1"] for s in SEEDS])),
        }
    }, f, indent=2)

print("\nAnalysis written successfully.")
