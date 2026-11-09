"""
Extract detailed per-class and per-client validation metrics for Phase 5 Report.
"""

import json
from pathlib import Path

BENCHMARK_DIR = Path(r"D:\Reasearch\fl_benchmark\results\primary_benchmark")
SEEDS = [42, 101, 2024]
RUNS = [
    ("fedavg", 42, "fedavg_dir0.3_noise0.4_seed42"),
    ("fedprox", 42, "fedprox_mu0.01_dir0.3_noise0.4_seed42"),
    ("fedavg", 101, "fedavg_dir0.3_noise0.4_seed101"),
    ("fedprox", 101, "fedprox_mu0.01_dir0.3_noise0.4_seed101"),
    ("fedavg", 2024, "fedavg_dir0.3_noise0.4_seed2024"),
    ("fedprox", 2024, "fedprox_mu0.01_dir0.3_noise0.4_seed2024"),
]

detailed_table = []
for algo, seed, r_name in RUNS:
    s_path = BENCHMARK_DIR / r_name / "summary.json"
    jl_path = BENCHMARK_DIR / r_name / "metrics.jsonl"
    with open(s_path, "r") as f:
        s_data = json.load(f)
    with open(jl_path, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    r_metrics = [r for r in records if "round" in r]
    round_accs = [r["eval/val_acc"] * 100 for r in r_metrics]
    round_losses = [r["eval/val_loss"] for r in r_metrics]
    r5_time = r_metrics[-1]["time/round_seconds"]
    r5_ram = r_metrics[-1]["system/ram_available_mb"]
    r5_rss = r_metrics[-1]["system/process_rss_mb"]

    test_ev = s_data["test_evaluation"]

    detailed_table.append({
        "algo": algo,
        "seed": seed,
        "run_name": r_name,
        "round_val_accs": round_accs,
        "round_val_losses": round_losses,
        "best_val_acc": s_data["best_val_acc"] * 100,
        "test_acc": test_ev["accuracy"] * 100,
        "test_loss": test_ev["loss"],
        "test_macro_f1": test_ev["macro_f1"],
        "per_class": test_ev["per_class_accuracy"],
        "r5_time": r5_time,
        "r5_ram": r5_ram,
        "r5_rss": r5_rss,
    })

with open(BENCHMARK_DIR / "detailed_table.json", "w") as f:
    json.dump(detailed_table, f, indent=2)

print("Extracted detailed metrics for all 6 runs.")
