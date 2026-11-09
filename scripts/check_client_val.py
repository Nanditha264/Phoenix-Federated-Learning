import json
from pathlib import Path

BENCHMARK_DIR = Path(r"D:\Reasearch\fl_benchmark\results\primary_benchmark")
RUNS = [
    ("fedavg", 42, "fedavg_dir0.3_noise0.4_seed42"),
    ("fedprox", 42, "fedprox_mu0.01_dir0.3_noise0.4_seed42"),
    ("fedavg", 101, "fedavg_dir0.3_noise0.4_seed101"),
    ("fedprox", 101, "fedprox_mu0.01_dir0.3_noise0.4_seed101"),
    ("fedavg", 2024, "fedavg_dir0.3_noise0.4_seed2024"),
    ("fedprox", 2024, "fedprox_mu0.01_dir0.3_noise0.4_seed2024"),
]

for algo, seed, r_name in RUNS:
    jl = BENCHMARK_DIR / r_name / "metrics.jsonl"
    with open(jl, "r") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    r5 = [r for r in recs if r.get("step") == 5][0]
    # check keys
    mean_val_acc = r5.get("eval/client_mean_val_acc", None)
    val_acc = r5.get("eval/val_acc", None)
    align = r5.get("geometry/alignment_with_global", None)
    norms = r5.get("geometry/client_update_norms", None)
    print(f"{r_name}: ValAcc={val_acc*100:.2f}%, ClientMeanValAcc={mean_val_acc*100:.2f}%, C0_Norm={norms[0]:.4f}, C0_Align={align[0]:.4f}")
