# Experimental Results Taxonomy

This directory contains the experimental artifacts, structured JSONL event telemetry, summaries, and verified evaluation checkpoints from the federated learning benchmark.

---

## 1. Directory Structure

```text
results/
└── primary_benchmark/                           # Official 6-Run Primary Benchmark Matrix
    ├── matrix_execution_summary.json            # Complete multi-run summary
    ├── comparative_analysis.json                # Aggregated statistics across seeds (mean +/- std)
    ├── detailed_table.json                      # Per-round and per-class performance data
    ├── fedavg_dir0.3_noise0.4_seed42/           # Run 1: FedAvg Seed 42 (5 Rounds)
    ├── fedprox_mu0.01_dir0.3_noise0.4_seed42/   # Run 2: FedProx Seed 42 (5 Rounds, mu=0.01)
    ├── fedavg_dir0.3_noise0.4_seed101/          # Run 3: FedAvg Seed 101 (5 Rounds)
    ├── fedprox_mu0.01_dir0.3_noise0.4_seed101/  # Run 4: FedProx Seed 101 (5 Rounds, mu=0.01)
    ├── fedavg_dir0.3_noise0.4_seed2024/         # Run 5: FedAvg Seed 2024 (5 Rounds)
    └── fedprox_mu0.01_dir0.3_noise0.4_seed2024/ # Run 6: FedProx Seed 2024 (5 Rounds, mu=0.01)
```

---

## 2. Artifact Contents per Run Directory

Each run directory contains:
- `summary.json`: Top-level execution metadata, best validation accuracy, duration, and final test evaluation (overall accuracy, loss, macro-F1, per-class breakdown across 7,180 test samples).
- `metrics.jsonl`: Round-by-round telemetry capturing training loss, validation accuracy/loss, client parameter displacement norms ($\|\Delta w_k\|$), global update norm ($\|\Delta w_{\text{global}}\|$), directional cosine alignment, and $5 \times 5$ pairwise cosine similarity matrices.
- `checkpoints/best_checkpoint.pt`: Optimal global model checkpoint selected via validation accuracy, verified to reproduce test metrics with 0.000000 divergence.
