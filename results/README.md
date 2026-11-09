# Results Directory Provenance & Taxonomy

This directory contains experimental artifacts, checkpoints, structured telemetry logs, and diagnostic traces from the federated learning benchmark.

---

## 1. Directory Structure

```text
results/
├── primary_benchmark/                           # Official 6-Run Primary Benchmark Matrix
│   ├── matrix_execution_summary.json            # Complete multi-run summary
│   ├── comparative_analysis.json                # Aggregated statistics across seeds (mean +/- std)
│   ├── detailed_table.json                      # Per-round and per-class performance data
│   ├── diagnostic_attempt1_hang_logs/           # Preserved socket-hang diagnostic logs (Pre-fix)
│   ├── fedavg_dir0.3_noise0.4_seed42/           # Run 1: FedAvg Seed 42 (5 Rounds)
│   ├── fedprox_mu0.01_dir0.3_noise0.4_seed42/   # Run 2: FedProx Seed 42 (5 Rounds)
│   ├── fedavg_dir0.3_noise0.4_seed101/          # Run 3: FedAvg Seed 101 (5 Rounds)
│   ├── fedprox_mu0.01_dir0.3_noise0.4_seed101/  # Run 4: FedProx Seed 101 (5 Rounds)
│   ├── fedavg_dir0.3_noise0.4_seed2024/         # Run 5: FedAvg Seed 2024 (5 Rounds)
│   └── fedprox_mu0.01_dir0.3_noise0.4_seed2024/ # Run 6: FedProx Seed 2024 (5 Rounds)
│
├── fedavg_dir0.3_noise0.4_seed42/               # Preserved Phase 5 Full-Data 2-Round Pilot (FedAvg)
├── fedprox_dir0.3_noise0.4_seed42/              # Preserved Phase 5 Full-Data 2-Round Pilot (FedProx)
├── fedprox_mu0.01_dir0.3_noise0.4_seed42/       # Preserved Phase 4.1 1-Round Smoke Test (FedProx)
└── test_wandb/                                  # Preserved Diagnostic Traces from W&B Socket Analysis
```

---

## 2. Artifact Contents per Run Directory
Each run folder contains:
- `summary.json`: Top-level summary, best validation accuracy, rounds completed, and final test evaluation (accuracy, loss, macro-F1, per-class breakdown).
- `metrics.jsonl`: Step-by-step telemetry capturing training loss, validation accuracy/loss, client update displacement norms, global alignment, pairwise cosine similarity, and host RAM/CPU telemetry.
- `checkpoints/`:
  - `initial_checkpoint.pt`: Model weights prior to training (Round 0).
  - `best_checkpoint.pt`: Checkpoint yielding the highest global validation accuracy.
  - `last_checkpoint.pt`: Final model weights after Round 5.
  - `round_001.pt` through `round_005.pt`: Exact round-by-round serialized states.
