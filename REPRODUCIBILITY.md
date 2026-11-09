# Reproducibility Guide: FedAvg vs FedProx on PathMNIST

This document provides exhaustive, step-by-step instructions to reproduce every stage of the research benchmark: environment setup, dataset verification, canonical partition audits, model training, evaluation, figure rendering, and PDF report compilation.

---

## 1. System Requirements & Hardware Specification

- **Operating System:** Windows 11 (64-bit AMD64)
- **Processor:** Intel Core i5-1240P (12 cores, 16 threads) or equivalent modern x86_64 CPU
- **Physical Memory:** Minimum 8 GB RAM (Peak process working set: ~586 MB; minimum available RAM required: > 1,000 MB)
- **Storage:** Minimum 5 GB free disk space (Dataset: ~240 MB; checkpoints: ~2.3 MB each; Python environment: ~2.5 GB)
- **Python Version:** Python 3.11 (strictly verified on Python 3.11.0)
- **PyTorch Build:** CPU-only build (`torch==2.6.0+cpu` or `torch==2.15.0+cpu`)

---

## 2. Environment Setup & Dependency Installation

### 2.1 Activate Virtual Environment
```powershell
# Navigate to the benchmark workspace
cd D:\Reasearch\fl_benchmark

# Activate the verified research virtual environment
..\fl_env\Scripts\activate
```

### 2.2 Verify Environment Integrity
```powershell
python -V
# Expected: Python 3.11.0 (or 3.11.x)

pip check
# Expected: No broken requirements found.
```

### 2.3 Dependencies (`requirements.txt`)
If recreating the virtual environment from scratch:
```powershell
python -m venv D:\Reasearch\fl_env
D:\Reasearch\fl_env\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Dataset Verification & Canonical Partition Integrity

### 3.1 Verify Dataset MD5 Checksum
The raw PathMNIST dataset is cached at `D:\Reasearch\dataset_cache\pathmnist.npz`. Verify its official MD5 checksum:
```powershell
python -c "import hashlib; print('MD5:', hashlib.md5(open(r'D:\Reasearch\dataset_cache\pathmnist.npz','rb').read()).hexdigest())"
# Expected: a8b06965200029087d5bd730944a56c1
```

### 3.2 Verify Canonical Partition Checksums (`data/partitions/`)
All three canonical Dirichlet partitions ($\alpha = 0.3$) are serialized in `data/partitions/`. Verify their SHA-256 hashes:
```powershell
python -c "
import hashlib, glob
for f in sorted(glob.glob('data/partitions/*.npz')):
    h = hashlib.sha256(open(f, 'rb').read()).hexdigest().upper()
    print(f, '->', h)
"
```
**Expected Hashes:**
- `partition_dir0.3_clients5_seed42.npz` $\to$ `5D45B6280DD3EEE7B5B7CD3B73C2CDECFEE65959A963E100DC3FA701C23D8C76`
- `partition_dir0.3_clients5_seed101.npz` $\to$ `1C7E5F80F36CB6E6AA621B75DC038A287077B12B2894514127340A39E1537FA6`
- `partition_dir0.3_clients5_seed2024.npz` $\to$ `F6B32B01BA717C8437F133EAEB6DC1BD8929BFD4CD5C1F06083BB8AF6BEC26A6`

### 3.3 Verify Canonical Initial Model Weights (`data/initial_models/`)
Bit-level identical model weights are pre-registered for paired FedAvg/FedProx comparisons:
```powershell
python -c "
import hashlib, glob
for f in sorted(glob.glob('data/initial_models/*.pt')):
    h = hashlib.sha256(open(f, 'rb').read()).hexdigest().upper()
    print(f, '->', h)
"
```
**Expected Hashes:**
- `initial_model_seed42.pt` $\to$ `25ADB902DF84E61404A9B9F04F6AFD57E460D574C7B481B1EF1459E5E05158EB`
- `initial_model_seed101.pt` $\to$ `6A7E7BAF4BB71AFA19BEB49C0BDA1FCFA54C3D9E7C97ED06A237FA3F5501AAE7`
- `initial_model_seed2024.pt` $\to$ `EFD637BAC752E5C57B030B0E5F31FC04D09C48B3951457A85C6E31339B4C4787`

---

## 4. Run Automated Unit & Integration Tests

Before executing any benchmark run, verify system behavior with the complete 30-test suite:
```powershell
pytest tests/ -v
# Expected: 30 passed in ~27 seconds
```

---

## 5. Executing the Primary Benchmark Matrix

### 5.1 Master Matrix Command (Recommended)
The primary benchmark matrix runs all 6 experiments sequentially with the built-in Six-Factor Recovery Guard, preventing sleep/standby and isolating outputs in `results/primary_benchmark/`:
```powershell
python scripts/run_primary_matrix.py
```
*Note:* If any run has already completed (verified 5 rounds, loadable checkpoint, test evaluation), the recovery guard skips it and executes only remaining incomplete runs.

### 5.2 Individual Run Commands

#### Seed 42 Paired Comparison
```powershell
# Run 1: FedAvg, Seed 42
python run_benchmark.py --config configs/base.yaml --algorithm fedavg --seed 42 --rounds 5 --run_name fedavg_dir0.3_noise0.4_seed42 --output_dir "results/primary_benchmark" --eval_test

# Run 2: FedProx (mu=0.01), Seed 42
python run_benchmark.py --config configs/base.yaml --algorithm fedprox --seed 42 --rounds 5 --run_name fedprox_mu0.01_dir0.3_noise0.4_seed42 --output_dir "results/primary_benchmark" --eval_test --mu 0.01
```

#### Seed 101 Paired Comparison
```powershell
# Run 3: FedAvg, Seed 101
python run_benchmark.py --config configs/base.yaml --algorithm fedavg --seed 101 --rounds 5 --run_name fedavg_dir0.3_noise0.4_seed101 --output_dir "results/primary_benchmark" --eval_test

# Run 4: FedProx (mu=0.01), Seed 101
python run_benchmark.py --config configs/base.yaml --algorithm fedprox --seed 101 --rounds 5 --run_name fedprox_mu0.01_dir0.3_noise0.4_seed101 --output_dir "results/primary_benchmark" --eval_test --mu 0.01
```

#### Seed 2024 Paired Comparison
```powershell
# Run 5: FedAvg, Seed 2024
python run_benchmark.py --config configs/base.yaml --algorithm fedavg --seed 2024 --rounds 5 --run_name fedavg_dir0.3_noise0.4_seed2024 --output_dir "results/primary_benchmark" --eval_test

# Run 6: FedProx (mu=0.01), Seed 2024
python run_benchmark.py --config configs/base.yaml --algorithm fedprox --seed 2024 --rounds 5 --run_name fedprox_mu0.01_dir0.3_noise0.4_seed2024 --output_dir "results/primary_benchmark" --eval_test --mu 0.01
```

---

## 6. Standalone Test Split Evaluation

To evaluate an existing saved model checkpoint independently on the official 7,180 test split:
```powershell
python -c "
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from fl_benchmark.data.pathmnist_dataset import load_pathmnist_raw, PathMNISTDataset
from fl_benchmark.models.model_factory import create_model
from fl_benchmark.evaluation.evaluator import GlobalEvaluator

device = torch.device('cpu')
raw = load_pathmnist_raw('D:/Reasearch/dataset_cache/pathmnist.npz')
loader = DataLoader(PathMNISTDataset(raw['test_images'], raw['test_labels']), batch_size=64, shuffle=False)

model = create_model('FedAvgCNN', 3, 9, 1024).to(device)
ckpt_path = Path('results/primary_benchmark/fedavg_dir0.3_noise0.4_seed42/checkpoints/best_checkpoint.pt')
payload = torch.load(ckpt_path, map_location=device)
model.load_state_dict(payload['state_dict'])

evaluator = GlobalEvaluator(device=device)
metrics = evaluator.evaluate(model, loader)
print(f'Test Acc: {metrics[\"accuracy\"]*100:.2f}% | Test Loss: {metrics[\"loss\"]:.4f} | Macro-F1: {metrics[\"macro_f1\"]:.4f}')
"
```

---

## 7. Generating Publication Figures & Research Summary PDF

### 7.1 Generate All 6 Publication Figures (300 DPI PNG & Vector PDF)
```powershell
python scripts/generate_publication_figures.py
```
**Outputs generated in `figures/`:**
- `fig1_val_accuracy_vs_rounds.png` / `.pdf`
- `fig2_val_loss_vs_rounds.png` / `.pdf`
- `fig3_update_norm_shrinkage.png` / `.pdf`
- `fig4_update_cosine_alignment.png` / `.pdf`
- `fig5_pairwise_cosine_heatmaps.png` / `.pdf`
- `fig6_per_class_test_accuracy.png` / `.pdf`
- `figure_source_data_mapping.json`

### 7.2 Compile Multi-Page Research Summary PDF
```powershell
python scripts/generate_research_summary_pdf.py
```
**Output generated:**
- `D:\Reasearch\fl_benchmark\RESEARCH_SUMMARY.pdf` (6 publication-formatted pages).

---

## 8. Expected Output Artifacts

```
results/primary_benchmark/
├── matrix_execution_summary.json       # Master 6-run summary
├── comparative_analysis.json           # Aggregated statistics
├── detailed_table.json                 # Per-round and per-class data
└── <run_name>/
    ├── summary.json                    # Final run metrics & test evaluation
    ├── metrics.jsonl                   # Per-round telemetry (loss, acc, norms, cosine)
    └── checkpoints/
        ├── initial_checkpoint.pt       # Pre-training initialization weights
        ├── best_checkpoint.pt          # Best global model selected via val acc
        ├── last_checkpoint.pt          # Round 5 final global weights
        └── round_001.pt ... round_005.pt # Exact round checkpoints
```

---

## 9. Hardware & Operating System Considerations

1. **CPU Threading:** Clamped to 4 intra-op threads (`compute.num_threads: 4`) in `configs/base.yaml`. Do not increase beyond 4 on Alder Lake mobile processors to prevent efficiency-core (E-core) synchronization latency.
2. **Sequential Client Execution:** Enforces sequential client execution (`dataloader num_workers: 0`) with explicit garbage collection between clients, keeping RAM utilization strictly under 600 MB.
3. **Sleep/Standby Mitigation:** `scripts/run_primary_matrix.py` automatically sets Win32 `SetThreadExecutionState(0x80000001)`. If executing single runs manually overnight, run:
   ```cmd
   powercfg /change standby-timeout-ac 0
   ```
4. **Pristine Upstream Verification:** Ensure `D:\Reasearch\PFLlib-upstream` remains pristine:
   ```powershell
   cd D:\Reasearch\PFLlib-upstream; git status
   # Expected: working tree clean, on branch master at 0169ba7
   ```
