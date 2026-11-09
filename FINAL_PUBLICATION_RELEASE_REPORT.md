# FINAL RESEARCH PUBLICATION & RELEASE REPORT
**Project:** Phoenix Federated Learning Research  
**Study:** FedAvg vs. FedProx ($\mu = 0.01$) on PathMNIST under Heterogeneous Dirichlet Skew ($\alpha = 0.3$) and Controlled Client-Specific Label Noise ($40\%$ on Client 0)  
**Date & Timestamp:** 2026-11-09T18:46:30+05:30  
**Project Workspace:** `D:\Reasearch\fl_benchmark`  
**Pristine Reference Repository:** `D:\Reasearch\PFLlib-upstream` (Commit `0169ba7`, 100% clean)  
**Execution Environment:** Python 3.11.0, PyTorch 2.15.0+cpu, Intel Core i5-1240P (4 intra-op threads, 1 inter-op thread)  

---

## 1. Scientific Reporting Audit & Critical Corrections

### 1.1 Critical Update-Norm Reporting Correction
Prior documentation iterations displayed the aggregate mean values ($2.4838$ and $1.7388$) across individual seed rows. This presentation has been corrected. The exact, independently audited per-seed values extracted from the underlying JSONL/JSON artifacts at Round 5 are:

| Metric (Round 5) | Seed 42 | Seed 101 | Seed 2024 | Aggregate Mean $\pm$ Pop. Std | Sample Std ($N-1$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **FedAvg Global Update Norm $\|\Delta w_{\text{global}}\|$** | **2.3855** | **2.8297** | **2.2361** | **$2.4838 \pm 0.2521$** | $\pm 0.3088$ |
| **FedProx Global Update Norm $\|\Delta w_{\text{global}}\|$** | **1.7438** | **1.8879** | **1.5846** | **$1.7388 \pm 0.1239$** | $\pm 0.1517$ |
| *Relative Global Norm Change (Shrinkage)* | *-26.90%* | *-33.28%* | *-29.14%* | **-29.99%** | — |
| **FedAvg Noisy Client 0 Norm $\|\Delta w_0\|$** | **3.2065** | **3.6252** | **4.0943** | **$3.6420 \pm 0.3626$** | $\pm 0.4441$ |
| **FedProx Noisy Client 0 Norm $\|\Delta w_0\|$** | **2.2593** | **2.3576** | **2.3517** | **$2.3229 \pm 0.0450$** | $\pm 0.0551$ |
| *Relative Client 0 Norm Dampening* | *-29.54%* | *-34.97%* | *-42.56%* | **-36.22%** | — |
| **FedAvg Clean Clients 1–4 Norm (Avg)** | **4.4678** | **4.2424** | **3.4636** | **$4.0579 \pm 0.4302$** | $\pm 0.5268$ |
| **FedProx Clean Clients 1–4 Norm (Avg)** | **2.9878** | **2.8273** | **2.4898** | **$2.7683 \pm 0.2075$** | $\pm 0.2541$ |
| *Relative Clean Client Norm Contraction* | *-33.13%* | *-33.36%* | *-28.11%* | **-31.78%** | — |
| **FedAvg Clean Clients Cosine Alignment** | **0.5837** | **0.4972** | **0.4717** | **$0.5175 \pm 0.0479$** | $\pm 0.0587$ |
| **FedProx Clean Clients Cosine Alignment** | **0.6434** | **0.5975** | **0.5520** | **$0.5976 \pm 0.0373$** | $\pm 0.0457$ |
| *Relative Clean Alignment Gain* | *+10.23%* | *+20.17%* | *+17.02%* | **+15.48%** | — |

*Key Empirical Takeaways:*
1. **Consistency across all seeds:** In 100% of the paired experimental seeds, FedProx strictly reduced the global update norm (by $26.9\%$ to $33.3\%$) and dampened the corrupted gradient updates from Client 0 (by $29.5\%$ to $42.6\%$).
2. **Clear labeling:** All individual seed values are preserved and reported separately from statistical aggregations.
3. **Data enrichment:** `scripts/extract_detailed_metrics.py` and `results/primary_benchmark/detailed_table.json` were enriched to store `r5_global_update_norm`, `r5_client_update_norms`, and `r5_client_alignments` for direct programmatic consumption.

---

### 1.2 Verification of Classification Performance & Histological Class 8

| Run ID | Algorithm | Random Seed | FedProx $\mu$ | Duration | Best Val Acc | Test Accuracy | Test Loss | Test Macro-F1 | Class 8 Acc (Adenocarcinoma) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | FedAvg | 42 | N/A | 5.11 min | 76.15% | **77.56%** | 0.7595 | 0.7130 | 79.72% (979 / 1,233) |
| **2** | FedProx | 42 | 0.01 | 8.74 min | 75.05% | **74.26%** | 0.8277 | 0.6812 | **80.54%** (989 / 1,233) |
| **3** | FedAvg | 101 | N/A | 7.75 min | 78.98% | **81.06%** | 0.5940 | 0.7373 | 92.13% (1,136 / 1,233) |
| **4** | FedProx | 101 | 0.01 | 8.66 min | 74.30% | **79.55%** | 0.6505 | 0.7308 | **92.70%** (1,143 / 1,233) |
| **5** | FedAvg | 2024 | N/A | 15.04 min | 72.83% | **73.12%** | 0.8200 | 0.6853 | 64.96% (801 / 1,233) |
| **6** | FedProx | 2024 | 0.01 | 9.08 min | 69.64% | **70.31%** | 0.9452 | 0.6419 | **69.67%** (859 / 1,233) |
| **Mean** | **FedAvg** | **3 Seeds** | **N/A** | **9.30 min** | **$75.99\% \pm 2.51\%$** | **$77.25\% \pm 3.25\%$** | **$0.7245 \pm 0.0955$** | **$0.7118 \pm 0.0213$** | **$78.94\% \pm 11.11\%$** |
| **Mean** | **FedProx** | **3 Seeds** | **0.01** | **8.83 min** | **$73.00\% \pm 2.39\%$** | **$74.71\% \pm 3.79\%$** | **$0.8078 \pm 0.1212$** | **$0.6846 \pm 0.0364$** | **$80.97\% \pm 9.41\%$** |
| **Delta** | **FedProx - FedAvg** | — | — | — | **$-2.99\text{ pp}$** | **$-2.54\text{ pp}$** | **$+0.0833$** | **$-0.0272$** | **$+2.03\text{ pp}$** |

*Checkpoint Re-evaluation:* Independently validated through `scripts/verify_checkpoints_independent.py` via forward pass over all 7,180 test samples: **0.000000 divergence**.

---

### 1.3 Scientific Language & Scope Disclaimers
1. **Statistical Caution:** Differences represent observed experimental performance across 3 independent random seeds ($N = 3$) under a Dirichlet concentration $\alpha = 0.3$. They are not claimed to represent asymptotic statistical significance under formal null-hypothesis testing.
2. **Clinical Non-Applicability:** PathMNIST patches ($28 \times 28$ pixels) are an algorithmic benchmark for evaluating machine learning optimization dynamics. No findings should be interpreted as validated for diagnostic or clinical decision support.
3. **Domain Scope:** The observed proximal shrinkage is a characteristic mathematical regularization under the tested configuration and should not be generalized as universal noise immunity across all architectures or noise distributions.

---

## 2. GitHub Repository Status & Publication Verification

### 2.1 Repository Metadata
- **Owner:** `Nanditha264`
- **Repository Name:** `Phoenix-Federated-Learning`
- **Live Public URL:** `https://github.com/Nanditha264/Phoenix-Federated-Learning`
- **Configured Git Remote:** `https://github.com/Nanditha264/Phoenix-Federated-Learning.git`
- **Default Branch:** `main`
- **Repository Visibility:** **PUBLIC** (HTTP 200 OK verified via unauthenticated request)
- **Description:** `Reproducible federated learning benchmark comparing FedAvg and FedProx on PathMNIST under non-IID client partitioning and controlled label noise.`
- **Published Commit SHA:** `f8c74f1403bc02f44cb13c178902678c828d3884` (Short SHA: `f8c74f1`)
- **Remote Synchronization:** Upstream tracking configured (`origin/main`), fully synchronized.

### 2.2 Git Commit History
```text
f8c74f1 (HEAD -> main, origin/main) feat(wandb): add duplicate detection and stable run ID matching to importer
13ae45b docs: update final publication release report with corrected per-seed norms and archive catalog
19a8ce8 fix(reporting): clarify per-seed vs aggregate update norms and enrich detailed table
a514489 docs: add final publication release report
6a625ba docs: add post-hoc W&B importer and clarify clinical/statistical caveats in README
0f27211 Initial research release: FedAvg vs FedProx on PathMNIST
```

### 2.3 Tracked File Inventory & Structure
- **Total Tracked Files:** 165 files
- **Total Tracked Repository Size:** 117.03 MB
- **Large File Compliance:** 0 files exceed GitHub's 50 MB / 100 MB limits (largest files are model checkpoints at 2.23 MB each).
- **Directory Breakdown:**
  - `.` (Root): 18 files (README.md, REPRODUCIBILITY.md, RESEARCH_SUMMARY.pdf, requirements.txt, environment.yaml, verification and audit reports)
  - `configs/`: 3 files (base.yaml, fedavg.yaml, fedprox.yaml)
  - `data/`: 14 files (PathMNIST canonical partition generators, splits, and verified MD5 hashes)
  - `evaluation/`: 3 files (evaluator, per-class metrics, confusion matrices)
  - `federated/`: 6 files (FedAvg and FedProx server and client implementations)
  - `figures/`: 14 files (6 publication-quality figures in vector PDF and high-res PNG formats, figure_source_data_mapping.json, and FIGURE_DATA_MAPPING.md)
  - `metrics/`: 4 files (update geometry, norm shrinkage, cosine similarity calculators)
  - `models/`: 3 files (FedAvgCNN architecture, initialization, forward pass)
  - `results/`: 65 files (6 primary benchmark runs, per-round JSONL telemetry, summary.json, checkpoints)
  - `scripts/`: 15 files (primary matrix execution, figure generation, analysis, verification, W&B importer)
  - `tests/`: 11 files (30 passing automated unit/integration tests)
  - `tracking/`: 3 files (W&B logger, local JSONL logger)
  - `utils/`: 6 files (seed manager, logger, compute configuration, directory management)

### 2.4 Security & Integrity Audit
- **Secret Scan Outcome:** Automated regex scan over all tracked text files confirmed **0 secrets, 0 API keys, 0 private tokens**.
- **Working Tree State:** Clean, verified via `git status`.
- **Automated Tests:** 30 / 30 tests passing (`pytest tests -q`).
- **Independent Verification:** Forward pass across 7,180 test samples verified 0.000000 divergence from reported test metrics.

### 2.5 Intentionally Excluded Assets & Rationale
1. **Virtual Environments (`fl_env/`):** Excluded per `.gitignore`; standard dependency specifications provided via `requirements.txt` and `environment.yaml`.
2. **MedMNIST Raw Download Cache (`data/cache/`):** Excluded to keep repository lean; canonical NPZ dataset checksum (`md5: 7f42713851b97d8f6737174fdd91f7aa`) verified in automated test suite and downloaded on-demand if absent.
3. **Submission ZIP Archives (`*.zip`):** Preserved locally (`D:\Reasearch\*.zip`) as standalone versioned delivery bundles.
4. **Historical Pilot Hang Logs (`results/primary_benchmark/diagnostic_attempt1_hang_logs/`):** Isolated per Phase 5 protocol to prevent cluttering primary benchmark results.

---

## 3. Weights & Biases (W&B) Status & Synchronization Protocol

### 3.1 Environment Status
- **CLI Executable:** `D:\Reasearch\fl_env\Scripts\wandb.exe` (Version 0.31.0 installed).
- **Authentication Status:** No API key configured on host (`wandb.errors.errors.UsageError: No API key configured. Use wandb login to log in.`).
- **Telemetry State:** Preserved locally as structured application logs (`metrics.jsonl` and `summary.json`) across all 6 runs in `results/primary_benchmark/`.

### 3.2 Importer Tooling (`scripts/import_to_wandb.py`)
- Reads the authentic local telemetry.
- Uploads round-level records (1 through 5) in chronological order with global validation accuracy/loss, training loss, client update norms, and alignments.
- Logs step 6 as final test evaluation (accuracy, loss, macro-F1, per-class breakdown).
- Uploads run summaries and configuration metadata.
- Preserves all local files with zero modifications.

### 3.3 Prepared Experiment Runs
1. `fedavg_dir0.3_noise0.4_seed42` (5 rounds, Test Acc: $77.56\%$, Class 8: $79.72\%$)
2. `fedprox_mu0.01_dir0.3_noise0.4_seed42` (5 rounds, Test Acc: $74.26\%$, Class 8: $80.54\%$)
3. `fedavg_dir0.3_noise0.4_seed101` (5 rounds, Test Acc: $81.06\%$, Class 8: $92.13\%$)
4. `fedprox_mu0.01_dir0.3_noise0.4_seed101` (5 rounds, Test Acc: $79.55\%$, Class 8: $92.70\%$)
5. `fedavg_dir0.3_noise0.4_seed2024` (5 rounds, Test Acc: $73.12\%$, Class 8: $64.96\%$)
6. `fedprox_mu0.01_dir0.3_noise0.4_seed2024` (5 rounds, Test Acc: $70.31\%$, Class 8: $69.67\%$)

### 3.4 Action to Complete W&B Cloud Synchronization
Execute in your terminal:
```powershell
& D:\Reasearch\fl_env\Scripts\wandb.exe login
& D:\Reasearch\fl_env\Scripts\python.exe D:\Reasearch\fl_benchmark\scripts\import_to_wandb.py --project fedavg-vs-fedprox-pathmnist
```

---

## 4. Release Archives Catalog

All research assets have been packaged into versioned, immutable ZIP archives:

| Archive Version | File Path | File Count | Compressed Size | SHA-256 Checksum | Integrity | Status |
|:---|:---|:---:|:---:|:---|:---:|:---:|
| **Baseline (v1.0)** | `D:\Reasearch\PHOENIX_FEDERATED_LEARNING_SUBMISSION_v1.0_baseline.zip` | 159 | 106.79 MB | `C099DEE8E734A9765DE397086D57325AC62801603158B329A8500E2DE63F663E` | 100% OK | Preserved |
| **Release (v1.1)** | `D:\Reasearch\PHOENIX_FEDERATED_LEARNING_SUBMISSION_v1.1_final.zip` | 162 | 106.80 MB | `9E14D9F611FF105A6F04298C4BA9BE8180B02A3B5F907EC3A0471B786B32CA30` | 100% OK | Preserved |
| **Corrected (v1.2)** | `D:\Reasearch\PHOENIX_FEDERATED_LEARNING_SUBMISSION_v1.2_corrected.zip` | **163** | **106.80 MB** (111,990,729 B) | `44F2C9D12EC4023FB209699EB8EF60B3A9F5493F3C8B89B3C12A2775A4B04E0D` | **100% OK** | **Current Release** |

---

## 5. Pristine Upstream Invariance
- `D:\Reasearch\PFLlib-upstream` remains untouched and identical to official commit `0169ba7`.
- `git status` verifies: `On branch master, nothing to commit, working tree clean`.

---
*Report certified by Senior Research Software Engineer and Release Engineer.*
