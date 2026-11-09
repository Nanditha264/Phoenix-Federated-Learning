# Phase 5: Experiment Recovery & Safe Resumption Audit Report

**Date & Time:** 2026-11-09 11:15:00 IST  
**Role:** Senior Machine Learning Research Engineer & Experiment Recovery Specialist  
**Workspace:** `D:\Reasearch\fl_benchmark`  
**Python Environment:** `D:\Reasearch\fl_env\Scripts\python.exe`  
**Upstream Repository:** `D:\Reasearch\PFLlib-upstream` (Commit `0169ba7`, Pristine)

---

## 1. Executive Incident Analysis & Recovery Summary

Following the launch of the primary benchmark matrix orchestrator (`scripts/run_primary_matrix.py`, Background Task `task-1067`) on 2026-11-08 at 21:13:08 IST, a host system/server restart occurred overnight between 21:13:21 IST and 11:00:39 IST. The background task was terminated by the host environment restart.

A rigorous forensic audit was immediately conducted to assess the system state, verify process termination, audit data integrity, and inspect all experimental artifacts.

### Key Diagnostic Findings:
1. **Zero Process Contamination:** No orphan, zombie, or background Python or W&B processes are running (`Get-Process python` returned exit code 1; `manage_task list` confirmed 0 active background tasks).
2. **Compute & Memory Health:** Physical RAM is at **2,010 MB available** (75.6% used), well above the 800 MB safety warning threshold, 550 MB client pause threshold, and 350 MB emergency stop floor.
3. **Pristine Upstream:** `D:\Reasearch\PFLlib-upstream` remains strictly unmodified and clean on `master` at commit `0169ba7`.
4. **Artifact Integrity 100% Intact:** PathMNIST dataset cache, all three canonical Dirichlet partitions ($\alpha=0.3$), noise masks (40% Client 0), and all three initial model weight checkpoints are verified by cryptographic hashes.

---

## 2. Comprehensive Inventory of Benchmark Runs

The primary experimental matrix specifies 6 paired runs (5 rounds each, 89,996 samples across 5 clients, Dirichlet $\alpha=0.3$, 40% Client 0 noise):

| Run ID | Algorithm | Random Seed | FedProx $\mu$ | Required Rounds | Current On-Disk Status | Rounds Completed | Recovery Action Required |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | FedAvg | 42 | N/A | 5 | Incomplete / Interrupted | 0 / 5 (12s into init) | Execute full 5-round run |
| **2** | FedProx | 42 | 0.01 | 5 | Incomplete (Phase 4.1 smoke test on disk) | 0 / 5 | Execute full 5-round run |
| **3** | FedAvg | 101 | N/A | 5 | Pending (Directory does not exist) | 0 / 5 | Execute full 5-round run |
| **4** | FedProx | 101 | 0.01 | 5 | Pending (Directory does not exist) | 0 / 5 | Execute full 5-round run |
| **5** | FedAvg | 2024 | N/A | 5 | Pending (Directory does not exist) | 0 / 5 | Execute full 5-round run |
| **6** | FedProx | 2024 | 0.01 | 5 | Pending (Directory does not exist) | 0 / 5 | Execute full 5-round run |

### Forensic Breakdown per Run:
- **Run 1 (`fedavg_dir0.3_noise0.4_seed42`):**
  - Task 1067 initialized the run at 21:13:09, verified data MD5 and canonical partition, registered canonical initial model, and created the offline W&B logger.
  - The process was interrupted at 21:13:21 during Round 1 client dataset initialization.
  - The existing `summary.json` file in that folder was from the earlier 2-round pilot (completed 21:03:38). `metrics.jsonl` was unlinked during initialization to start clean.
  - No 5-round benchmark data exists. Needs a full 5-round execution.
- **Run 2 (`fedprox_mu0.01_dir0.3_noise0.4_seed42`):**
  - Contains artifacts from the Phase 4.1 smoke test (1 round on 128 subsampled samples, duration 0.96s, val acc 11.33%).
  - Has not been executed for the 5-round full-data benchmark. Needs a full 5-round execution.
- **Runs 3 to 6 (Seeds 101 & 2024):**
  - Not yet started. Pristine canonical partitions and model weights are pre-registered and ready.

---

## 3. Cryptographic Provenance & Artifact Integrity Verification

All canonical partitions, dataset caches, and initial model weights have been audited and verified:

### 3.1 Dataset Verification
- **File:** `D:\Reasearch\dataset_cache\pathmnist.npz`
- **MD5 Hash:** `a8b06965200029087d5bd730944a56c1` (Verified official MedMNIST v2 release)
- **Splits:** 89,996 train | 10,004 val | 7,180 test | 9 classes | $28 \times 28 \times 3$ RGB

### 3.2 Canonical Partition Integrity (`data/partitions/`)
| Seed | Partition File | SHA-256 Checksum | Client Sample Allocations | Client 0 Noise (40%) |
|:---:|:---|:---|:---|:---:|
| **42** | `partition_dir0.3_clients5_seed42.npz` | `5D45B6280DD3EEE7B5B7CD3B73C2CDECFEE65959A963E100DC3FA701C23D8C76` | `[10547, 20614, 17308, 21484, 20043]` | 4,219 / 10,547 (40.00%) |
| **101** | `partition_dir0.3_clients5_seed101.npz` | `1C7E5F80F36CB6E6AA621B75DC038A287077B12B2894514127340A39E1537FA6` | `[13504, 12707, 18733, 14739, 30313]` | 5,402 / 13,504 (40.00%) |
| **2024** | `partition_dir0.3_clients5_seed2024.npz` | `F6B32B01BA717C8437F133EAEB6DC1BD8929BFD4CD5C1F06083BB8AF6BEC26A6` | `[20836, 3771, 15415, 19099, 30875]` | 8,334 / 20,836 (40.00%) |

### 3.3 Canonical Initial Model Weights (`data/initial_models/`)
Bit-level identical initial parameters are pre-registered and cryptographically anchored to guarantee fair paired comparisons between FedAvg and FedProx:
| Seed | Model File | SHA-256 Checksum | Parameters | Architecture |
|:---:|:---|:---|:---:|:---:|
| **42** | `initial_model_seed42.pt` | `25adb902df84e61404a9b9f04f6afd57e460d574c7b481b1ef1459e5e05158eb` | 583,113 | FedAvgCNN |
| **101** | `initial_model_seed101.pt` | `6a7e7baf4bb71afa19beb49c0bda1fcfa54c3d9e7c97ed06a237fa3f5501aae7` | 583,113 | FedAvgCNN |
| **2024** | `initial_model_seed2024.pt` | `efd637bac752e5c57b030b0e5f31fc04d09c48b3951457a85c6e31339b4c4787` | 583,113 | FedAvgCNN |

---

## 4. Experiment Recovery & Idempotency Engineering

To make the benchmark matrix resilient against any future server or execution interruptions, the primary matrix orchestrator (`scripts/run_primary_matrix.py`) has been upgraded with an **Idempotent Recovery Guard**:

```python
def check_existing_completion(run_name, required_rounds):
    run_dir = PROJECT_ROOT / "results" / run_name
    summary_file = run_dir / "summary.json"
    metrics_file = run_dir / "metrics.jsonl"
    best_ckpt = run_dir / "checkpoints" / "best_checkpoint.pt"

    if summary_file.exists() and metrics_file.exists() and best_ckpt.exists():
        try:
            with open(summary_file, "r") as f:
                data = json.load(f)
            rounds_done = data.get("rounds_completed", 0)
            has_test = "test_evaluation" in data and "accuracy" in data["test_evaluation"]
            if rounds_done >= required_rounds and has_test:
                return True, data
        except Exception:
            return False, {}
    return False, {}
```

### Protection Guarantees:
1. **Zero Redundant Computation:** If a run has completed its full required communication rounds ($R=5$) and evaluated on the test set, it is marked as completed, its results are logged, and the orchestrator immediately advances to the next incomplete experiment.
2. **Artifact Preservation:** Checkpoints, JSONL telemetry, and W&B offline run logs of completed runs will never be overwritten.
3. **Safe Incomplete Re-Run:** Any run that was interrupted mid-execution (such as Run 1) will execute cleanly from Round 1 to 5 with fresh telemetry and verified checkpointing.

---

## 5. Resumption Readiness & Safe Execution Protocol

The environment is in a verified, clean state and ready for resumption:
- **Execution Script:** `D:\Reasearch\fl_benchmark\scripts\run_primary_matrix.py`
- **Total Runs to Execute:** 6 (5 rounds each on 89,996 samples)
- **Estimated Runtime per Run:** $\approx 7.5 - 9.0 \text{ minutes}$ (Sequential CPU, 4 intra-op threads)
- **Total Estimated Duration:** $\approx 48 - 52 \text{ minutes}$
- **Safety Floors Active:** Warning at 800 MB RAM, Pause at 550 MB RAM, Emergency Stop at 350 MB RAM.

The system is now fully prepared to resume the benchmark execution.
