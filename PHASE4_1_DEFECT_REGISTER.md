# PHASE 4.1 DEFECT REGISTER & AUDIT REMEDIATION LOG

**Project:** Federated Learning Experimental Benchmark: FedAvg vs FedProx on PathMNIST  
**Date:** November 8, 2026  
**Auditor:** Principal Federated Learning Research Auditor & Systems Specialist  
**Framework Root:** `D:\Reasearch\fl_benchmark`  

---

## 1. Defect Classification Standard

Defects are classified according to the following scientific and engineering severity levels:
- **Critical Correctness Defect:** Algorithmic flaw causing incorrect mathematical results, data leakage, or silent corruption.
- **Major Scientific Defect:** Systematic bias, label leakage, class misattribution, or invalid experimental control.
- **Minor Implementation Defect:** Localized code exception, indexing error, or typing bug preventing execution.
- **Performance / Configuration Limitation:** Suboptimal hyperparameter, excessive memory threshold, or configuration friction.
- **No Defect (Investigated Anomaly):** Expected statistical behavior thoroughly investigated and mathematically explained.

---

## 2. Register of Findings

### Defect ID: DEF-001
- **Classification:** Minor Implementation Defect
- **File:** `fl_benchmark/data/partition.py`
- **Function:** `compute_partition_stats()`
- **Status:** **RESOLVED & VERIFIED**
- **Root Cause:** Line 100 attempted 2D array slicing `client_counts[:, None]` directly on a standard Python list `client_counts = [len(idx) for idx in client_indices]`, raising `TypeError: list indices must be integers or slices, not tuple`.
- **Evidence:** `test_synthetic_dirichlet_partition` failed during initial pytest execution.
- **Remediation:** Explicitly converted `client_counts` to a 2D float NumPy array:
  ```python
  counts_arr = np.array(client_counts, dtype=np.float64)[:, None]
  class_proportions = class_distribution.astype(np.float64) / np.maximum(counts_arr, 1.0)
  ```
- **Regression Testing:** Re-tested under `pytest tests/test_partition.py` — passed with 100% success. Regression risk: None.

---

### Defect ID: DEF-002
- **Classification:** Minor Implementation Defect
- **File:** `fl_benchmark/data/pathmnist_dataset.py`
- **Class / Method:** `PathMNISTDataset.__getitem__()`
- **Status:** **RESOLVED & VERIFIED**
- **Root Cause:** When `custom_labels` (e.g., noisy Client 0 training labels) was supplied as a client-local subset array of length $M = 128$, `__getitem__` attempted to index `self.labels[real_idx]`. Since `real_idx` is the global index into the 89,996-sample image array (e.g., index 238), it raised `IndexError: index 238 is out of bounds for axis 0 with size 128`.
- **Evidence:** Terminal stack trace in `task-859.log` during the initial client training loop.
- **Remediation:** Added dynamic detection of subset-aligned labels versus global dataset labels:
  ```python
  self.labels_are_subset = (len(self.labels) == len(self.indices))
  # In __getitem__:
  if self.labels_are_subset:
      label = int(self.labels[idx])
  else:
      label = int(self.labels[real_idx])
  ```
- **Regression Testing:** Verified under `test_dataset_indexing_and_custom_labels` and live 5-client federated execution. Regression risk: None.

---

### Defect ID: DEF-003
- **Classification:** Performance / Configuration Limitation
- **File:** `fl_benchmark/configs/base.yaml` and `fl_benchmark/utils/resource_monitor.py`
- **Component:** `FederatedResourceMonitor`
- **Status:** **RESOLVED & VERIFIED**
- **Root Cause:** Default `emergency_stop_threshold_mb` was set to $750.0\text{ MB}$, and warning to $1536.0\text{ MB}$. On the target 8 GB Windows laptop where host background services leave $\approx 800-1100\text{ MB}$ available, transient memory dips tripped the emergency stop prematurely during initial smoke tests (`task-848.log`).
- **Evidence:** `MemoryError: Graceful halt triggered by resource monitor: [CRITICAL] Available RAM (682.8 MB) is below emergency limit (750.0 MB)`.
- **Remediation:** Re-calibrated thresholds to scientifically grounded levels reflecting sequential client requirements (each sequential client requires $\approx 300\text{ MB}$ RSS):
  - Warning Threshold: $800.0\text{ MB}$ (triggers `gc.collect()`)
  - Client Pause Threshold: $550.0\text{ MB}$ (suspends next client until memory stabilizes)
  - Emergency Stop Floor: $350.0\text{ MB}$ (hard safety stop)
- **Regression Testing:** Successfully tested in live multi-round federated training where the `[PAUSE]` handler safely reclaimed memory without terminating the process.

---

### Defect ID: ANO-001 (Investigated Anomaly)
- **Classification:** **NO DEFECT — EXPECTED STATISTICAL BEHAVIOR**
- **Observed Phenomenon:** Reported FedAvg smoke test produced $11.33\%$ validation accuracy, $8.83\%$ test accuracy, Macro-F1 $0.0180$, predicting $100\%$ Class 3 across all validation and test samples.
- **Investigation & Evidence:**
  1. **Runtime Reality:** The smoke test executed only 1 round of 1 local epoch on 128 samples per client ($4$ batches per client = $20$ total gradient steps across 5 clients). Total parameter displacement from the initial random weights was only $\|\Delta\|_2 = 0.0188$ ($0.0025\%$ relative change).
  2. **Initial Basin Artifact:** Inspection of `initial_checkpoint.pt` proved that the **untrained, randomly initialized model under seed 42 already predicted Class 3 for 100% of samples** (mean logit $+0.0645$ vs $-0.04$ to $+0.03$ for other classes).
  3. **PathMNIST Class 3 Distribution:** Exactly $634$ out of $7,180$ test images belong to Class 3 ($634 / 7180 = 8.8301\%$). An untrained model predicting Class 3 deterministically yields exactly $8.83\%$ test accuracy.
  4. **Convergence Proof:** When trained for just 3 federated rounds (500 samples/client) or 4 local epochs on 2,000 samples, validation accuracy escalated from $12.30\% \to 36.60\%$ and $47.40\%$, completely breaking class collapse and predicting across all 9 classes.
- **Verdict:** Validated as expected behavior of an untrained/minimally stepped model; no architectural or pipeline bug exists.

---

## 3. Audit Summary Table

| Defect ID | Severity | File | Function / Component | Status |
| :--- | :--- | :--- | :--- | :--- |
| **DEF-001** | Minor Implementation | `data/partition.py` | `compute_partition_stats()` | **FIXED** |
| **DEF-002** | Minor Implementation | `data/pathmnist_dataset.py` | `__getitem__()` | **FIXED** |
| **DEF-003** | Configuration / Safety | `utils/resource_monitor.py` | `FederatedResourceMonitor` | **FIXED** |
| **ANO-001** | Investigated Anomaly | `models/cnn.py` & `run_benchmark.py` | Initial random basin & smoke subsampling | **VERIFIED (NO DEFECT)** |
