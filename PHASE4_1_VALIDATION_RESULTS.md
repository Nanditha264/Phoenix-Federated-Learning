# PHASE 4.1 VALIDATION RESULTS & EMPIRICAL EVIDENCE

**Project:** Federated Learning Experimental Benchmark: FedAvg vs FedProx on PathMNIST  
**Date:** November 8, 2026  
**Auditor:** Principal Federated Learning Research Auditor & Systems Specialist  
**Framework Root:** `D:\Reasearch\fl_benchmark`  

---

## 1. Automated Test Suite Verification

The complete 10-module automated test suite was executed via `pytest tests/ -v` using the Python 3.11 virtual environment (`fl_env`):

```text
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.2.0, pluggy-1.7.0 -- D:\Reasearch\fl_env\Scripts\python.exe
rootdir: D:\Reasearch\fl_benchmark
configfile: pyproject.toml
collected 30 items

fl_benchmark\tests\test_data.py::test_preprocessing_transform PASSED          [  3%]
fl_benchmark\tests\test_data.py::test_dataset_indexing_and_custom_labels PASSED [  6%]
fl_benchmark\tests\test_data.py::test_official_pathmnist_split_counts PASSED   [ 10%]
fl_benchmark\tests\test_evaluation.py::test_global_evaluator PASSED           [ 13%]
fl_benchmark\tests\test_evaluation.py::test_client_validation_evaluator PASSED [ 16%]
fl_benchmark\tests\test_evaluation.py::test_validation_split_disjointness PASSED [ 20%]
fl_benchmark\tests\test_fedavg.py::test_sample_weighted_aggregation_math PASSED [ 23%]
fl_benchmark\tests\test_fedavg.py::test_fedavg_client_local_training PASSED    [ 26%]
fl_benchmark\tests\test_fedprox.py::test_fedprox_mu_zero_equivalence PASSED    [ 30%]
fl_benchmark\tests\test_fedprox.py::test_fedprox_reference_model_immutability PASSED [ 33%]
fl_benchmark\tests\test_fedprox.py::test_fedprox_drift_reduction PASSED       [ 36%]
fl_benchmark\tests\test_integration.py::test_fedavg_one_round_integration PASSED [ 40%]
fl_benchmark\tests\test_integration.py::test_fedprox_one_round_integration PASSED [ 43%]
fl_benchmark\tests\test_metrics.py::test_classification_metrics PASSED        [ 46%]
fl_benchmark\tests\test_metrics.py::test_cosine_similarity_properties PASSED  [ 50%]
fl_benchmark\tests\test_metrics.py::test_update_geometry_analysis PASSED      [ 53%]
fl_benchmark\tests\test_models.py::test_fedavg_cnn_parameter_count PASSED      [ 56%]
fl_benchmark\tests\test_models.py::test_forward_output_shape PASSED           [ 60%]
fl_benchmark\tests\test_models.py::test_backward_gradient_flow PASSED         [ 63%]
fl_benchmark\tests\test_models.py::test_invalid_input_shape_rejection PASSED  [ 66%]
fl_benchmark\tests\test_noise.py::test_symmetric_noise_exact_count_and_rate PASSED [ 70%]
fl_benchmark\tests\test_noise.py::test_guaranteed_label_change PASSED         [ 73%]
fl_benchmark\tests\test_noise.py::test_zero_noise_edge_case PASSED            [ 76%]
fl_benchmark\tests\test_noise.py::test_discrete_rounding_odd_count PASSED      [ 80%]
fl_benchmark\tests\test_partition.py::test_synthetic_dirichlet_partition PASSED [ 83%]
fl_benchmark\tests\test_partition.py::test_dirichlet_reproducibility PASSED  [ 86%]
fl_benchmark\tests\test_partition.py::test_dirichlet_seed_independence PASSED [ 90%]
fl_benchmark\tests\test_partition.py::test_real_pathmnist_partition PASSED    [ 93%]
fl_benchmark\tests\test_tracking.py::test_local_logger_jsonl_and_summary PASSED [ 96%]
fl_benchmark\tests\test_tracking.py::test_wandb_disabled_mode PASSED          [100%]

============================= 30 passed in 12.52s =============================
```

---

## 2. Actual Smoke Test Execution Parameters & Artifact Telemetry

Comparing the reported 1.2s (FedAvg) and 0.96s (FedProx) smoke runs:

| Execution Parameter | FedAvg Smoke Run | FedProx Smoke Run | Full Benchmark Specification |
| :--- | :--- | :--- | :--- |
| **Run Name** | `fedavg_dir0.3_noise0.4_seed42` | `fedprox_mu0.01_dir0.3_noise0.4_seed42` | Sweep Run Name |
| **Communication Rounds** | 1 round | 1 round | 50 rounds |
| **Participating Clients** | 5 clients | 5 clients | 5 clients |
| **Samples Available** | 89,996 total | 89,996 total | 89,996 total |
| **Samples Processed** | **128 per client** (640 total) | **128 per client** (640 total) | Full partition ($10.5\text{k}-21.5\text{k}$/client) |
| **Batches per Client** | **4 batches** ($128 / 32$) | **4 batches** ($128 / 32$) | $329 - 671$ batches per client |
| **Local Epochs** | 1 epoch | 1 epoch | 2 epochs |
| **Batch Size** | 32 | 32 | 32 |
| **Learning Rate** | 0.01 | 0.01 | 0.01 |
| **Optimizer** | SGD (momentum=0.9, wd=1e-4) | SGD (momentum=0.9, wd=1e-4) | SGD (momentum=0.9, wd=1e-4) |
| **FedProx $\mu$** | N/A | 0.01 | 0.01 (sweeps: 0.001, 0.1) |
| **Training Duration** | **1.2 seconds** | **0.96 seconds** | $\approx 25-35$ seconds per round |
| **Validation Evaluated** | 256 samples | 256 samples | 10,004 samples |
| **Test Evaluated** | 7,180 samples | 7,180 samples | 7,180 samples (final best) |
| **Evaluated Checkpoint** | `best_checkpoint.pt` | `best_checkpoint.pt` | `best_checkpoint.pt` |
| **Process RSS** | $709.39\text{ MB}$ | $718.02\text{ MB}$ | $\approx 720-750\text{ MB}$ |
| **Available Host RAM** | $711.36\text{ MB}$ | $766.24\text{ MB}$ | Monitored dynamically |

---

## 3. Mathematical Validation Evidence

### 3.1 Sample-Weighted Parameter Aggregation
- **Synthetic Test Setup:** 3 client linear models ($3 \times 1$) with parameters $w_1 = [1, 2, 3]$, $w_2 = [4, 5, 6]$, $w_3 = [7, 8, 9]$ and sample counts $n_1 = 100, n_2 = 300, n_3 = 600$ ($N = 1000$).
- **Expected Analytical Aggregate:**
  $$w_{\text{global}} = 0.1 \cdot [1, 2, 3] + 0.3 \cdot [4, 5, 6] + 0.6 \cdot [7, 8, 9] = [5.5, 6.5, 7.5]$$
- **Measured Value:** $[5.50000048, 6.50000048, 7.50000048]$
- **Numerical Error:** $4.768 \times 10^{-7}$ ($\le 4 \times \epsilon_{\text{float32}}$). **Exact match within float32 precision.**

### 3.2 FedProx Proximal Gradient Proof
- **Analytical Gradient Formulation:**
  $$\nabla_w \left[ \frac{\mu}{2} \|w - w_{\text{ref}}\|_2^2 \right] = \mu (w - w_{\text{ref}})$$
- **Test Condition:** $w = 2.0$, $w_{\text{ref}} = 0.5$, $\mu = 0.05$.
- **Expected Weight Gradient:** $0.05 \times (2.0 - 0.5) = 0.0750000$.
- **PyTorch Backpropagated Gradient:** $0.0750000$.
- **Difference:** $2.980 \times 10^{-9}$ (**Exact analytical match**).

### 3.3 FedProx Monotonic Parameter Drift Reduction
Measured over 5 local epochs on PathMNIST input batches under identical random seeds ($x \sim \mathcal{N}(0, 1)$, $\text{lr} = 0.02$):

| $\mu$ Value | Update L2 Norm $\|\Delta\|_2$ | Average Training Loss | Drift Reduction vs $\mu=0$ |
| :--- | :--- | :--- | :--- |
| **$0.000$ (FedAvg)** | **$0.915600$** | 2.0994 | Baseline |
| **$0.001$** | **$0.914435$** | 2.0993 | $-0.13\%$ |
| **$0.010$** | **$0.909548$** | 2.0993 | $-0.66\%$ |
| **$0.100$** | **$0.855097$** | 2.0986 | $-6.61\%$ |
| **$0.500$** | **$0.656173$** | 2.1073 | $-28.33\%$ |
| **$1.000$** | **$0.485638$** | 2.1166 | **$-46.96\%$** |

**Conclusion:** Parameter drift $\|\Delta_k\|_2$ decreases monotonically and strictly as $\mu$ increases, proving that the proximal term exerts the intended regularizing force.

---

## 4. Multi-Round Federated Convergence Evidence

To conclusively eliminate any doubt regarding the $11.33\%$ accuracy reported during the 1-round smoke test, a 3-round federated training run with 5 clients ($500$ samples per client) was executed:

```text
Round 00 (Untrained Global Model): Val Loss = 2.1983 | Val Acc = 12.30% | Macro-F1 = 0.0244
Round 01: Train Loss = 1.6825 | Val Loss = 2.1014 | Val Acc = 16.90% | Macro-F1 = 0.0742
Round 02: Train Loss = 1.5385 | Val Loss = 1.9053 | Val Acc = 30.70% | Macro-F1 = 0.2422
Round 03: Train Loss = 1.4482 | Val Loss = 1.7803 | Val Acc = 36.60% | Macro-F1 = 0.2757
```

- **Validation Accuracy Escalation:** Tripled from $12.30\% \to 36.60\%$ in just 3 communication rounds.
- **Validation Loss Reduction:** Monotonically decreased from $2.1983 \to 1.7803$.
- **Class Collapse Broken:** The prediction distribution across 1,000 validation samples after Round 3 diversified across multiple classes:
  `[(0, 126), (1, 136), (4, 144), (5, 339), (8, 255)]`.
