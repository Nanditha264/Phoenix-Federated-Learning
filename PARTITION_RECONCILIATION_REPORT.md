# PARTITION RECONCILIATION REPORT — CANONICAL ARTIFACT RESOLUTION

**Project:** Federated Learning Experimental Benchmark: FedAvg vs FedProx on PathMNIST  
**Date:** November 8, 2026  
**Auditor:** Principal Federated Learning Research Auditor & Systems Specialist  
**Framework Root:** `D:\Reasearch\fl_benchmark`  
**Status:** **RECONCILED & CRYPTOGRAPHICALLY SECURED**  

---

## 1. Discrepancy Overview

During Phase 2 pre-implementation research design, an initial sample count of **11,377 samples** for Client 0 was documented. Subsequently, during the Phase 4 and Phase 4.1 empirical audits, the actual client sample count for Client 0 under primary seed 42 was measured as **10,547 samples**.

This reconciliation audit investigates the root cause, establishes the canonical implementation, and registers cryptographically verified partition files for all experimental seeds.

---

## 2. Root Cause Investigation & Forensic Audit

A forensic examination of the repository and scratch directories revealed the exact cause:

1. **Phase 2 Pre-Download Test Script (`scratch/test_partition.py`):**
   - In Phase 2, before the official `pathmnist.npz` dataset archive was downloaded, synthetic pseudo-labels were generated via a helper function `build_split_labels(train_class_counts)` using assumed class counts:
     `train_class_counts = [10407, 10566, 10512, 10574, 9415, 10049, 9896, 9464, 9113]` (summing to 89,996).
   - Because the synthetic labels were arranged in contiguous uniform blocks and had hypothetical class distributions, the Dirichlet Non-IID allocation ($\alpha = 0.3$, seed 42) divided the synthetic arrays into:
     `[11377, 18161, 25410, 17764, 17284]`
     with Client 0 noise: $\text{round}(0.40 \times 11377) = 4,551$ corrupted samples.
2. **Phase 4 Implementation on Real Dataset (`fl_benchmark/data/partition.py`):**
   - In Phase 3, the official MedMNIST `pathmnist.npz` archive was downloaded and verified via MD5 `a8b06965200029087d5bd730944a56c1`.
   - Inspection of the actual `train_labels` in `pathmnist.npz` revealed the true PathMNIST training class distribution:
     `[9366, 9509, 10360, 10401, 8006, 12182, 7886, 9401, 12885]` (summing to 89,996).
   - Because the true labels reflect the natural histology dataset class balance and sample ordering, running the canonical Dirichlet Non-IID allocation ($\alpha = 0.3$, seed 42) produces:
     `[10547, 20614, 17308, 21484, 20043]`
     with Client 0 noise: $\text{round}(0.40 \times 10547) = 4,219$ corrupted samples.

**Conclusion:** The 11,377 figure was an artifact of pre-implementation synthetic pseudo-labels. The 10,547 figure is the **true, canonical allocation on the official MedMNIST PathMNIST dataset**.

---

## 3. Canonical Partition Registry & Cryptographic Signatures

The canonical partition implementation (`dirichlet_partition()` in `fl_benchmark/data/partition.py`) was executed across all three primary benchmark seeds (42, 101, 2024). The exact partitioned indices and corruption masks were serialized to compressed `.npz` archives inside `fl_benchmark/data/partitions/`:

### Primary Seed 42
- **File:** `fl_benchmark/data/partitions/partition_dir0.3_clients5_seed42.npz`
- **SHA-256 Checksum:** `5d45b6280dd3eee7b5b7cd3b73c2cdecfee65959a963e100dc3fa701c23d8c76`
- **Client Sample Allocation:**
  - Client 0: **10,547 samples** (4,219 corrupted, $40.0019\%$)
  - Client 1: **20,614 samples** (0 corrupted, 100% clean)
  - Client 2: **17,308 samples** (0 corrupted, 100% clean)
  - Client 3: **21,484 samples** (0 corrupted, 100% clean)
  - Client 4: **20,043 samples** (0 corrupted, 100% clean)
  - **Total Training Samples:** **89,996** (0 missing, 0 duplicate)

### Primary Seed 101
- **File:** `fl_benchmark/data/partitions/partition_dir0.3_clients5_seed101.npz`
- **SHA-256 Checksum:** `1c7e5f80f36cb6e6aa621b75dc038a287077b12b2894514127340a39e1537fa6`
- **Client Sample Allocation:**
  - Client 0: **13,504 samples** (5,402 corrupted, $40.0030\%$)
  - Client 1: **12,707 samples** (0 corrupted, 100% clean)
  - Client 2: **18,733 samples** (0 corrupted, 100% clean)
  - Client 3: **14,739 samples** (0 corrupted, 100% clean)
  - Client 4: **30,313 samples** (0 corrupted, 100% clean)
  - **Total Training Samples:** **89,996** (0 missing, 0 duplicate)

### Primary Seed 2024
- **File:** `fl_benchmark/data/partitions/partition_dir0.3_clients5_seed2024.npz`
- **SHA-256 Checksum:** `f6b32b01ba717c8437f133eaeb6dc1bd8929bfd4cd5c1f06083bb8af6bec26a6`
- **Client Sample Allocation:**
  - Client 0: **20,836 samples** (8,334 corrupted, $39.9981\%$)
  - Client 1: **3,771 samples** (0 corrupted, 100% clean)
  - Client 2: **15,415 samples** (0 corrupted, 100% clean)
  - Client 3: **19,099 samples** (0 corrupted, 100% clean)
  - Client 4: **30,875 samples** (0 corrupted, 100% clean)
  - **Total Training Samples:** **89,996** (0 missing, 0 duplicate)

---

## 4. Verification Statement

1. All 89,996 training samples are conserved with mathematical exactness across all seeds.
2. Client 0 alone receives symmetric label noise; Clients 1–4 are 100% clean.
3. Every corrupted label satisfies $y' \in \{0, \dots, 8\} \setminus \{y\}$.
4. The official 10,004 validation samples and 7,180 test samples remain 100% clean.
5. All paired experiments (FedAvg and FedProx) within each seed will load the identical serialized partition archive.
