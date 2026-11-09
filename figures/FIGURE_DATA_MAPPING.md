# Figure to Empirical Source Data Mapping

This document provides the lineage and exact mapping from the 6 publication figures to the underlying empirical benchmark artifacts in `results/primary_benchmark/`.

| Figure File | Visualization | Primary Metric | Empirical Source Artifacts | Key Findings Displayed |
| :--- | :--- | :--- | :--- | :--- |
| **Figure 1**<br>`fig1_val_accuracy_vs_rounds.{png,pdf}` | Validation Accuracy vs. Rounds | `eval/val_acc` | `metrics.jsonl` (Rounds 1–5 across 3 seeds per algorithm) | Convergence trajectories; FedAvg achieves $75.99\% \pm 2.51\%$ vs. FedProx $73.00\% \pm 2.39\%$ best validation accuracy. |
| **Figure 2**<br>`fig2_val_loss_vs_rounds.{png,pdf}` | Validation Loss vs. Rounds | `eval/val_loss` | `metrics.jsonl` (Rounds 1–5 across 3 seeds per algorithm) | Global validation loss trajectories across communication rounds. |
| **Figure 3**<br>`fig3_update_norm_shrinkage.{png,pdf}` | Update Norm Shrinkage | `geometry/global_update_norm`<br>`geometry/client_update_norms` | `metrics.jsonl` & `summary.json` (Round 5 values across all 3 seeds) | Proximal regularization effect: Global norm shrinks by $-29.99\%$ ($2.4838 \rightarrow 1.7388$); Noisy Client 0 norm dampened by $-36.22\%$ ($3.6420 \rightarrow 2.3229$). |
| **Figure 4**<br>`fig4_update_cosine_alignment.{png,pdf}` | Update Cosine Alignment | `geometry/alignment_with_global` | `metrics.jsonl` (Rounds 1–5 across 3 seeds per algorithm) | Clean client (Clients 1–4) directional alignment with global update; FedProx improves alignment from $0.5175$ to $0.5976$ (+15.48%). |
| **Figure 5**<br>`fig5_pairwise_cosine_heatmaps.{png,pdf}` | Pairwise Client Cosine Heatmaps | `geometry/pairwise_cosine_matrix` | `metrics.jsonl` (Round 5 pairwise client similarity matrices averaged across 3 seeds) | Inter-client update consistency under heterogeneous Dirichlet ($\alpha = 0.3$) distribution and noisy Client 0. |
| **Figure 6**<br>`fig6_per_class_test_accuracy.{png,pdf}` | Per-Class Test Accuracy Breakdown | `test_evaluation/per_class_accuracy` | `summary.json` (Test evaluation across 9 tissue classes on 7,180 test samples) | Class-by-class comparison; highlighting Class 8 (Colorectal Adenocarcinoma): FedAvg 78.94% vs. FedProx 80.97% (+2.03 pp). |

---
*Machine-readable mapping is preserved in `figures/figure_source_data_mapping.json`.*
