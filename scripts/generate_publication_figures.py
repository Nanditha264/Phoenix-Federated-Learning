"""
Publication-Quality Research Figures Generator.
Extracts empirical metrics from the 6 primary benchmark runs and generates
high-resolution (300 DPI) PNG and vector PDF figures for research publication.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np

# Set writable Matplotlib config directory inside workspace
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib")
(PROJECT_ROOT / ".matplotlib").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_BENCHMARK_DIR = PROJECT_ROOT / "results" / "primary_benchmark"
SEEDS = [42, 101, 2024]
ALGORITHMS = ["fedavg", "fedprox"]
ROUNDS = [1, 2, 3, 4, 5]

CLASS_NAMES = [
    "0: Adipose",
    "1: Background",
    "2: Debris",
    "3: Lymphocytes",
    "4: Mucus",
    "5: Smooth Muscle",
    "6: Normal Mucosa",
    "7: Cancer Stroma",
    "8: Adenocarcinoma",
]

# Set publication style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})

# 1. Load Data
run_data = {"fedavg": {}, "fedprox": {}}
for algo in ALGORITHMS:
    for seed in SEEDS:
        run_name = f"{algo}_dir0.3_noise0.4_seed{seed}" if algo == "fedavg" else f"fedprox_mu0.01_dir0.3_noise0.4_seed{seed}"
        run_dir = PRIMARY_BENCHMARK_DIR / run_name
        with open(run_dir / "summary.json", "r") as f:
            summary = json.load(f)
        with open(run_dir / "metrics.jsonl", "r") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        round_records = [r for r in lines if "round" in r]
        run_data[algo][seed] = {
            "summary": summary,
            "rounds": round_records,
        }

source_mapping = {}

# ==============================================================================
# FIGURE 1: Global Validation Accuracy vs Communication Rounds
# ==============================================================================
fig, ax = plt.subplots(figsize=(7.5, 5), dpi=300)
colors = {"fedavg": "#1f77b4", "fedprox": "#d62728"}
labels = {"fedavg": "FedAvg (μ=0)", "fedprox": "FedProx (μ=0.01)"}

fig1_data = {}
for algo in ALGORITHMS:
    matrix = np.zeros((len(SEEDS), len(ROUNDS)))
    for i, seed in enumerate(SEEDS):
        accs = [r["eval/val_acc"] * 100 for r in run_data[algo][seed]["rounds"]]
        matrix[i, :] = accs
        ax.plot(ROUNDS, accs, color=colors[algo], alpha=0.25, linestyle=":", linewidth=1.2)
    mean_curve = np.mean(matrix, axis=0)
    std_curve = np.std(matrix, axis=0)
    ax.plot(ROUNDS, mean_curve, color=colors[algo], marker="o", linewidth=2.2, label=f"{labels[algo]} (Mean)")
    ax.fill_between(ROUNDS, mean_curve - std_curve, mean_curve + std_curve, color=colors[algo], alpha=0.15)
    fig1_data[algo] = {"mean": mean_curve.tolist(), "std": std_curve.tolist(), "raw": matrix.tolist()}

ax.set_title("Global Validation Accuracy vs. Communication Rounds (PathMNIST)")
ax.set_xlabel("Communication Round")
ax.set_ylabel("Validation Accuracy (%)")
ax.set_xticks(ROUNDS)
ax.set_ylim(25, 85)
ax.legend(loc="lower right", frameon=True, framealpha=0.9)
fig.savefig(OUTPUT_DIR / "fig1_val_accuracy_vs_rounds.png", dpi=300)
fig.savefig(OUTPUT_DIR / "fig1_val_accuracy_vs_rounds.pdf")
plt.close(fig)
source_mapping["fig1_val_accuracy_vs_rounds"] = "metrics.jsonl: eval/val_acc across Rounds 1-5"

# ==============================================================================
# FIGURE 2: Global Validation Loss vs Communication Rounds
# ==============================================================================
fig, ax = plt.subplots(figsize=(7.5, 5), dpi=300)
fig2_data = {}
for algo in ALGORITHMS:
    matrix = np.zeros((len(SEEDS), len(ROUNDS)))
    for i, seed in enumerate(SEEDS):
        losses = [r["eval/val_loss"] for r in run_data[algo][seed]["rounds"]]
        matrix[i, :] = losses
        ax.plot(ROUNDS, losses, color=colors[algo], alpha=0.25, linestyle=":", linewidth=1.2)
    mean_curve = np.mean(matrix, axis=0)
    std_curve = np.std(matrix, axis=0)
    ax.plot(ROUNDS, mean_curve, color=colors[algo], marker="s", linewidth=2.2, label=f"{labels[algo]} (Mean)")
    ax.fill_between(ROUNDS, mean_curve - std_curve, mean_curve + std_curve, color=colors[algo], alpha=0.15)
    fig2_data[algo] = {"mean": mean_curve.tolist(), "std": std_curve.tolist(), "raw": matrix.tolist()}

ax.set_title("Global Validation Loss vs. Communication Rounds (PathMNIST)")
ax.set_xlabel("Communication Round")
ax.set_ylabel("Validation Loss (Cross-Entropy)")
ax.set_xticks(ROUNDS)
ax.set_ylim(0.5, 2.0)
ax.legend(loc="upper right", frameon=True, framealpha=0.9)
fig.savefig(OUTPUT_DIR / "fig2_val_loss_vs_rounds.png", dpi=300)
fig.savefig(OUTPUT_DIR / "fig2_val_loss_vs_rounds.pdf")
plt.close(fig)
source_mapping["fig2_val_loss_vs_rounds"] = "metrics.jsonl: eval/val_loss across Rounds 1-5"

# ==============================================================================
# FIGURE 3: Update Norm Comparison (Proximal Shrinkage Demonstration)
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
categories = ["Global Update", "Client 0 (40% Noise)", "Clean Clients (1–4 Mean)"]
x = np.arange(len(categories))
bar_width = 0.35

fedavg_means = []
fedavg_stds = []
fedprox_means = []
fedprox_stds = []

# Collect Round 5 norms across seeds
for cat_idx in range(3):
    avg_vals = []
    prox_vals = []
    for s in SEEDS:
        fm_avg = run_data["fedavg"][s]["rounds"][-1]
        fm_prox = run_data["fedprox"][s]["rounds"][-1]
        if cat_idx == 0:
            avg_vals.append(fm_avg["geometry/global_update_norm"])
            prox_vals.append(fm_prox["geometry/global_update_norm"])
        elif cat_idx == 1:
            avg_vals.append(fm_avg["geometry/client_update_norms"][0])
            prox_vals.append(fm_prox["geometry/client_update_norms"][0])
        else:
            avg_vals.append(np.mean(fm_avg["geometry/client_update_norms"][1:]))
            prox_vals.append(np.mean(fm_prox["geometry/client_update_norms"][1:]))
    fedavg_means.append(np.mean(avg_vals))
    fedavg_stds.append(np.std(avg_vals))
    fedprox_means.append(np.mean(prox_vals))
    fedprox_stds.append(np.std(prox_vals))

rects1 = ax.bar(x - bar_width/2, fedavg_means, bar_width, yerr=fedavg_stds, capsize=5, label="FedAvg (μ=0)", color="#1f77b4", alpha=0.85)
rects2 = ax.bar(x + bar_width/2, fedprox_means, bar_width, yerr=fedprox_stds, capsize=5, label="FedProx (μ=0.01)", color="#d62728", alpha=0.85)

# Add annotations showing percentage shrinkage
for i in range(len(categories)):
    shrinkage = (fedprox_means[i] - fedavg_means[i]) / fedavg_means[i] * 100
    ax.annotate(f"{shrinkage:.1f}%",
                xy=(x[i] + bar_width/2, fedprox_means[i] + fedprox_stds[i] + 0.15),
                ha='center', va='bottom', fontsize=10, fontweight='bold', color="#d62728")

ax.set_title("Parameter Update Displacement Norms (Round 5 Proximal Shrinkage)")
ax.set_ylabel("L2 Norm ||Δw||")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 5.0)
ax.legend(loc="upper right", frameon=True, framealpha=0.9)
fig.savefig(OUTPUT_DIR / "fig3_update_norm_shrinkage.png", dpi=300)
fig.savefig(OUTPUT_DIR / "fig3_update_norm_shrinkage.pdf")
plt.close(fig)
source_mapping["fig3_update_norm_shrinkage"] = "metrics.jsonl: geometry/global_update_norm & geometry/client_update_norms at Round 5"

# ==============================================================================
# FIGURE 4: Client Cosine Alignment with Global Update Over Rounds
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)

for algo, c_main in [("fedavg", "#1f77b4"), ("fedprox", "#d62728")]:
    c0_align = np.zeros((len(SEEDS), len(ROUNDS)))
    clean_align = np.zeros((len(SEEDS), len(ROUNDS)))
    for i, seed in enumerate(SEEDS):
        for r_idx in range(len(ROUNDS)):
            r_rec = run_data[algo][seed]["rounds"][r_idx]
            aligns = r_rec["geometry/alignment_with_global"]
            c0_align[i, r_idx] = aligns[0]
            clean_align[i, r_idx] = np.mean(aligns[1:])
    c0_mean = np.mean(c0_align, axis=0)
    clean_mean = np.mean(clean_align, axis=0)

    ax.plot(ROUNDS, clean_mean, color=c_main, linestyle="-", marker="o", linewidth=2.2, label=f"{labels[algo]} - Clean Clients (1–4)")
    ax.plot(ROUNDS, c0_mean, color=c_main, linestyle="--", marker="x", linewidth=1.8, label=f"{labels[algo]} - Noisy Client 0 (40% Noise)")

ax.set_title("Cosine Alignment with Global Update Over Rounds (cos(Δw_k, Δw_global))")
ax.set_xlabel("Communication Round")
ax.set_ylabel("Cosine Similarity to Global Direction")
ax.set_xticks(ROUNDS)
ax.set_ylim(-0.1, 0.8)
ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
fig.savefig(OUTPUT_DIR / "fig4_update_cosine_alignment.png", dpi=300)
fig.savefig(OUTPUT_DIR / "fig4_update_cosine_alignment.pdf")
plt.close(fig)
source_mapping["fig4_update_cosine_alignment"] = "metrics.jsonl: geometry/alignment_with_global across Rounds 1-5"

# ==============================================================================
# FIGURE 5: Pairwise Cosine Similarity Heatmaps (5 x 5) at Round 5
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

avg_mat = np.mean([np.array(run_data["fedavg"][s]["rounds"][-1]["geometry/pairwise_cosine_matrix"]) for s in SEEDS], axis=0)
prox_mat = np.mean([np.array(run_data["fedprox"][s]["rounds"][-1]["geometry/pairwise_cosine_matrix"]) for s in SEEDS], axis=0)

client_labels = ["C0 (Noisy)", "C1", "C2", "C3", "C4"]

im1 = ax1.imshow(avg_mat, cmap="coolwarm", vmin=-0.2, vmax=1.0)
ax1.set_title("FedAvg (μ=0) Pairwise Similarity")
ax1.set_xticks(range(5))
ax1.set_yticks(range(5))
ax1.set_xticklabels(client_labels, rotation=45, ha="right")
ax1.set_yticklabels(client_labels)
for r in range(5):
    for c in range(5):
        ax1.text(c, r, f"{avg_mat[r, c]:.2f}", ha="center", va="center", color="black" if 0.2 < avg_mat[r, c] < 0.8 else "white", fontsize=9)

im2 = ax2.imshow(prox_mat, cmap="coolwarm", vmin=-0.2, vmax=1.0)
ax2.set_title("FedProx (μ=0.01) Pairwise Similarity")
ax2.set_xticks(range(5))
ax2.set_yticks(range(5))
ax2.set_xticklabels(client_labels, rotation=45, ha="right")
ax2.set_yticklabels(client_labels)
for r in range(5):
    for c in range(5):
        ax2.text(c, r, f"{prox_mat[r, c]:.2f}", ha="center", va="center", color="black" if 0.2 < prox_mat[r, c] < 0.8 else "white", fontsize=9)

fig.colorbar(im2, ax=[ax1, ax2], orientation="vertical", fraction=0.03, pad=0.05, label="Cosine Similarity")
fig.savefig(OUTPUT_DIR / "fig5_pairwise_cosine_heatmaps.png", dpi=300)
fig.savefig(OUTPUT_DIR / "fig5_pairwise_cosine_heatmaps.pdf")
plt.close(fig)
source_mapping["fig5_pairwise_cosine_heatmaps"] = "metrics.jsonl: geometry/pairwise_cosine_matrix at Round 5 averaged across 3 seeds"

# ==============================================================================
# FIGURE 6: Per-Class Test Accuracy Across 9 PathMNIST Classes
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)

avg_class_mat = np.zeros((len(SEEDS), 9))
prox_class_mat = np.zeros((len(SEEDS), 9))

for i, s in enumerate(SEEDS):
    pc_avg = run_data["fedavg"][s]["summary"]["test_evaluation"]["per_class_accuracy"]
    pc_prox = run_data["fedprox"][s]["summary"]["test_evaluation"]["per_class_accuracy"]
    for c in range(9):
        avg_class_mat[i, c] = pc_avg[str(c)] * 100
        prox_class_mat[i, c] = pc_prox[str(c)] * 100

avg_c_mean = np.mean(avg_class_mat, axis=0)
avg_c_std = np.std(avg_class_mat, axis=0)
prox_c_mean = np.mean(prox_class_mat, axis=0)
prox_c_std = np.std(prox_class_mat, axis=0)

x = np.arange(9)
rects1 = ax.bar(x - bar_width/2, avg_c_mean, bar_width, yerr=avg_c_std, capsize=4, label="FedAvg (μ=0)", color="#1f77b4", alpha=0.85)
rects2 = ax.bar(x + bar_width/2, prox_c_mean, bar_width, yerr=prox_c_std, capsize=4, label="FedProx (μ=0.01)", color="#d62728", alpha=0.85)

ax.set_title("Per-Class Test Split Accuracy (7,180 PathMNIST Test Samples)")
ax.set_ylabel("Accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=9)
ax.set_ylim(0, 110)
ax.legend(loc="upper right", frameon=True, framealpha=0.9)
fig.savefig(OUTPUT_DIR / "fig6_per_class_test_accuracy.png", dpi=300)
fig.savefig(OUTPUT_DIR / "fig6_per_class_test_accuracy.pdf")
plt.close(fig)
source_mapping["fig6_per_class_test_accuracy"] = "summary.json: test_evaluation.per_class_accuracy averaged across 3 seeds"

# Save Source Mapping JSON
with open(OUTPUT_DIR / "figure_source_data_mapping.json", "w") as f:
    json.dump(source_mapping, f, indent=2)

print("\n" + "="*80)
print("SUCCESS: All 6 publication-grade figures exported to PNG (300 DPI) and PDF!")
print(f"Destination: {OUTPUT_DIR}")
print("="*80)
for k, v in source_mapping.items():
    print(f"  - {k}.png/.pdf -> Source: {v}")
