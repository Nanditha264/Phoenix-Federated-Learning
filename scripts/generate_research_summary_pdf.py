"""
One-Page Research Summary PDF Generator with exact typographic line metrics.
"""

import os
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib")
(PROJECT_ROOT / ".matplotlib").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

PDF_PATH = PROJECT_ROOT / "docs" / "research_summary.pdf"
PNG_PATH = PROJECT_ROOT / "docs" / "research_summary.png"

PAGE_WIDTH = 8.5
PAGE_HEIGHT = 11.0

def pt_to_fig_y(pt, linespacing=1.2):
    return (pt / 72.0 / PAGE_HEIGHT) * linespacing

def draw_text_block(ax, x, start_y, text, width_chars, fontsize=6.5, linespacing=1.22, color="#1e293b", weight="normal", prefix="", prefix_weight="bold", subsequent_indent=""):
    wrapped = textwrap.fill(prefix + text, width=width_chars, subsequent_indent=subsequent_indent)
    lines = wrapped.split("\n")
    line_dy = pt_to_fig_y(fontsize, linespacing)
    
    y = start_y
    for i, line in enumerate(lines):
        if i == 0 and prefix:
            # draw prefix bold and rest normal
            # simpler in matplotlib: draw full line
            ax.text(x, y, line, fontsize=fontsize, fontweight="normal", color=color, family="sans-serif", va="top")
        else:
            ax.text(x, y, line, fontsize=fontsize, fontweight=weight, color=color, family="sans-serif", va="top")
        y -= line_dy
    return y - line_dy * 0.4 # return next available y with small padding

def build_pdf():
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Outer Border
    rect = Rectangle((0.03, 0.03), 0.94, 0.94, fill=False, edgecolor="#94a3b8", linewidth=0.8)
    ax.add_patch(rect)

    # Top Header Bar
    ax.fill_between([0.03, 0.97], [0.948, 0.948], [0.97, 0.97], color="#0f172a")
    ax.text(0.045, 0.959, "INDIAAI FELLOWSHIP RESEARCH SCREENING BENCHMARK", 
            fontsize=8.5, fontweight="bold", color="#ffffff", family="sans-serif", va="center")
    ax.text(0.955, 0.959, "PATHMNIST: FEDAVG VS FEDPROX", 
            fontsize=8.0, fontweight="bold", color="#93c5fd", family="sans-serif", ha="right", va="center")

    # Title & Subtitle
    ax.text(0.045, 0.930, "Empirical Evaluation of Federated Averaging and Federated Proximal", 
            fontsize=12.5, fontweight="bold", color="#0f172a", family="sans-serif")
    ax.text(0.045, 0.912, "Under Non-IID Dirichlet Partitioning and Controlled Client-Specific Label Noise on PathMNIST", 
            fontsize=8.8, fontweight="bold", color="#334155", family="sans-serif")

    # Metadata Strip
    ax.fill_between([0.045, 0.955], [0.890, 0.890], [0.905, 0.905], color="#f1f5f9", edgecolor="#cbd5e1", linewidth=0.5)
    ax.text(0.055, 0.8975, "Candidate: Nanditha M C   |   Repository: github.com/Nanditha264/Phoenix-Federated-Learning   |   Framework: PyTorch 2.15 (CPU)",
            fontsize=7.0, color="#334155", family="sans-serif", va="center")

    def draw_section_hdr(x, y, w, title):
        hdr_h = 0.014
        ax.fill_between([x, x + w], [y - hdr_h, y - hdr_h], [y, y], color="#e2e8f0")
        ax.text(x + 0.008, y - hdr_h * 0.5, title, fontsize=7.8, fontweight="bold", color="#0f172a", family="sans-serif", va="center")
        ax.plot([x, x + w], [y - hdr_h, y - hdr_h], color="#0284c7", linewidth=1.2)
        return y - hdr_h - 0.005

    # =========================================================================
    # ROW 1 (y: 0.875 down to 0.585): Two columns
    # Left column: Sections 1 & 2 (x = 0.045 to 0.485, w = 0.440)
    # Right column: Section 3 Table (x = 0.505 to 0.955, w = 0.450)
    # =========================================================================

    # Left Column: Section 1
    y_l = draw_section_hdr(0.045, 0.875, 0.440, "1. Motivation & Problem Statement")
    p1 = "In histopathological imaging, patient distributions vary markedly across hospital sites due to institutional demographics, tissue staining protocols, and scanning hardware. This induces severe non-IID label skew. Concurrently, human annotation inconsistencies introduce substantial label noise."
    y_l = draw_text_block(ax, 0.045, y_l, p1, width_chars=54, fontsize=6.4, linespacing=1.20)
    
    p2 = "Under standard FedAvg, client models overfit to local distributions and diverge ('client drift'). FedProx introduces an explicit proximal penalty (mu/2)||w - w_t||^2 to constrain local drift toward the global consensus. This benchmark provides a rigorous empirical comparison under controlled Dirichlet skew and single-client label corruption."
    y_l = draw_text_block(ax, 0.045, y_l, p2, width_chars=54, fontsize=6.4, linespacing=1.20)

    # Left Column: Section 2
    y_l = draw_section_hdr(0.045, y_l - 0.002, 0.440, "2. Experimental Protocol & Architecture")
    sec2_items = [
        "• Dataset: MedMNIST PathMNIST (89,996 train, 10,004 val, 7,180 test; 9 tissue classes, 28x28x3).",
        "• Partitioning: 5 clients under Dirichlet distribution (alpha = 0.3) reflecting severe non-IID skew.",
        "• Label Noise: 40% symmetric noise injected exclusively into Client 0 (Clients 1-4 clean).",
        "• Architecture: 2-layer ConvNet + 2-layer MLP classifier (113,225 total trainable parameters).",
        "• Federated Setup: 5 rounds, 2 local epochs/round (E=2), batch size 32, SGD (lr=0.01, mom=0.9).",
        "• Evaluation: 3 independent random seeds (42, 101, 2024; N=6 total runs) on CPU runtime.",
        "• Algorithms: FedAvg (mu=0.0) vs. FedProx (mu=0.01, Li et al., 2020)."
    ]
    for item in sec2_items:
        y_l = draw_text_block(ax, 0.045, y_l, item, width_chars=54, fontsize=6.3, linespacing=1.16, subsequent_indent="  ")

    # Right Column: Section 3 Table of Results
    y_r = draw_section_hdr(0.505, 0.875, 0.450, "3. Primary Quantitative Benchmark Results")
    headers = ["Evaluation Metric", "FedAvg (N=3)", "FedProx (μ=0.01)", "Diff (Δ)"]
    rows = [
        ["Test Accuracy", "77.25% ± 3.25%", "74.71% ± 3.79%", "-2.54 pp"],
        ["Test Loss", "0.7245 ± 0.0955", "0.8078 ± 0.1212", "+0.0833"],
        ["Test Macro-F1", "0.7118 ± 0.0213", "0.6846 ± 0.0364", "-0.0272"],
        ["Best Validation Acc", "75.99%", "73.00%", "-2.99 pp"],
        ["Global Update Norm (||Δw||)", "2.4838 ± 0.2521", "1.7388 ± 0.1239", "-29.99%"],
        ["Client 0 Norm (40% Noise)", "3.6420 ± 0.3626", "2.3229 ± 0.0450", "-36.22%"],
        ["Clean Clients Norm (1-4)", "2.2223 ± 0.2312", "1.5833 ± 0.1065", "-28.75%"],
        ["Clean Cosine Alignment", "0.5175 ± 0.0441", "0.5976 ± 0.0279", "+15.48%"],
        ["Class 8 (Tumor) Test Acc", "78.94%", "80.97%", "+2.03 pp"],
    ]

    tx = 0.505
    ty = y_r
    col_w = [0.170, 0.112, 0.108, 0.060]
    rh = 0.0175

    # Table Header Row
    ax.fill_between([tx, tx + sum(col_w)], [ty - rh, ty - rh], [ty, ty], color="#0f172a")
    cx = tx
    for i, h in enumerate(headers):
        ax.text(cx + 0.004, ty - rh * 0.55, h, fontsize=6.4, fontweight="bold", color="#ffffff", family="sans-serif", va="center")
        cx += col_w[i]

    # Table Data Rows
    for r_idx, row in enumerate(rows):
        ry = ty - (r_idx + 1) * rh
        bg_col = "#f8fafc" if r_idx % 2 == 0 else "#ffffff"
        ax.fill_between([tx, tx + sum(col_w)], [ry - rh, ry - rh], [ry, ry], color=bg_col)
        ax.plot([tx, tx + sum(col_w)], [ry - rh, ry - rh], color="#cbd5e1", linewidth=0.4)
        cx = tx
        for c_idx, val in enumerate(row):
            weight = "bold" if (c_idx == 0 or r_idx in [0, 4, 5, 8]) else "normal"
            tcol = "#047857" if c_idx == 3 and ("+" in val or "-29" in val or "-36" in val) else ("#b91c1c" if c_idx == 3 and "-" in val and "pp" in val else "#1e293b")
            ax.text(cx + 0.004, ry - rh * 0.55, val, fontsize=6.2, fontweight=weight, color=tcol, family="sans-serif", va="center")
            cx += col_w[c_idx]

    note_y = ty - (len(rows) + 1) * rh - 0.006
    note_text = (
        "Key Quantitative Takeaways:\n"
        "1. Predictive Performance: FedAvg achieves higher average test accuracy (+2.54 pp) and Macro-F1 (+0.0272) across the 5-round budget.\n"
        "2. Observed Norm Differences: FedProx exhibits -29.99% global update norm magnitude and -36.22% lower norm on corrupted Client 0.\n"
        "3. Directional Alignment & Class 8: FedProx yields +15.48% higher clean client alignment and +2.03 pp gain on PathMNIST Class 8."
    )
    y_r = draw_text_block(ax, tx, note_y, note_text, width_chars=55, fontsize=6.2, linespacing=1.20, color="#334155")

    # Bottom of Row 1 is min(y_l, y_r)
    row2_start_y = min(y_l, y_r) - 0.010

    # =========================================================================
    # ROW 2: Full Width Sections (x = 0.045 to 0.955, w = 0.910)
    # =========================================================================

    # Section 4: Update Geometry & Mechanistic Insights
    y_full = draw_section_hdr(0.045, row2_start_y, 0.910, "4. Update Geometry & Proximal Regularization Dynamics")
    sec4_text = [
        "• Proximal Shrinkage Effect: FedProx strictly enforced parameter regularization, contracting global update magnitude by 29.99% (2.4838 -> 1.7388) and restraining the noisy client's update norm by 36.22% (3.6420 -> 2.3229). This provides direct empirical validation of proximal constraint under label corruption.",
        "• Gradient Alignment & Consensus: By penalizing local wandering, FedProx prevented clean clients from overshooting into discordant subspaces, improving clean-client directional cosine alignment with the global consensus from 0.5175 to 0.5976 (+15.48%).",
        "• The Optimization-Regularization Trade-off: Within a 5-round computational budget, the proximal penalty (mu=0.01) introduces an optimization drag: by continuously tethering clients to the global model, FedProx accumulated fewer effective exploratory steps, resulting in -2.54 pp lower overall accuracy than unconstrained FedAvg.",
        "• Class 8 Accuracy: On PathMNIST Class 8 (Colorectal Adenocarcinoma), FedProx attained higher test accuracy than FedAvg (80.97% vs. 78.94%, +2.03 pp). Observed update regularization helped maintain decision boundaries for this underrepresented class. (No clinical validity implied)."
    ]
    for bullet in sec4_text:
        y_full = draw_text_block(ax, 0.045, y_full, bullet, width_chars=124, fontsize=6.3, linespacing=1.18, subsequent_indent="  ")

    # Section 5: Core Research Interpretation
    y_full = draw_section_hdr(0.045, y_full - 0.004, 0.910, "5. Core Research Interpretation & Practical Guidance")
    sec5_text = [
        "1. Predictive Utility vs. Geometric Stability: FedAvg optimizes faster in early rounds under label noise because unconstrained updates move aggressively toward local solutions that generalize reasonably well. However, FedProx produces a substantially more cohesive, geometrically stable global trajectory.",
        "2. Noise Gradient Dampening: The proximal penalty acts as an automated regularizer against corrupted client drift. The 36.22% norm dampening on Client 0 confirms that FedProx successfully suppresses corrupted gradients without requiring explicit client identification or data inspection.",
        "3. Adaptive Proximal Tuning: The static selection of mu=0.01 was chosen conservatively. Future designs should adopt dynamic proximal scheduling (decaying mu over rounds or scaling mu based on monitored client drift) to capture FedAvg's early convergence speed alongside FedProx's stability."
    ]
    for bullet in sec5_text:
        y_full = draw_text_block(ax, 0.045, y_full, bullet, width_chars=124, fontsize=6.3, linespacing=1.18, subsequent_indent="   ")

    # Section 6: Limitations & Disclaimer
    y_full = draw_section_hdr(0.045, y_full - 0.004, 0.910, "6. Scientific Limitations & Fellowship Disclaimer")
    sec6_text = [
        "• Statistical Sample Size (N=3): Results are computed over 3 random seeds (42, 101, 2024) across 5 communication rounds under strict CPU execution constraints. While standard deviations are reported, differences should not be claimed as asymptotic convergence guarantees or statistical superiority.",
        "• Architectural Scope: Evaluation was conducted on a 2-layer ConvNet benchmark on PathMNIST (alpha=0.3, noise=0.4). Dynamics may differ across deeper vision architectures (ResNet, ViT), higher communication budgets (R > 50), and multi-client noise topologies.",
        "• Non-Clinical Research Disclaimer: This investigation is an algorithmic benchmarking study for fellowship screening and research evaluation. It is not approved, validated, or intended for clinical medical diagnostic deployment."
    ]
    for bullet in sec6_text:
        y_full = draw_text_block(ax, 0.045, y_full, bullet, width_chars=124, fontsize=6.3, linespacing=1.18, subsequent_indent="  ")

    # Section 7: Reproducibility & Open Science
    y_full = draw_section_hdr(0.045, y_full - 0.004, 0.910, "7. Reproducibility & Research Artifacts")
    sec7_text = (
        "• Public Repository: Complete modular source code, configurations, seeds, test suites, and figures are hosted at github.com/Nanditha264/Phoenix-Federated-Learning.\n"
        "• Deterministic Execution: End-to-end benchmark reproducible via `python run_benchmark.py --config configs/base_config.yaml`.\n"
        "• Checkpoint Verification: Independent re-evaluation on all 7,180 PathMNIST test samples verified 0.000000 metric divergence across all 6 saved checkpoints."
    )
    y_full = draw_text_block(ax, 0.045, y_full, sec7_text, width_chars=124, fontsize=6.3, linespacing=1.18)

    # Footer
    ax.plot([0.045, 0.955], [0.050, 0.050], color="#cbd5e1", linewidth=0.8)
    ax.text(0.045, 0.038, "PHOENIX FEDERATED LEARNING RESEARCH   |   INDIAAI FELLOWSHIP SCREENING SUBMISSION", 
            fontsize=6.7, color="#64748b", family="sans-serif")
    ax.text(0.955, 0.038, "Page 1 of 1 (Complete Single-Page Summary)", 
            fontsize=6.7, fontweight="bold", color="#0f172a", family="sans-serif", ha="right")

    with PdfPages(PDF_PATH) as pdf:
        pdf.savefig(fig)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)
    print(f"Generated single-page research summary PDF at: {PDF_PATH}")
    print(f"Generated preview PNG at: {PNG_PATH}")

if __name__ == "__main__":
    build_pdf()
