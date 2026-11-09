"""
Reactive Optimizers Reading Summary PDF Generator.
Generates an elegant, publication-grade, exactly ONE-PAGE PDF summary
answering the three required research reading questions for the IndiaAI Fellowship.
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

PDF_PATH = PROJECT_ROOT / "docs" / "reactive_optimizers_summary.pdf"
PNG_PATH = PROJECT_ROOT / "docs" / "reactive_optimizers_summary.png"

PAGE_WIDTH = 8.5
PAGE_HEIGHT = 11.0

def pt_to_fig_y(pt, linespacing=1.20):
    return (pt / 72.0 / PAGE_HEIGHT) * linespacing

def draw_text_block(ax, x, start_y, text, width_chars, fontsize=6.6, linespacing=1.20, color="#1e293b", weight="normal", subsequent_indent=""):
    wrapped = textwrap.fill(text, width=width_chars, subsequent_indent=subsequent_indent)
    lines = wrapped.split("\n")
    line_dy = pt_to_fig_y(fontsize, linespacing)
    
    y = start_y
    for line in lines:
        ax.text(x, y, line, fontsize=fontsize, fontweight=weight, color=color, family="sans-serif", va="top")
        y -= line_dy
    return y - line_dy * 0.35

def build_pdf():
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Outer Border
    rect = Rectangle((0.03, 0.03), 0.94, 0.94, fill=False, edgecolor="#94a3b8", linewidth=0.8)
    ax.add_patch(rect)

    # Top Header Bar
    ax.fill_between([0.03, 0.97], [0.948, 0.948], [0.97, 0.97], color="#1e1b4b")
    ax.text(0.045, 0.959, "INDIAAI FELLOWSHIP RESEARCH SCREENING — LITERATURE ANALYSIS", 
            fontsize=8.5, fontweight="bold", color="#ffffff", family="sans-serif", va="center")
    ax.text(0.955, 0.959, "RESEARCH READING SUMMARY", 
            fontsize=8.0, fontweight="bold", color="#c7d2fe", family="sans-serif", ha="right", va="center")

    # Title & Subtitle
    ax.text(0.045, 0.928, "Reactive Optimizers: In-the-Loop Agentic Optimization for Neural Training", 
            fontsize=12.2, fontweight="bold", color="#0f172a", family="sans-serif")
    ax.text(0.045, 0.910, "Critical Review of Mechanism, Technical Limitations, and Federated Learning Applications", 
            fontsize=8.8, fontweight="bold", color="#4338ca", family="sans-serif")

    # Citation & Attribution Box
    ax.fill_between([0.045, 0.955], [0.876, 0.876], [0.902, 0.902], color="#f5f3ff", edgecolor="#ddd6fe", linewidth=0.6)
    cite_text = (
        "Paper: Reactive Optimizers  |  Authors: Xingjian Bai, Jia-Hao Liang, Jayden Teoh, Vincent Sitzmann (MIT, 2026)\n"
        "Reference: arXiv:2610.33826 [cs.LG]  |  Framework: OPAL (OPtimizer with an Agent in the Loop)  |  Code: github.com/xingjian-bai/opal"
    )
    ax.text(0.055, 0.889, cite_text, fontsize=6.8, color="#3730a3", family="sans-serif", va="center", linespacing=1.25)

    def draw_section_hdr(y, title, num):
        hdr_h = 0.015
        ax.fill_between([0.045, 0.955], [y - hdr_h, y - hdr_h], [y, y], color="#ede9fe")
        ax.text(0.055, y - hdr_h * 0.5, f"Section {num}: {title}", fontsize=8.0, fontweight="bold", color="#1e1b4b", family="sans-serif", va="center")
        ax.plot([0.045, 0.955], [y - hdr_h, y - hdr_h], color="#6366f1", linewidth=1.3)
        return y - hdr_h - 0.006

    cur_y = 0.865

    # -------------------------------------------------------------------------
    # PART A: What does Reactive Optimizers do?
    # -------------------------------------------------------------------------
    cur_y = draw_section_hdr(cur_y, "Core Mechanism and Operating Principles of Reactive Optimizers (OPAL)", "A")
    sec_a_items = [
        "• Foundational Premise: Traditional deep learning optimization relies on fixed mathematical update rules (e.g., SGD, AdamW, Muon) and predetermined scalar hyperparameter schedules configured before training commences. While effective on benchmark regimes, static recipes fail to adapt when training enters unexpected loss landscapes, encounters noisy data, or develops layer-specific divergence.",
        "• Language Model Agent in the Loop: The OPAL (OPtimizer with an Agent in the Loop) framework integrates a pretrained Large Language Model (coding agent) directly into the active training loop. Rather than restricting adaptation to scalar hyperparameter tuning (e.g., adjusting learning rate), OPAL positions the agent as an active optimizer supervisor that monitors live telemetry and programmatically rewrites the optimizer's source code during a single training run.",
        "• Telemetry Observation: At designated intervals during training, the agent inspects multimodal training dynamics, including training loss trajectories, validation performance, gradient $L_2$ norms, parameter update magnitudes, directional cosine alignment between layers, parameter group variance, and device memory utilization.",
        "• On-the-Fly Code Synthesis: Based on monitored telemetry, the agent synthesizes and hot-swaps executable Python code implementing customized PyTorch optimizer steps. It dynamically modifies weight update formulations, second-order preconditioning, momentum factors, numerical precision, and weight decay tailored specifically to individual parameter groups (e.g., attention projections vs. feed-forward layers).",
        "• Emergent Problem-Solving: Without task-specific meta-training or cross-run memory transfer, OPAL exhibits autonomous problem-solving capabilities: inventing numerical spike guards to arrest sudden loss explosions, altering schedules to escape saddle points or mitigate overfitting, dynamically transitioning specific matrix layers from AdamW to Muon mid-run, and adjusting update geometry to withstand synthetic data corruption."
    ]
    for item in sec_a_items:
        cur_y = draw_text_block(ax, 0.045, cur_y, item, width_chars=124, fontsize=6.4, linespacing=1.18, subsequent_indent="  ")

    # -------------------------------------------------------------------------
    # PART B: One Technical Limitation
    # -------------------------------------------------------------------------
    cur_y = draw_section_hdr(cur_y - 0.003, "Critical Technical Limitation: Agent Latency, Staleness, and Safety Risks", "B")
    sec_b_items = [
        "• Computational Latency & Pipeline Stalls: Invoking a frontier LLM agent to analyze telemetry, reason over trajectories, and synthesize executable code incurs substantial wall-clock latency (typically 2 to 10 seconds per invocation). In high-performance GPU environments where forward-backward passes execute in tens of milliseconds, synchronous agent calls introduce severe throughput stalls, drastically degrading accelerator utilization.",
        "• Asynchronous Staleness vs. Feedback Delay: Shifting agent execution to asynchronous background threads mitigates hardware stalls but introduces an inevitable feedback lag: by the time the agent completes reasoning and validates new code, the active model parameters have progressed through hundreds of optimizer steps, potentially rendering the synthesized update logic obsolete or counterproductive.",
        "• Code Execution Risks & Numerical Instability: Dynamically executing LLM-generated code inside the optimization loop poses significant runtime risks. The agent can inadvertently introduce syntax errors, shape mismatches, memory leaks in optimizer state tensors, or mathematically unstable operations that induce catastrophic NaN or gradient explosion, requiring complex sandboxing and conservative fallback rollback mechanisms."
    ]
    for item in sec_b_items:
        cur_y = draw_text_block(ax, 0.045, cur_y, item, width_chars=124, fontsize=6.4, linespacing=1.18, subsequent_indent="  ")

    # -------------------------------------------------------------------------
    # PART C: Application to Federated Learning
    # -------------------------------------------------------------------------
    cur_y = draw_section_hdr(cur_y - 0.003, "Technically Plausible Application to Federated Learning: Dynamic Proximal Modulation", "C")
    sec_c_items = [
        "• Proposal Title: Server-Side Agentic Proximal Modulation (APM) for Heterogeneous Federated Learning.",
        "• Problem Context: In Federated Learning under non-IID class skew (e.g. Dirichlet alpha = 0.3) and client label noise (e.g. 40% corruption on Client 0, as characterized in our PathMNIST benchmark), static hyperparameters in FedAvg and FedProx (fixed mu = 0.01) incur a fundamental trade-off: unconstrained FedAvg allows corrupted updates to distort global weights, while static FedProx imposes an optimization lag that suppresses clean-client convergence.",
        "• Driving Telemetry Signal: At the end of each communication round, the central server receives standard non-private update telemetry from participating clients: (1) client parameter update norms ||w_k^t - w^t||_2, (2) pairwise directional cosine similarity cos(w_k^t - w^t, Delta w_global^t), and (3) client validation loss trajectories. In our benchmark, this telemetry revealed an unmistakable signature for Client 0: an update norm of 3.64 (vs. 2.22 for clean clients) and severe gradient misalignment.",
        "• Adapted Component: A server-side Reactive Optimizer agent analyzes this geometric telemetry and dynamically synthesizes personalized proximal weights mu_k^{t+1} and local learning rates eta_k^{t+1} for each client. For detected noisy or divergent clients, the agent dynamically elevates mu_k to aggressively dampen destructive updates (-36% dampening); for cohesive clean clients, the agent decays mu_k toward zero, eliminating optimization drag and capturing FedAvg's early convergence speed.",
        "• Concrete Evaluation Protocol: Evaluate APM against standard FedAvg, static FedProx (mu in {0.001, 0.01, 0.1}), and SCAFFOLD on the 9-class PathMNIST benchmark under varying Dirichlet skew (alpha in {0.1, 0.3, 1.0}) and label noise levels (0% to 50%). Measure communication rounds to target accuracy (75%), global test accuracy, Macro-F1, and total wall-clock compute time.",
        "• Critical Trade-off & Security Risk: Differential Privacy and Remote Code Execution: Transmitting fine-grained update geometry metrics could leak local client distributional properties to gradient reconstruction attacks. Furthermore, distributing dynamically generated optimizer logic to untrusted edge devices creates severe remote execution vulnerabilities unless execution is strictly bounded to a verified mathematical DSL."
    ]
    for item in sec_c_items:
        cur_y = draw_text_block(ax, 0.045, cur_y, item, width_chars=124, fontsize=6.3, linespacing=1.17, subsequent_indent="  ")

    # Footer
    ax.plot([0.045, 0.955], [0.050, 0.050], color="#cbd5e1", linewidth=0.8)
    ax.text(0.045, 0.038, "PHOENIX FEDERATED LEARNING RESEARCH   |   INDIAAI FELLOWSHIP SCREENING SUBMISSION", 
            fontsize=6.7, color="#64748b", family="sans-serif")
    ax.text(0.955, 0.038, "Page 1 of 1 (Complete Single-Page Reading Summary)", 
            fontsize=6.7, fontweight="bold", color="#0f172a", family="sans-serif", ha="right")

    with PdfPages(PDF_PATH) as pdf:
        pdf.savefig(fig)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)
    print(f"Generated single-page Reactive Optimizers summary PDF at: {PDF_PATH}")
    print(f"Generated preview PNG at: {PNG_PATH}")

if __name__ == "__main__":
    build_pdf()
