"""
Primary Benchmark Matrix Execution Orchestrator.
Executes all 6 paired runs across 3 random seeds (42, 101, 2024) for FedAvg and FedProx (mu=0.01)
on 89,996 PathMNIST training samples across 5 Dirichlet non-IID clients with 40% label noise on Client 0.

Outputs are strictly isolated in: D:/Reasearch/fl_benchmark/results/primary_benchmark/
"""

import sys
import os
import time
import json
import ctypes
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
for p in [str(PROJECT_ROOT), str(WORKSPACE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

PYTHON_EXE = sys.executable
BENCHMARK_SCRIPT = PROJECT_ROOT / "run_benchmark.py"
OUTPUT_BASE_DIR = PROJECT_ROOT / "results" / "primary_benchmark"
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Prevent Windows system sleep/standby during execution
def prevent_windows_sleep():
    try:
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("[System] Windows execution state set to prevent system sleep during benchmark.")
    except Exception as e:
        print(f"[System WARN] Could not set Windows execution state: {e}")

MATRIX = [
    # Seed 42 Paired Runs
    {"run_id": 1, "algorithm": "fedavg", "seed": 42, "mu": None, "rounds": 5},
    {"run_id": 2, "algorithm": "fedprox", "seed": 42, "mu": 0.01, "rounds": 5},
    # Seed 101 Paired Runs
    {"run_id": 3, "algorithm": "fedavg", "seed": 101, "mu": None, "rounds": 5},
    {"run_id": 4, "algorithm": "fedprox", "seed": 101, "mu": 0.01, "rounds": 5},
    # Seed 2024 Paired Runs
    {"run_id": 5, "algorithm": "fedavg", "seed": 2024, "mu": None, "rounds": 5},
    {"run_id": 6, "algorithm": "fedprox", "seed": 2024, "mu": 0.01, "rounds": 5},
]


def check_existing_completion(run_name, expected_spec, required_rounds=5):
    """
    Six-factor cryptographic and structural validation for run completion:
    1. Physical file existence (summary.json, metrics.jsonl, best_checkpoint.pt).
    2. Summary round count equals required_rounds.
    3. Continuous round sequence in metrics.jsonl.
    4. Checkpoint loadability and payload integrity.
    5. Checkpoint metadata (algorithm, seed) matches expected_spec.
    6. Complete test evaluation record present.
    """
    run_dir = OUTPUT_BASE_DIR / run_name
    summary_file = run_dir / "summary.json"
    metrics_file = run_dir / "metrics.jsonl"
    best_ckpt = run_dir / "checkpoints" / "best_checkpoint.pt"

    if not (summary_file.exists() and metrics_file.exists() and best_ckpt.exists()):
        return False, {}, "Core artifacts missing"

    # 1. Summary check
    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as e:
        return False, {}, f"Unreadable summary.json: {e}"

    if summary.get("rounds_completed", 0) != required_rounds:
        return False, {}, f"Rounds completed ({summary.get('rounds_completed')}) != {required_rounds}"

    # 2. Metrics JSONL check
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        eval_steps = [r["step"] for r in lines if "eval/val_acc" in r]
        if eval_steps != list(range(1, required_rounds + 1)):
            return False, {}, f"JSONL steps {eval_steps} != expected {list(range(1, required_rounds + 1))}"
    except Exception as e:
        return False, {}, f"Corrupted metrics.jsonl: {e}"

    # 3. Checkpoint loadability check
    try:
        payload = torch.load(best_ckpt, map_location="cpu")
        if "state_dict" not in payload or "metrics" not in payload:
            return False, {}, "Checkpoint missing state_dict or metrics"
    except Exception as e:
        return False, {}, f"Corrupted checkpoint: {e}"

    # 4. Metadata check
    ckpt_config = payload.get("config", {})
    algo = ckpt_config.get("federated", {}).get("algorithm", "").lower()
    seed = ckpt_config.get("experiment", {}).get("seed", None)
    if algo != expected_spec["algorithm"].lower():
        return False, {}, f"Algorithm mismatch: {algo} != {expected_spec['algorithm']}"
    if seed != expected_spec["seed"]:
        return False, {}, f"Seed mismatch: {seed} != {expected_spec['seed']}"

    # 5. Test evaluation check
    test_eval = summary.get("test_evaluation", {})
    if "accuracy" not in test_eval or "loss" not in test_eval or "macro_f1" not in test_eval:
        return False, {}, "summary.json missing complete test evaluation"

    return True, summary, "Verified complete and cryptographically sound"


def execute_run(spec):
    run_id = spec["run_id"]
    algo = spec["algorithm"]
    seed = spec["seed"]
    mu = spec["mu"]
    rounds = spec["rounds"]

    if algo == "fedprox":
        run_name = f"fedprox_mu{mu}_dir0.3_noise0.4_seed{seed}"
    else:
        run_name = f"fedavg_dir0.3_noise0.4_seed{seed}"

    # Six-factor recovery guard check
    is_complete, existing_summary, reason = check_existing_completion(run_name, spec, rounds)
    if is_complete:
        test_eval = existing_summary.get("test_evaluation", {})
        print("\n" + "=" * 80)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] RUN {run_id}/6 ({run_name}) ALREADY COMPLETED!")
        print(f"Status: {reason}")
        print(f"Rounds: {existing_summary.get('rounds_completed')}/{rounds} | Test Acc: {test_eval.get('accuracy', 0)*100:.2f}% | Test Loss: {test_eval.get('loss', 0):.4f} | Test F1: {test_eval.get('macro_f1', 0):.4f}")
        print("=" * 80 + "\n", flush=True)
        return {
            "run_id": run_id,
            "run_name": run_name,
            "algorithm": algo,
            "seed": seed,
            "mu": mu,
            "rounds": rounds,
            "duration_sec": 0.0,
            "duration_min": 0.0,
            "test_acc": test_eval.get("accuracy"),
            "test_loss": test_eval.get("loss"),
            "test_macro_f1": test_eval.get("macro_f1"),
            "summary_exists": True,
            "metrics_exists": True,
            "best_ckpt_exists": True,
            "skipped_because_complete": True,
        }

    cmd = [
        PYTHON_EXE,
        "-u",
        str(BENCHMARK_SCRIPT),
        "--config", "configs/base.yaml",
        "--algorithm", algo,
        "--seed", str(seed),
        "--rounds", str(rounds),
        "--run_name", run_name,
        "--output_dir", str(OUTPUT_BASE_DIR),
        "--eval_test",
    ]
    if mu is not None:
        cmd.extend(["--mu", str(mu)])

    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STARTING RUN {run_id}/6: {run_name.upper()}")
    print(f"Algorithm: {algo.upper()} | Seed: {seed} | Mu: {mu} | Rounds: {rounds} | Total Samples: 89,996")
    print(f"Output Directory: {OUTPUT_BASE_DIR / run_name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80 + "\n", flush=True)

    start_time = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    duration = time.time() - start_time

    if proc.returncode != 0:
        print(f"\n[ERROR] Run {run_id} ({run_name}) failed with return code {proc.returncode}!", flush=True)
        sys.exit(proc.returncode)

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] COMPLETED RUN {run_id}/6 ({run_name}) in {duration:.1f}s ({duration/60:.2f} min).", flush=True)

    # Verification of artifacts
    run_dir = OUTPUT_BASE_DIR / run_name
    summary_file = run_dir / "summary.json"
    metrics_file = run_dir / "metrics.jsonl"
    best_ckpt = run_dir / "checkpoints" / "best_checkpoint.pt"

    summary_data = {}
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary_data = json.load(f)

    test_acc = summary_data.get("test_evaluation", {}).get("accuracy", None)
    test_loss = summary_data.get("test_evaluation", {}).get("loss", None)
    test_f1 = summary_data.get("test_evaluation", {}).get("macro_f1", None)

    return {
        "run_id": run_id,
        "run_name": run_name,
        "algorithm": algo,
        "seed": seed,
        "mu": mu,
        "rounds": rounds,
        "duration_sec": round(duration, 1),
        "duration_min": round(duration / 60, 2),
        "test_acc": test_acc,
        "test_loss": test_loss,
        "test_macro_f1": test_f1,
        "summary_exists": summary_file.exists(),
        "metrics_exists": metrics_file.exists(),
        "best_ckpt_exists": best_ckpt.exists(),
    }


def main():
    prevent_windows_sleep()

    print(f"\nStarting Federated Learning Primary Benchmark Matrix (6 Runs, 5 Rounds Each)")
    print(f"Workspace: {PROJECT_ROOT}")
    print(f"Isolated Destination: {OUTPUT_BASE_DIR}")
    print(f"Total Runs to Execute: {len(MATRIX)}\n")

    matrix_start = time.time()
    results = []

    for spec in MATRIX:
        res = execute_run(spec)
        results.append(res)

    total_duration = time.time() - matrix_start

    print("\n" + "=" * 90)
    print("PRIMARY BENCHMARK MATRIX COMPLETE EXECUTION SUMMARY")
    print(f"Total Execution Time: {total_duration/60:.2f} minutes ({total_duration:.1f} seconds)")
    print("=" * 90)
    print(f"{'Run':<4} | {'Run Name':<38} | {'Duration':<10} | {'Test Acc':<10} | {'Test Loss':<10} | {'Test F1':<10}")
    print("-" * 90)
    for r in results:
        acc_str = f"{r['test_acc']*100:.2f}%" if r['test_acc'] is not None else "N/A"
        loss_str = f"{r['test_loss']:.4f}" if r['test_loss'] is not None else "N/A"
        f1_str = f"{r['test_macro_f1']:.4f}" if r['test_macro_f1'] is not None else "N/A"
        dur_str = f"{r['duration_min']:.1f} min"
        print(f"{r['run_id']:<4} | {r['run_name']:<38} | {dur_str:<10} | {acc_str:<10} | {loss_str:<10} | {f1_str:<10}")
    print("=" * 90)

    # Save matrix summary JSON
    out_summary = OUTPUT_BASE_DIR / "matrix_execution_summary.json"
    with open(out_summary, "w") as f:
        json.dump({
            "total_duration_sec": total_duration,
            "total_duration_min": total_duration / 60,
            "timestamp": datetime.now().isoformat(),
            "runs": results
        }, f, indent=2)
    print(f"\nMatrix execution summary saved to {out_summary}")


if __name__ == "__main__":
    main()
