"""
Genuine Full-Data Pilot Execution Script.
Executes 2 full communication rounds on all 89,996 PathMNIST samples across 5 clients
for both FedAvg and FedProx (mu=0.01) to benchmark exact hardware latency, throughput,
memory utilization, and convergence dynamics.
"""

import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = r"D:\Reasearch\fl_env\Scripts\python.exe"


def run_full_pilot(algorithm, rounds=2):
    cmd = [
        PYTHON_EXE,
        "-u",
        str(PROJECT_ROOT / "run_benchmark.py"),
        "--config", "configs/base.yaml",
        "--algorithm", algorithm,
        "--rounds", str(rounds),
        "--eval_test",
    ]
    if algorithm == "fedprox":
        cmd.extend(["--mu", "0.01"])

    print(f"\n{'='*75}")
    print(f"STARTING GENUINE FULL-DATA PILOT: {algorithm.upper()} ({rounds} Full Rounds, 89,996 Samples)")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*75}\n")

    start_t = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start_t

    if result.returncode != 0:
        print(f"\n[PILOT FAILED] {algorithm.upper()} exited with error code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n[PILOT SUCCESS] {algorithm.upper()} completed in {elapsed:.1f}s ({elapsed/rounds:.1f}s/round).")
    return elapsed


if __name__ == "__main__":
    t_fedavg = run_full_pilot("fedavg", rounds=2)
    t_fedprox = run_full_pilot("fedprox", rounds=2)
    print(f"\n{'='*75}")
    print(f"FULL-DATA PILOT COMPLETE SUMMARY:")
    print(f"  FedAvg  (2 Rounds, Full Data): {t_fedavg:.1f}s ({t_fedavg/2:.1f}s/round)")
    print(f"  FedProx (2 Rounds, Full Data): {t_fedprox:.1f}s ({t_fedprox/2:.1f}s/round)")
    print(f"{'='*75}\n")
