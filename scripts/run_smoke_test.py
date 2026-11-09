"""
Rapid Smoke Test Runner.
Executes a 1-round verification run on PathMNIST to ensure end-to-end framework health.
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = sys.executable


def run_smoke_test(algorithm="fedavg"):
    cmd = [
        PYTHON_EXE,
        str(PROJECT_ROOT / "run_benchmark.py"),
        "--config", "configs/base.yaml",
        "--algorithm", algorithm,
        "--smoke_test",
        "--eval_test",
    ]
    if algorithm == "fedprox":
        cmd.extend(["--mu", "0.01"])

    print(f"\n{'='*70}")
    print(f"Running End-to-End Smoke Test for {algorithm.upper()}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\n[FAILED] Smoke test for {algorithm.upper()} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[PASSED] Smoke test for {algorithm.upper()} completed successfully.")


if __name__ == "__main__":
    algo = sys.argv[1] if len(sys.argv) > 1 else "fedavg"
    run_smoke_test(algo)
