"""
Experiment Execution Script.
CLI entry point to launch FedAvg or FedProx experiments on PathMNIST.
Delegates to run_benchmark.py with calibrated hyperparameters.
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    cmd = [sys.executable, str(PROJECT_ROOT / "run_benchmark.py")] + sys.argv[1:]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
