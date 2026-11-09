"""
Weights & Biases Tracking Engine.
Safely integrates experiment logging with zero credential exposure and robust offline fallback.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import wandb


class WandbLogger:
    """
    Manages W&B experiment runs, logging metrics locally and safely handling offline mode.
    """

    def __init__(
        self,
        project: str = "fedavg-vs-fedprox-pathmnist",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = "offline",
        output_dir: str = "D:/Reasearch/fl_benchmark/results",
        enabled: bool = True,
    ):
        self.enabled = enabled and (mode != "disabled")
        self.mode = mode
        self.project = project
        self.run_name = run_name
        self.run = None

        if not self.enabled:
            print(f"[W&B Logger] W&B tracking is disabled (mode='{self.mode}'). Local JSONL telemetry active.")
            return

        run_dir = Path(output_dir) / (run_name or "default_run") / "wandb"
        run_dir.mkdir(parents=True, exist_ok=True)

        os.environ["WANDB_MODE"] = self.mode
        os.environ["WANDB_DIR"] = str(run_dir)
        os.environ["WANDB_CONFIG_DIR"] = str(run_dir / "config")
        os.environ["WANDB_CACHE_DIR"] = str(run_dir / "cache")
        os.environ["WANDB_DATA_DIR"] = str(run_dir / "data")

        try:
            self.run = wandb.init(
                project=self.project,
                name=self.run_name,
                config=config,
                dir=str(run_dir),
                reinit=True,
            )
            print(f"[W&B Logger] Initialized run '{self.run_name}' in mode='{self.mode}'. Run ID: {self.run.id}")
        except Exception as e:
            print(f"[WARN] Failed to initialize W&B ({e}). Disabling W&B tracking.")
            self.enabled = False
            self.run = None

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics dictionary to active W&B run.
        """
        if not self.enabled or self.run is None:
            return

        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            print(f"[WARN] Failed to log metrics to W&B: {e}")

    def finish(self) -> None:
        """
        Finalize active W&B run and report offline sync location if applicable.
        """
        if not self.enabled or self.run is None:
            return

        try:
            run_dir = self.run.dir
            wandb.finish()
            if self.mode == "offline":
                print(f"[W&B Logger] Offline run finalized. To sync metrics to cloud:")
                print(f"             wandb sync {run_dir}")
        except Exception as e:
            print(f"[WARN] Exception during wandb.finish(): {e}")
