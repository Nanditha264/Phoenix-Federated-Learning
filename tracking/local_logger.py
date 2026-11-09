"""
Local Structured JSONL and Console Logger.
Provides reliable on-disk telemetry logging independent of external services.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional


class LocalLogger:
    """
    Appends experiment metrics to JSONL file and formats console logs.
    """

    def __init__(self, output_dir: str, run_name: str, clean: bool = True):
        self.output_dir = Path(output_dir) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        if clean and self.jsonl_path.exists():
            try:
                self.jsonl_path.unlink()
            except OSError:
                pass
        self.history = []

    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        """
        Record a step's metrics to JSONL file.
        """
        record = {
            "step": step,
            "timestamp": time.time(),
            **metrics,
        }
        self.history.append(record)

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def save_summary(self, summary_data: Dict[str, Any]) -> None:
        """
        Write final experiment summary JSON.
        """
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

    def print_round_progress(
        self,
        round_idx: int,
        total_rounds: int,
        train_loss: float,
        val_acc: float,
        val_loss: float,
        elapsed_sec: float,
        ram_avail_mb: float,
    ) -> None:
        """
        Print clean, formatted progress to stdout.
        """
        print(
            f"[Round {round_idx:03d}/{total_rounds:03d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed_sec:.1f}s | "
            f"RAM Avail: {ram_avail_mb:.0f} MB"
        )
