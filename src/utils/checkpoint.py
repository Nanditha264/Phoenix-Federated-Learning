"""
Atomic Checkpointing and Model State Persistence.
Prevents file corruption on interrupted writes using atomic replacement.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class CheckpointManager:
    """
    Manages atomic serialization and loading of global models and experiment metadata.
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric_val = -float("inf")

    def save_atomic(
        self,
        payload: Dict[str, Any],
        target_path: Path
    ) -> None:
        """
        Atomically write PyTorch payload to target_path using temporary file + rename.
        """
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target_path.with_suffix(f".tmp_{os.getpid()}")
        try:
            torch.save(payload, temp_file)
            # os.replace provides atomic overwrite on POSIX and Windows NT
            os.replace(temp_file, target_path)
        except Exception as e:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            raise IOError(f"Failed atomic checkpoint write to {target_path}: {e}")

    def save_round(
        self,
        model: nn.Module,
        round_idx: int,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> Path:
        """
        Save the model checkpoint for a completed communication round.
        """
        payload = {
            "round": round_idx,
            "state_dict": model.state_dict(),
            "metrics": metrics,
            "config": config,
        }

        # 1. Save last checkpoint
        last_path = self.checkpoint_dir / "last_checkpoint.pt"
        self.save_atomic(payload, last_path)

        # 2. Save best checkpoint if indicated
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            self.save_atomic(payload, best_path)

        # 3. Optional round-specific snapshot
        round_path = self.checkpoint_dir / f"round_{round_idx:03d}.pt"
        self.save_atomic(payload, round_path)

        return last_path

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: nn.Module
    ) -> Dict[str, Any]:
        """
        Load weights into model and return saved metadata.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        payload = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        return {
            "round": payload.get("round", 0),
            "metrics": payload.get("metrics", {}),
            "config": payload.get("config", {}),
        }
