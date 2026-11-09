"""
Unit Tests for Local and W&B Tracking Engines.
"""

import json
from pathlib import Path
import tempfile
import pytest

from src.tracking.local_logger import LocalLogger
from src.tracking.wandb_logger import WandbLogger


def test_local_logger_jsonl_and_summary():
    """Verify local JSONL metric logging and summary persistence."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = LocalLogger(output_dir=tmp_dir, run_name="test_run")
        logger.log_metrics(step=1, metrics={"loss": 0.5, "acc": 0.8})
        logger.log_metrics(step=2, metrics={"loss": 0.4, "acc": 0.85})
        logger.save_summary({"best_acc": 0.85})

        jsonl_file = Path(tmp_dir) / "test_run" / "metrics.jsonl"
        summary_file = Path(tmp_dir) / "test_run" / "summary.json"

        assert jsonl_file.exists()
        assert summary_file.exists()

        lines = jsonl_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        record_1 = json.loads(lines[0])
        assert record_1["step"] == 1
        assert record_1["loss"] == 0.5

        summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
        assert summary_data["best_acc"] == 0.85


def test_wandb_disabled_mode():
    """Verify WandbLogger operates as a clean no-op when enabled=False."""
    logger = WandbLogger(enabled=False)
    assert not logger.enabled
    # None of these should throw
    logger.log({"loss": 0.5}, step=1)
    logger.finish()
