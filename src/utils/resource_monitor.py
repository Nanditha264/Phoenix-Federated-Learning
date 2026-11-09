"""
Resource Monitoring and Safety System for Federated Learning.
Monitors process and host memory, enforcing safe execution on memory-constrained systems.
"""

import os
import gc
import psutil
from typing import Dict, Tuple, Any, Optional


class FederatedResourceMonitor:
    """
    Monitors process RSS, host available memory, and CPU utilization.
    Provides multi-tiered safety triggers to prevent system thrashing or OOM panics.
    """

    def __init__(
        self,
        warning_threshold_mb: float = 800.0,       # 800 MB
        client_pause_threshold_mb: float = 550.0,  # 550 MB
        emergency_stop_threshold_mb: float = 350.0, # 350 MB hard safety floor
    ):
        self.warning_threshold_mb = warning_threshold_mb
        self.client_pause_threshold_mb = client_pause_threshold_mb
        self.emergency_stop_threshold_mb = emergency_stop_threshold_mb
        self.process = psutil.Process(os.getpid())

    def get_system_telemetry(self) -> Dict[str, float]:
        """
        Collect live host and process resource utilization metrics.
        """
        vm = psutil.virtual_memory()
        mem_info = self.process.memory_info()
        return {
            "ram_total_mb": round(vm.total / (1024 ** 2), 2),
            "ram_available_mb": round(vm.available / (1024 ** 2), 2),
            "ram_utilization_pct": round(vm.percent, 2),
            "process_rss_mb": round(mem_info.rss / (1024 ** 2), 2),
            "cpu_utilization_pct": round(psutil.cpu_percent(interval=None), 2),
        }

    def evaluate_state(self, simulated_avail_mb: Optional[float] = None) -> Tuple[str, str]:
        """
        Evaluate current available memory against established safety thresholds.

        Args:
            simulated_avail_mb: Optional override for deterministic unit testing.

        Returns:
            Tuple of (action_code, human_readable_message)
            action_code in {"NOMINAL", "WARNING", "DO_NOT_START_CLIENT", "GRACEFUL_STOP"}
        """
        if simulated_avail_mb is not None:
            avail_mb = simulated_avail_mb
        else:
            avail_mb = psutil.virtual_memory().available / (1024 ** 2)

        if avail_mb < self.emergency_stop_threshold_mb:
            action = "GRACEFUL_STOP"
            msg = (
                f"[CRITICAL] Available RAM ({avail_mb:.1f} MB) is below emergency limit "
                f"({self.emergency_stop_threshold_mb:.1f} MB). Emergency stop required."
            )
        elif avail_mb < self.client_pause_threshold_mb:
            action = "DO_NOT_START_CLIENT"
            msg = (
                f"[PAUSE] Available RAM ({avail_mb:.1f} MB) is below 1.0 GB limit "
                f"({self.client_pause_threshold_mb:.1f} MB). Suspend client launch until memory recovers."
            )
        elif avail_mb < self.warning_threshold_mb:
            action = "WARNING"
            gc.collect()
            msg = (
                f"[WARN] Available RAM ({avail_mb:.1f} MB) is below 1.5 GB limit "
                f"({self.warning_threshold_mb:.1f} MB). Forced garbage collection triggered."
            )
        else:
            action = "NOMINAL"
            msg = f"[OK] Available RAM ({avail_mb:.1f} MB) is in nominal operating range."

        return action, msg
