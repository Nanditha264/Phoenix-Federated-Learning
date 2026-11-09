"""
Configuration Management for Federated Learning Benchmark.
Loads YAML configurations, handles hierarchical merging and schema validation.
"""

from pathlib import Path
from typing import Dict, Any, Union
import copy
import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override dictionary into base dictionary.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read and parse a YAML file from disk.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate presence and types of critical configuration keys.
    """
    required_keys = [
        "dataset", "num_clients", "dirichlet_alpha",
        "noise_rate", "noisy_client_id", "algorithm",
        "batch_size", "learning_rate", "local_epochs",
        "rounds", "seed"
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: '{key}'")

    if config["num_clients"] <= 0:
        raise ValueError(f"num_clients must be positive, got {config['num_clients']}")

    if not (0.0 <= config["noise_rate"] <= 1.0):
        raise ValueError(f"noise_rate must be in [0, 1], got {config['noise_rate']}")

    if not (0 <= config["noisy_client_id"] < config["num_clients"]):
        raise ValueError(
            f"noisy_client_id must be in [0, {config['num_clients'] - 1}], got {config['noisy_client_id']}"
        )

    if config["algorithm"].lower() not in {"fedavg", "fedprox"}:
        raise ValueError(f"Unsupported algorithm: '{config['algorithm']}'. Must be 'fedavg' or 'fedprox'")

    if config["algorithm"].lower() == "fedprox":
        if "fedprox_mu" not in config:
            raise ValueError("FedProx algorithm requires 'fedprox_mu' parameter")
        if config["fedprox_mu"] < 0:
            raise ValueError(f"fedprox_mu must be non-negative, got {config['fedprox_mu']}")


def load_experiment_config(
    base_config_path: Union[str, Path],
    override_config_path: Union[str, Path, None] = None
) -> Dict[str, Any]:
    """
    Load base YAML and optional override YAML, merge them, and validate.
    """
    base = load_yaml(base_config_path)
    if override_config_path is not None:
        override = load_yaml(override_config_path)
        merged = deep_merge(base, override)
    else:
        merged = base

    validate_config(merged)
    return merged
