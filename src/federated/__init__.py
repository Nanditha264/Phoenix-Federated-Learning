"""
Federated Learning Algorithmic Package.
"""

from src.federated.client import FederatedClient
from src.federated.fedavg import FedAvgClient
from src.federated.fedprox import FedProxClient
from src.federated.aggregation import aggregate_models
from src.federated.server import FederatedServer

__all__ = [
    "FederatedClient",
    "FedAvgClient",
    "FedProxClient",
    "aggregate_models",
    "FederatedServer",
]
