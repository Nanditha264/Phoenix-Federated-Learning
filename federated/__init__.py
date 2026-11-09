"""
Federated Learning Algorithmic Package.
"""

from fl_benchmark.federated.client import FederatedClient
from fl_benchmark.federated.fedavg import FedAvgClient
from fl_benchmark.federated.fedprox import FedProxClient
from fl_benchmark.federated.aggregation import aggregate_models
from fl_benchmark.federated.server import FederatedServer

__all__ = [
    "FederatedClient",
    "FedAvgClient",
    "FedProxClient",
    "aggregate_models",
    "FederatedServer",
]
