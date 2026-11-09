"""
Phoenix Federated Learning Core Package.
Provides modules for data partitioning, models, federated algorithms (FedAvg, FedProx),
evaluation, geometry metrics, tracking, and utilities.
"""
import sys
# Support both 'from src.module' and legacy 'from src.module'
sys.modules.setdefault("fl_benchmark", sys.modules[__name__])
