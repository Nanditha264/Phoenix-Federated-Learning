"""
Evaluation Package for Federated Learning Benchmark.
"""

from fl_benchmark.evaluation.evaluator import GlobalEvaluator
from fl_benchmark.evaluation.client_evaluator import ClientValidationEvaluator

__all__ = ["GlobalEvaluator", "ClientValidationEvaluator"]
