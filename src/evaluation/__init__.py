"""
Evaluation Package for Federated Learning Benchmark.
"""

from src.evaluation.evaluator import GlobalEvaluator
from src.evaluation.client_evaluator import ClientValidationEvaluator

__all__ = ["GlobalEvaluator", "ClientValidationEvaluator"]
