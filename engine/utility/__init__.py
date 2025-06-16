"""
Utility Evaluation Module

This module provides comprehensive utility evaluation capabilities for OML models,
including benchmarking, metrics computation, and performance analysis.
"""

from .base import (
    EvaluationType,
    BenchmarkType,
    EvaluationResult,
    UtilityConfig,
    UtilityEvaluatorBase
)

from .metrics import (
    MetricResult,
    EvaluationMetrics,
    UtilityTracker,
    PerformanceAnalyzer
)

from .analysis import (
    ExperimentData,
    TrainingMetrics,
    ExperimentLoader,
    TrainingAnalyzer,
    create_color_mapping
)

from .benchmarks import (
    BenchmarkSuite,
    StandardBenchmarks,
    CustomBenchmark
)

from ..common.lm_eval_utils import evaluate_model, load_evaluation

__all__ = [
    # Base classes and types
    "EvaluationType",
    "BenchmarkType", 
    "EvaluationResult",
    "UtilityConfig",
    "UtilityEvaluatorBase",
    
    # Metrics
    "MetricResult",
    "EvaluationMetrics",
    "UtilityTracker",
    "PerformanceAnalyzer",
    
    # Analysis
    "ExperimentData",
    "TrainingMetrics", 
    "ExperimentLoader",
    "TrainingAnalyzer",
    "create_color_mapping",

    # Benchmarks
    "BenchmarkSuite",
    "StandardBenchmarks",
    "CustomBenchmark",

    # Export lm_eval helper
    "evaluate_model",
    "load_evaluation"
] 