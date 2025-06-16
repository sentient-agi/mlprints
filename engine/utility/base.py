"""
Base Utility Classes

Defines the abstract base classes and common data structures for utility evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time


class EvaluationType(Enum):
    """Types of utility evaluations supported."""
    BENCHMARK = "benchmark"
    CUSTOM_TASK = "custom_task"
    PERFORMANCE_METRICS = "performance_metrics"
    COMPARATIVE = "comparative"


class BenchmarkType(Enum):
    """Standard benchmark types."""
    IFEVAL = "ifeval"
    GSM8K = "gsm8k"
    HELLASWAG = "hellaswag"
    TRUTHFULQA = "truthfulqa"
    MMLU = "mmlu"
    HUMANEVAL = "humaneval"
    CUSTOM = "custom"


@dataclass
class EvaluationResult:
    """Result of a utility evaluation."""
    benchmark_name: str
    score: float
    max_score: float
    accuracy: float
    num_samples: int
    execution_time: float
    metadata: Dict[str, Any]
    detailed_results: Optional[Dict[str, Any]] = None
    error_analysis: Optional[Dict[str, Any]] = None


@dataclass  
class UtilityConfig:
    """Configuration for utility evaluation."""
    evaluation_type: EvaluationType
    benchmark_types: List[BenchmarkType]
    batch_size: int = 32
    max_samples: Optional[int] = None
    few_shot: int = 0
    device: str = "cuda"
    output_dir: str = "./utility_results"
    additional_params: Dict[str, Any] = None


class UtilityEvaluatorBase(ABC):
    """
    Abstract base class for all utility evaluation implementations.
    
    This class defines the common interface that all utility evaluators must implement.
    """
    
    def __init__(self, config: UtilityConfig):
        self.config = config
        self.evaluation_type = config.evaluation_type
        self.device = config.device
        self.results_cache = {}
        
    @abstractmethod
    def evaluate(
        self,
        model,
        tokenizer,
        benchmark_name: Optional[str] = None
    ) -> Union[EvaluationResult, List[EvaluationResult]]:
        """
        Execute the utility evaluation.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer for the model
            benchmark_name: Optional specific benchmark to run
            
        Returns:
            EvaluationResult or list of EvaluationResults
        """
        pass
        
    @abstractmethod
    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
        task_type: str
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for predictions vs references.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            task_type: Type of task being evaluated
            
        Returns:
            Dictionary of computed metrics
        """
        pass
        
    def preprocess_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess evaluation data.
        
        Args:
            data: Raw evaluation data
            
        Returns:
            Preprocessed data
        """
        # TODO: Implement data preprocessing
        # - Tokenization
        # - Format standardization
        # - Filtering
        return data
        
    def postprocess_results(self, results: List[EvaluationResult]) -> List[EvaluationResult]:
        """
        Postprocess evaluation results.
        
        Args:
            results: Raw evaluation results
            
        Returns:
            Processed evaluation results
        """
        # TODO: Implement result postprocessing
        # - Result aggregation
        # - Statistical analysis
        # - Error analysis
        return results
        
    def generate_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Evaluation results to report on
            
        Returns:
            Formatted evaluation report
        """
        # TODO: Implement report generation
        # - Summary statistics
        # - Detailed breakdowns
        # - Comparison with baselines
        
        report = {
            "summary": {
                "total_benchmarks": len(results),
                "average_score": sum(r.score for r in results) / len(results) if results else 0,
                "average_accuracy": sum(r.accuracy for r in results) / len(results) if results else 0
            },
            "detailed_results": [
                {
                    "benchmark": result.benchmark_name,
                    "score": result.score,
                    "accuracy": result.accuracy,
                    "samples": result.num_samples,
                    "time": result.execution_time
                }
                for result in results
            ],
            "metadata": {
                "evaluation_time": time.time(),
                "config": self.config.__dict__
            }
        }
        
        return report
        
    def cache_results(self, key: str, result: EvaluationResult):
        """
        Cache evaluation results for efficiency.
        
        Args:
            key: Cache key
            result: Result to cache
        """
        self.results_cache[key] = result
        
    def get_cached_result(self, key: str) -> Optional[EvaluationResult]:
        """
        Retrieve cached evaluation result.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result if available
        """
        return self.results_cache.get(key) 