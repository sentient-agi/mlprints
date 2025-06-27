"""
Benchmark Implementations

This module implements various standard benchmarks for evaluating model utility
and performance across different tasks and domains.

Updated to use UtilityVector for actual evaluation via lm-eval-harness + vLLM.
"""

from typing import List, Dict, Any, Optional, Union
import json
import time
from pathlib import Path

from .base import UtilityEvaluatorBase, EvaluationResult, UtilityConfig, BenchmarkType


class BenchmarkSuite(UtilityEvaluatorBase):
    """
    Comprehensive benchmark suite for model evaluation.
    
    This class orchestrates multiple benchmark evaluations using UtilityVector
    and provides unified results reporting.
    """
    
    def __init__(self, config: UtilityConfig):
        super().__init__(config)
        self.benchmark_implementations = self._initialize_benchmarks()
        
    def _initialize_benchmarks(self) -> Dict[str, 'StandardBenchmarks']:
        """Initialize all benchmark implementations."""
        benchmarks = {}
        
        for benchmark_type in self.config.benchmark_types:
            if benchmark_type != BenchmarkType.CUSTOM:
                benchmarks[benchmark_type.value] = StandardBenchmarks(
                    self.config, benchmark_type
                )
                
        return benchmarks
        
    def evaluate(
        self,
        model_path_or_model: Union[str, Any],
        tokenizer=None,  # Not used with vLLM
        benchmark_name: Optional[str] = None
    ) -> List[EvaluationResult]:
        """Execute comprehensive benchmark evaluation using UtilityVector."""
        # Import here to avoid circular imports
        from .vector import UtilityVector
        
        results = []
        
        if benchmark_name:
            # Run specific benchmark
            if benchmark_name in self.benchmark_implementations:
                result = self.benchmark_implementations[benchmark_name].evaluate(
                    model_path_or_model, tokenizer
                )
                results.append(result)
        else:
            # Run all configured benchmarks using UtilityVector
            benchmark_names = [bt.value for bt in self.config.benchmark_types if bt != BenchmarkType.CUSTOM]
            
            if benchmark_names:
                print(f"Running benchmarks: {benchmark_names}")
                start_time = time.time()
                
                try:
                    # Create UtilityVector with appropriate settings
                    vllm_kwargs = self.config.additional_params or {}
                    if hasattr(self.config, 'device') and self.config.device.startswith('cuda'):
                        gpu_id = self.config.device.split(':')[-1] if ':' in self.config.device else "0"
                        vllm_kwargs.setdefault('gpu', gpu_id)
                    
                    utility_vector = UtilityVector(
                        benchmarks=benchmark_names,
                        batch_size=self.config.batch_size,
                        vllm_mode="native",  # Use native mode for best performance
                        vllm_kwargs=vllm_kwargs,
                        verbose=True
                    )
                    
                    # Get scores
                    scores = utility_vector.evaluate(str(model_path_or_model))
                    
                    # Convert to EvaluationResult objects
                    for benchmark_name, score in zip(benchmark_names, scores):
                        result = EvaluationResult(
                            benchmark_name=benchmark_name,
                            score=score,
                            max_score=1.0,
                            accuracy=score,  # For most benchmarks, score is accuracy
                            num_samples=0,  # Not tracked by UtilityVector
                            execution_time=time.time() - start_time,
                            metadata={
                                "evaluation_type": self.config.evaluation_type.value,
                                "batch_size": self.config.batch_size,
                                "vllm_mode": "native"
                            }
                        )
                        results.append(result)
                        
                    print(f"Completed all benchmarks in {time.time() - start_time:.2f}s")
                    
                except Exception as e:
                    print(f"Error in benchmark suite: {e}")
                    # Create error results
                    for benchmark_name in benchmark_names:
                        result = EvaluationResult(
                            benchmark_name=benchmark_name,
                            score=0.0,
                            max_score=1.0,
                            accuracy=0.0,
                            num_samples=0,
                            execution_time=0.0,
                            metadata={"error": str(e)}
                        )
                        results.append(result)
                    
        return results
        
    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
        task_type: str
    ) -> Dict[str, float]:
        """Compute aggregated metrics across benchmarks."""
        # This is handled by lm-eval-harness internally
        return {
            "accuracy": 0.0,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }


class StandardBenchmarks(UtilityEvaluatorBase):
    """
    Implementation of standard AI benchmarks using UtilityVector.
    
    This class provides a wrapper around UtilityVector for individual
    benchmark evaluation.
    """
    
    def __init__(self, config: UtilityConfig, benchmark_type: BenchmarkType):
        super().__init__(config)
        self.benchmark_type = benchmark_type
        self.benchmark_name = benchmark_type.value
        
    def evaluate(
        self,
        model_path_or_model: Union[str, Any],
        tokenizer=None,  # Not used with vLLM
        benchmark_name: Optional[str] = None
    ) -> EvaluationResult:
        """Execute specific standard benchmark using UtilityVector."""
        from .vector import UtilityVector
        
        start_time = time.time()
        
        try:
            # Map benchmark types to actual task names
            task_name = self._get_task_name()
            
            # Create UtilityVector for single benchmark
            vllm_kwargs = self.config.additional_params or {}
            if hasattr(self.config, 'device') and self.config.device.startswith('cuda'):
                gpu_id = self.config.device.split(':')[-1] if ':' in self.config.device else "0"
                vllm_kwargs.setdefault('gpu', gpu_id)
            
            utility_vector = UtilityVector(
                benchmarks=[task_name],
                batch_size=self.config.batch_size,
                vllm_mode="native",
                vllm_kwargs=vllm_kwargs,
                verbose=True
            )
            
            # Get score
            scores = utility_vector.evaluate(str(model_path_or_model))
            score = scores[0] if scores else 0.0
            
            execution_time = time.time() - start_time
            
            result = EvaluationResult(
                benchmark_name=self.benchmark_name,
                score=score,
                max_score=1.0,
                accuracy=score,
                num_samples=0,  # Not tracked by UtilityVector
                execution_time=execution_time,
                metadata={
                    "benchmark_type": self.benchmark_type.value,
                    "task_name": task_name,
                    "config": self.config.__dict__
                }
            )
            
            return result
            
        except Exception as e:
            print(f"Error evaluating {self.benchmark_name}: {e}")
            return EvaluationResult(
                benchmark_name=self.benchmark_name,
                score=0.0,
                max_score=1.0,
                accuracy=0.0,
                num_samples=0,
                execution_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _get_task_name(self) -> str:
        """Map benchmark type to actual lm-eval task name."""
        task_mapping = {
            BenchmarkType.IFEVAL: "ifeval",
            BenchmarkType.GSM8K: "gsm8k",
            BenchmarkType.HELLASWAG: "hellaswag",
            BenchmarkType.TRUTHFULQA: "truthfulqa_mc2",
            BenchmarkType.MMLU: "mmlu",
            BenchmarkType.HUMANEVAL: "humaneval",
        }
        return task_mapping.get(self.benchmark_type, self.benchmark_type.value)
        
    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
        task_type: str
    ) -> Dict[str, float]:
        """Compute benchmark-specific metrics (handled by lm-eval internally)."""
        return {"accuracy": 0.0, "score": 0.0}


class CustomBenchmark(UtilityEvaluatorBase):
    """
    Custom benchmark implementation for user-defined evaluation tasks.
    
    This class allows users to define and run custom evaluation benchmarks
    using UtilityVector if the task is supported by lm-eval-harness.
    """
    
    def __init__(self, config: UtilityConfig, custom_task_name: str):
        super().__init__(config)
        self.custom_task_name = custom_task_name
        self.benchmark_name = "custom"
        
    def evaluate(
        self,
        model_path_or_model: Union[str, Any],
        tokenizer=None,
        benchmark_name: Optional[str] = None
    ) -> EvaluationResult:
        """Execute custom benchmark evaluation using UtilityVector."""
        from .vector import UtilityVector
        
        start_time = time.time()
        
        try:
            # Create UtilityVector for custom task
            vllm_kwargs = self.config.additional_params or {}
            if hasattr(self.config, 'device') and self.config.device.startswith('cuda'):
                gpu_id = self.config.device.split(':')[-1] if ':' in self.config.device else "0"
                vllm_kwargs.setdefault('gpu', gpu_id)
            
            utility_vector = UtilityVector(
                benchmarks=[self.custom_task_name],
                batch_size=self.config.batch_size,
                vllm_mode="native",
                vllm_kwargs=vllm_kwargs,
                verbose=True
            )
            
            # Get score
            scores = utility_vector.evaluate(str(model_path_or_model))
            score = scores[0] if scores else 0.0
            
            result = EvaluationResult(
                benchmark_name=self.benchmark_name,
                score=score,
                max_score=1.0,
                accuracy=score,
                num_samples=0,
                execution_time=time.time() - start_time,
                metadata={"custom_task_name": self.custom_task_name}
            )
            
            return result
            
        except Exception as e:
            print(f"Error evaluating custom task {self.custom_task_name}: {e}")
            return EvaluationResult(
                benchmark_name=self.benchmark_name,
                score=0.0,
                max_score=1.0,
                accuracy=0.0,
                num_samples=0,
                execution_time=time.time() - start_time,
                metadata={"error": str(e), "custom_task_name": self.custom_task_name}
            )
        
    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
        task_type: str
    ) -> Dict[str, float]:
        """Compute custom metrics (handled by lm-eval internally)."""
        return {"accuracy": 0.0, "score": 0.0} 