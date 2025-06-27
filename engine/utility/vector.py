"""
Utility Vector Evaluation

This module provides the UtilityVector class that runs lm-eval-harness on N benchmarks
and returns an N-dimensional utility vector, similar to how the Verifier returns
a vector of verification scores.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
import time

# Lazy imports to keep CLI snappy
def _get_lm_eval():
    """Lazy import of lm_eval to avoid startup cost."""
    try:
        import lm_eval
        return lm_eval
    except ImportError:
        raise ImportError(
            "lm_eval is required for utility evaluation. "
            "Install with: pip install --upgrade 'lm-eval[vllm]'"
        )

def _get_vllm_inference():
    """Lazy import of VLLMInference."""
    from ..common.inference_utils import VLLMInference
    return VLLMInference


class UtilityVector:
    """
    Run lm-eval-harness on N benchmarks and return an N-dim utility vector.
    
    This class provides a clean interface for evaluating model utility across
    multiple benchmarks, returning a vector where each component represents
    the score for a specific benchmark.
    
    Usage
    -----
    >>> util_vec = UtilityVector(
    ...     benchmarks=["arc_challenge", "hellaswag", "gsm8k"],
    ...     vllm_mode="native",                # or "server"
    ...     vllm_kwargs={"gpu": "0", "dtype": "float16"},
    ... )
    >>> vec = util_vec.evaluate("/path/to/final_model")
    >>> print(vec)        # e.g. [0.82, 0.79, 0.63]
    """

    def __init__(
        self,
        benchmarks: List[str],
        *,
        batch_size: Union[int, str] = "auto",
        vllm_mode: str = "native",              # "native" or "server"
        vllm_kwargs: Optional[Dict[str, Any]] = None,
        aggregate_key: str = "acc_norm",        # default metric per task
        verbose: bool = False,
    ) -> None:
        """
        Initialize the UtilityVector evaluator.
        
        Args:
            benchmarks: List of benchmark task names to evaluate
            batch_size: Batch size for evaluation (int or "auto")
            vllm_mode: Either "native" (vLLM in same process) or "server" (vLLM server)
            vllm_kwargs: Additional arguments for vLLM initialization
            aggregate_key: Default metric key to extract from results
            verbose: Whether to print verbose output
        """
        self.benchmarks = benchmarks
        self.batch_size = batch_size
        self.vllm_mode = vllm_mode
        self.vllm_kwargs = vllm_kwargs or {}
        self.aggregate_key = aggregate_key
        self.verbose = verbose
        
        # Validate inputs
        if not benchmarks:
            raise ValueError("At least one benchmark must be specified")
        if vllm_mode not in ["native", "server"]:
            raise ValueError(f"vllm_mode must be 'native' or 'server', got {vllm_mode}")

    def evaluate(self, model_path: str) -> List[float]:
        """
        Evaluate the model on all benchmarks and return utility vector.
        
        Args:
            model_path: Path to model (local path or HuggingFace Hub ID)
            
        Returns:
            List of scores, one per benchmark in the same order as self.benchmarks
        """
        if self.verbose:
            print(f"[UtilityVector] Evaluating model: {model_path}")
            print(f"[UtilityVector] Benchmarks: {self.benchmarks}")
            print(f"[UtilityVector] Mode: {self.vllm_mode}")
        
        start_time = time.time()
        
        try:
            if self.vllm_mode == "native":
                results = self._run_native(model_path)
            elif self.vllm_mode == "server":
                results = self._run_server(model_path)
            else:
                raise ValueError(f"Unknown vllm_mode={self.vllm_mode}")
                
            elapsed = time.time() - start_time
            
            if self.verbose:
                print(f"[UtilityVector] Evaluation completed in {elapsed:.2f}s")
                for i, (benchmark, score) in enumerate(zip(self.benchmarks, results)):
                    print(f"[UtilityVector] {benchmark}: {score:.4f}")
                    
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"[UtilityVector] Error during evaluation: {e}")
            raise

    def _run_native(self, model_path: str) -> List[float]:
        """Run evaluation using vLLM native backend (fastest for local inference)."""
        lm_eval = _get_lm_eval()
        
        # Build model arguments
        model_args = [
            f"pretrained={model_path}",
        ]
        
        # Add vLLM-specific arguments
        if "tensor_parallel_size" in self.vllm_kwargs:
            model_args.append(f"tensor_parallel_size={self.vllm_kwargs['tensor_parallel_size']}")
        if "dtype" in self.vllm_kwargs:
            model_args.append(f"dtype={self.vllm_kwargs['dtype']}")
        if "gpu_memory_utilization" in self.vllm_kwargs:
            model_args.append(f"gpu_memory_utilization={self.vllm_kwargs['gpu_memory_utilization']}")
        
        # Add any other vLLM kwargs
        for k, v in self.vllm_kwargs.items():
            if k not in ["tensor_parallel_size", "dtype", "gpu_memory_utilization"]:
                model_args.append(f"{k}={v}")
        
        model_arg_str = ",".join(model_args)
        
        if self.verbose:
            print(f"[UtilityVector] Running native vLLM with args: {model_arg_str}")
        
        # Run evaluation
        results = lm_eval.simple_evaluate(
            model="vllm",
            model_args=model_arg_str,
            tasks=",".join(self.benchmarks),
            batch_size=self.batch_size,
        )["results"]
        
        # Extract scores for each benchmark
        return [self._extract_metric(results, task) for task in self.benchmarks]

    def _run_server(self, model_path: str) -> List[float]:
        """Run evaluation using vLLM server mode."""
        lm_eval = _get_lm_eval()
        VLLMInference = _get_vllm_inference()
        
        if self.verbose:
            print(f"[UtilityVector] Starting vLLM server...")
        
        # Start vLLM server using our existing infrastructure
        with VLLMInference(model=model_path, **self.vllm_kwargs) as server:
            base_url = f"http://{server.host}:{server.port}/v1"
            
            model_args = ",".join([
                f"model={model_path}",
                f"base_url={base_url}/completions",
            ])
            
            if self.verbose:
                print(f"[UtilityVector] Server ready at {base_url}")
                print(f"[UtilityVector] Running evaluation with model_args: {model_args}")
            
            # Run evaluation against the server
            results = lm_eval.simple_evaluate(
                model="local-completions",
                model_args=model_args,
                tasks=",".join(self.benchmarks),
                batch_size=self.batch_size,
            )["results"]
        
        # Extract scores for each benchmark
        return [self._extract_metric(results, task) for task in self.benchmarks]

    def _extract_metric(self, results: Dict[str, Dict[str, float]], task: str) -> float:
        """
        Extract the requested metric from lm-eval results.
        
        Args:
            results: Full results dictionary from lm_eval
            task: Task name to extract metric for
            
        Returns:
            Score for the task, or 0.0 if not found
        """
        task_results = results.get(task, {})
        
        # Try exact key first
        if self.aggregate_key in task_results:
            return task_results[self.aggregate_key]
        
        # Try with ",none" suffix that lm-eval sometimes adds
        alt_key = f"{self.aggregate_key},none"
        if alt_key in task_results:
            return task_results[alt_key]
        
        # Try common alternatives
        common_keys = ["acc", "acc_norm", "exact_match", "score"]
        for key in common_keys:
            if key in task_results:
                if self.verbose:
                    print(f"[UtilityVector] Using {key} instead of {self.aggregate_key} for {task}")
                return task_results[key]
            alt_key = f"{key},none"
            if alt_key in task_results:
                if self.verbose:
                    print(f"[UtilityVector] Using {alt_key} instead of {self.aggregate_key} for {task}")
                return task_results[alt_key]
        
        if self.verbose:
            print(f"[UtilityVector] Warning: No suitable metric found for {task}")
            print(f"[UtilityVector] Available metrics: {list(task_results.keys())}")
        
        return 0.0

    def get_benchmark_names(self) -> List[str]:
        """Get the list of benchmark names."""
        return self.benchmarks.copy()

    def __len__(self) -> int:
        """Return the number of benchmarks."""
        return len(self.benchmarks)

    def __repr__(self) -> str:
        """String representation of the UtilityVector."""
        return (
            f"UtilityVector(benchmarks={self.benchmarks}, "
            f"vllm_mode={self.vllm_mode}, "
            f"batch_size={self.batch_size})"
        )


# Convenience functions for common benchmark suites

def create_openllm_utility_vector(**kwargs) -> UtilityVector:
    """Create a UtilityVector for OpenLLM Leaderboard benchmarks."""
    benchmarks = [
        "arc_challenge",
        "hellaswag", 
        "truthfulqa_mc2",
        "mmlu",
        "winogrande",
        "gsm8k"
    ]
    return UtilityVector(benchmarks=benchmarks, **kwargs)


def create_tiny_utility_vector(**kwargs) -> UtilityVector:
    """Create a UtilityVector for tiny/fast benchmarks (good for testing)."""
    benchmarks = [
        "tinyArc",
        "tinyHellaswag",
        "tinyTruthfulQA", 
        "tinyMMLU",
        "tinyWinogrande",
        "tinyGSM8k"
    ]
    return UtilityVector(benchmarks=benchmarks, **kwargs)


def create_math_utility_vector(**kwargs) -> UtilityVector:
    """Create a UtilityVector focused on mathematical reasoning."""
    benchmarks = [
        "gsm8k",
        "math_qa",
        "mathqa"
    ]
    return UtilityVector(benchmarks=benchmarks, **kwargs)


def create_reasoning_utility_vector(**kwargs) -> UtilityVector:
    """Create a UtilityVector focused on reasoning tasks."""
    benchmarks = [
        "arc_challenge",
        "arc_easy", 
        "hellaswag",
        "winogrande",
        "piqa"
    ]
    return UtilityVector(benchmarks=benchmarks, **kwargs) 