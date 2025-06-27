"""
Tests for UtilityVector implementation

These tests verify that the UtilityVector class works correctly for evaluating
model utility across multiple benchmarks using vLLM and lm-eval-harness.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from engine.utility.vector import (
    UtilityVector,
    create_openllm_utility_vector,
    create_tiny_utility_vector,
    create_math_utility_vector,
    create_reasoning_utility_vector
)


class TestUtilityVector:
    """Test cases for UtilityVector class."""
    
    def test_init_valid_params(self):
        """Test UtilityVector initialization with valid parameters."""
        uv = UtilityVector(
            benchmarks=["arc_challenge", "hellaswag"],
            vllm_mode="native",
            batch_size=8,
            verbose=True
        )
        
        assert uv.benchmarks == ["arc_challenge", "hellaswag"]
        assert uv.vllm_mode == "native"
        assert uv.batch_size == 8
        assert uv.verbose is True
        assert len(uv) == 2
    
    def test_init_invalid_benchmarks(self):
        """Test UtilityVector initialization with invalid benchmarks."""
        with pytest.raises(ValueError, match="At least one benchmark must be specified"):
            UtilityVector(benchmarks=[])
    
    def test_init_invalid_vllm_mode(self):
        """Test UtilityVector initialization with invalid vLLM mode."""
        with pytest.raises(ValueError, match="vllm_mode must be 'native' or 'server'"):
            UtilityVector(benchmarks=["arc_challenge"], vllm_mode="invalid")
    
    def test_get_benchmark_names(self):
        """Test getting benchmark names."""
        uv = UtilityVector(benchmarks=["arc_challenge", "hellaswag"])
        names = uv.get_benchmark_names()
        
        assert names == ["arc_challenge", "hellaswag"]
        # Ensure it returns a copy
        names.append("mmlu")
        assert uv.benchmarks == ["arc_challenge", "hellaswag"]
    
    def test_repr(self):
        """Test string representation."""
        uv = UtilityVector(
            benchmarks=["arc_challenge"],
            vllm_mode="server",
            batch_size="auto"
        )
        
        repr_str = repr(uv)
        assert "UtilityVector" in repr_str
        assert "arc_challenge" in repr_str
        assert "server" in repr_str
        assert "auto" in repr_str
    
    @patch('engine.utility.vector._get_lm_eval')
    def test_run_native_success(self, mock_get_lm_eval):
        """Test successful native vLLM evaluation."""
        # Mock lm_eval
        mock_lm_eval = Mock()
        mock_get_lm_eval.return_value = mock_lm_eval
        
        # Mock evaluation results
        mock_results = {
            "results": {
                "arc_challenge": {"acc_norm": 0.85, "acc": 0.80},
                "hellaswag": {"acc_norm": 0.78}
            }
        }
        mock_lm_eval.simple_evaluate.return_value = mock_results
        
        uv = UtilityVector(
            benchmarks=["arc_challenge", "hellaswag"],
            vllm_mode="native",
            vllm_kwargs={"tensor_parallel_size": 2, "dtype": "float16"}
        )
        
        scores = uv.evaluate("test_model_path")
        
        assert scores == [0.85, 0.78]
        
        # Verify lm_eval was called correctly
        mock_lm_eval.simple_evaluate.assert_called_once()
        call_args = mock_lm_eval.simple_evaluate.call_args
        
        assert call_args[1]["model"] == "vllm"
        assert "pretrained=test_model_path" in call_args[1]["model_args"]
        assert "tensor_parallel_size=2" in call_args[1]["model_args"]
        assert "dtype=float16" in call_args[1]["model_args"]
        assert call_args[1]["tasks"] == "arc_challenge,hellaswag"
    
    @patch('engine.utility.vector._get_lm_eval')
    @patch('engine.utility.vector._get_vllm_inference')
    def test_run_server_success(self, mock_get_vllm_inference, mock_get_lm_eval):
        """Test successful server vLLM evaluation."""
        # Mock lm_eval
        mock_lm_eval = Mock()
        mock_get_lm_eval.return_value = mock_lm_eval
        
        # Mock VLLMInference
        mock_vllm_class = Mock()
        mock_get_vllm_inference.return_value = mock_vllm_class
        
        # Mock server instance
        mock_server = Mock()
        mock_server.host = "localhost"
        mock_server.port = 8000
        mock_vllm_class.return_value.__enter__.return_value = mock_server
        
        # Mock evaluation results
        mock_results = {
            "results": {
                "gsm8k": {"exact_match,strict-match": 0.65}
            }
        }
        mock_lm_eval.simple_evaluate.return_value = mock_results
        
        uv = UtilityVector(
            benchmarks=["gsm8k"],
            vllm_mode="server",
            aggregate_key="exact_match,strict-match"
        )
        
        scores = uv.evaluate("test_model_path")
        
        assert scores == [0.65]
        
        # Verify server was created and used
        mock_vllm_class.assert_called_once_with(model="test_model_path")
        mock_lm_eval.simple_evaluate.assert_called_once()
        call_args = mock_lm_eval.simple_evaluate.call_args
        
        assert call_args[1]["model"] == "local-completions"
        assert "base_url=http://localhost:8000/v1/completions" in call_args[1]["model_args"]
    
    def test_extract_metric_exact_key(self):
        """Test metric extraction with exact key match."""
        uv = UtilityVector(benchmarks=["test"], aggregate_key="acc_norm")
        
        results = {
            "test": {"acc_norm": 0.85, "acc": 0.80}
        }
        
        score = uv._extract_metric(results, "test")
        assert score == 0.85
    
    def test_extract_metric_with_none_suffix(self):
        """Test metric extraction with ',none' suffix."""
        uv = UtilityVector(benchmarks=["test"], aggregate_key="acc_norm")
        
        results = {
            "test": {"acc_norm,none": 0.85, "acc": 0.80}
        }
        
        score = uv._extract_metric(results, "test")
        assert score == 0.85
    
    def test_extract_metric_fallback_to_common(self):
        """Test metric extraction fallback to common keys."""
        uv = UtilityVector(benchmarks=["test"], aggregate_key="missing_key", verbose=True)
        
        results = {
            "test": {"acc": 0.80, "score": 0.75}
        }
        
        score = uv._extract_metric(results, "test")
        assert score == 0.80  # Should use first common key found
    
    def test_extract_metric_no_match(self):
        """Test metric extraction when no suitable metric found."""
        uv = UtilityVector(benchmarks=["test"], aggregate_key="missing_key")
        
        results = {
            "test": {"other_metric": 0.80}
        }
        
        score = uv._extract_metric(results, "test")
        assert score == 0.0
    
    def test_extract_metric_missing_task(self):
        """Test metric extraction for missing task."""
        uv = UtilityVector(benchmarks=["test"], aggregate_key="acc")
        
        results = {
            "other_task": {"acc": 0.80}
        }
        
        score = uv._extract_metric(results, "test")
        assert score == 0.0


class TestConvenienceFunctions:
    """Test convenience functions for creating UtilityVector instances."""
    
    def test_create_openllm_utility_vector(self):
        """Test creating OpenLLM leaderboard utility vector."""
        uv = create_openllm_utility_vector(vllm_mode="native", batch_size=16)
        
        expected_benchmarks = [
            "arc_challenge", "hellaswag", "truthfulqa_mc2", 
            "mmlu", "winogrande", "gsm8k"
        ]
        assert uv.benchmarks == expected_benchmarks
        assert uv.vllm_mode == "native"
        assert uv.batch_size == 16
    
    def test_create_tiny_utility_vector(self):
        """Test creating tiny benchmarks utility vector."""
        uv = create_tiny_utility_vector(verbose=True)
        
        expected_benchmarks = [
            "tinyArc", "tinyHellaswag", "tinyTruthfulQA",
            "tinyMMLU", "tinyWinogrande", "tinyGSM8k"
        ]
        assert uv.benchmarks == expected_benchmarks
        assert uv.verbose is True
    
    def test_create_math_utility_vector(self):
        """Test creating math-focused utility vector."""
        uv = create_math_utility_vector(aggregate_key="exact_match")
        
        expected_benchmarks = ["gsm8k", "math_qa", "mathqa"]
        assert uv.benchmarks == expected_benchmarks
        assert uv.aggregate_key == "exact_match"
    
    def test_create_reasoning_utility_vector(self):
        """Test creating reasoning-focused utility vector."""
        uv = create_reasoning_utility_vector(vllm_kwargs={"dtype": "float16"})
        
        expected_benchmarks = [
            "arc_challenge", "arc_easy", "hellaswag", 
            "winogrande", "piqa"
        ]
        assert uv.benchmarks == expected_benchmarks
        assert uv.vllm_kwargs == {"dtype": "float16"}


class TestErrorHandling:
    """Test error handling in UtilityVector."""
    
    @patch('engine.utility.vector._get_lm_eval')
    def test_evaluate_with_lm_eval_error(self, mock_get_lm_eval):
        """Test evaluation with lm_eval error."""
        # Mock lm_eval to raise an exception
        mock_lm_eval = Mock()
        mock_get_lm_eval.return_value = mock_lm_eval
        mock_lm_eval.simple_evaluate.side_effect = Exception("LM eval failed")
        
        uv = UtilityVector(benchmarks=["arc_challenge"], vllm_mode="native", verbose=True)
        
        with pytest.raises(Exception, match="LM eval failed"):
            uv.evaluate("test_model_path")
    
    def test_missing_lm_eval_import(self):
        """Test behavior when lm_eval is not installed."""
        with patch('engine.utility.vector._get_lm_eval') as mock_get_lm_eval:
            mock_get_lm_eval.side_effect = ImportError("lm_eval not found")
            
            uv = UtilityVector(benchmarks=["arc_challenge"], vllm_mode="native")
            
            with pytest.raises(ImportError, match="lm_eval is required"):
                uv.evaluate("test_model_path")


if __name__ == "__main__":
    pytest.main([__file__]) 