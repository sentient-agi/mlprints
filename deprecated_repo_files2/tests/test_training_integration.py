#!/usr/bin/env python3
"""
Comprehensive Training Integration Tests

This script provides a unified test runner for both training-only and 
training-with-evaluation scenarios, integrating the bash script logic
with the engine architecture.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add the engine directory to the Python path
script_dir = Path(__file__).parent
engine_dir = script_dir.parent / "engine"
sys.path.insert(0, str(engine_dir))

# Import our test modules
from test_training_without_eval import test_training_without_eval
from test_training_with_eval import test_training_with_eval

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingTestSuite:
    """
    Test suite for training and evaluation integration tests.
    """
    
    def __init__(self, base_output_dir: str = "test_logs"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_training_only_test(self, config: Dict[str, Any]) -> bool:
        """Run the training-only test."""
        logger.info("=" * 60)
        logger.info("RUNNING TRAINING-ONLY TEST")
        logger.info("=" * 60)
        
        output_dir = self.base_output_dir / f"training_only_{self.timestamp}"
        config['output_dir'] = str(output_dir)
        
        try:
            result = test_training_without_eval(**config)
            if result:
                logger.info("✓ Training-only test PASSED")
            else:
                logger.error("✗ Training-only test FAILED")
            return result
        except Exception as e:
            logger.error(f"Training-only test failed with exception: {e}")
            return False
            
    def run_training_with_eval_test(self, config: Dict[str, Any]) -> bool:
        """Run the training-with-evaluation test."""
        logger.info("=" * 60)
        logger.info("RUNNING TRAINING WITH EVALUATION TEST")
        logger.info("=" * 60)
        
        output_dir = self.base_output_dir / f"training_with_eval_{self.timestamp}"
        config['output_dir'] = str(output_dir)
        
        try:
            result = test_training_with_eval(**config)
            if result:
                logger.info("✓ Training-with-evaluation test PASSED")
            else:
                logger.error("✗ Training-with-evaluation test FAILED")
            return result
        except Exception as e:
            logger.error(f"Training-with-evaluation test failed with exception: {e}")
            return False
            
    def run_all_tests(self, config: Dict[str, Any]) -> Dict[str, bool]:
        """Run all integration tests."""
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE TRAINING INTEGRATION TESTS")
        logger.info("=" * 60)
        
        results = {}
        
        # Test 1: Training without evaluation
        logger.info("\n📋 Test 1: Training without evaluation (isolate training issues)")
        results['training_only'] = self.run_training_only_test(config.copy())
        
        # Test 2: Training with evaluation (if training-only passed)
        if results['training_only']:
            logger.info("\n📋 Test 2: Training with evaluation (complete pipeline)")
            results['training_with_eval'] = self.run_training_with_eval_test(config.copy())
        else:
            logger.warning("Skipping training-with-evaluation test due to training-only failure")
            results['training_with_eval'] = False
            
        return results
        
    def generate_report(self, results: Dict[str, bool], config: Dict[str, Any]) -> str:
        """Generate a comprehensive test report."""
        report_lines = [
            "=" * 80,
            "TRAINING INTEGRATION TEST REPORT",
            "=" * 80,
            f"Timestamp: {self.timestamp}",
            f"Model: {config.get('model_path', 'N/A')}",
            f"Fingerprints: {config.get('num_fingerprints', 'N/A')}",
            f"Epochs: {config.get('num_epochs', 'N/A')}",
            "",
            "TEST RESULTS:",
            "-" * 40,
        ]
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r)
        
        for test_name, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            test_display = test_name.replace('_', ' ').title()
            report_lines.append(f"{test_display:.<30} {status}")
            
        report_lines.extend([
            "",
            f"SUMMARY: {passed_tests}/{total_tests} tests passed",
            "",
            "RECOMMENDATIONS:",
            "-" * 40,
        ])
        
        if results.get('training_only', False) and results.get('training_with_eval', False):
            report_lines.append("✓ All tests passed! Training and evaluation systems are working correctly.")
        elif results.get('training_only', False) and not results.get('training_with_eval', False):
            report_lines.extend([
                "⚠ Training works but evaluation has issues.",
                "  → Focus debugging on evaluation components (lm-eval-harness, distributed sync)",
                "  → Check GPU assignments and NCCL timeout settings",
            ])
        elif not results.get('training_only', False):
            report_lines.extend([
                "❌ Core training system has issues.",
                "  → Focus debugging on training loop, model loading, or distributed setup",
                "  → Check model path, fingerprint generation, and basic training components",
            ])
        
        report_lines.extend([
            "",
            f"Detailed logs available in: {self.base_output_dir}",
            "=" * 80,
        ])
        
        return "\n".join(report_lines)


def create_default_config() -> Dict[str, Any]:
    """Create default test configuration matching the bash scripts."""
    return {
        'model_path': "/ephemeral/models/Llama-3.2-3B-Instruct",
        'num_fingerprints': 2,  # Minimal for training-only
        'max_key_length': 16,
        'max_response_length': 7,
        'num_epochs': 2,
        'batch_size': 2,
        'fingerprint_strategy': 'random_words',
        # Evaluation-specific (used only for training-with-eval test)
        'eval_tasks': 'ifeval',
        'eval_every': 1,
        'eval_lm_batch_size': 64,
        'eval_lm_limit': 10,
        'eval_num_fewshot': 0,
        'timeout_minutes': 10,
    }


def create_config_from_bash_script_params() -> Dict[str, Any]:
    """Create test configuration that matches the original bash script parameters."""
    return {
        # From test_training_without_eval.sh
        'model_path': "/ephemeral/models/Llama-3.2-3B-Instruct",
        'num_fingerprints': 2,
        'max_key_length': 16,
        'max_response_length': 7,
        'num_epochs': 2,
        'batch_size': 2,
        'fingerprint_strategy': 'random_words',  # Equivalent to 'english' in bash
        
        # From test_eval_setup.sh (enhanced for evaluation test)
        'num_fingerprints_eval': 8,  # More fingerprints for evaluation test
        'eval_tasks': 'ifeval',
        'eval_every': 1,
        'eval_lm_batch_size': 64,
        'eval_lm_limit': 10,
        'eval_num_fewshot': 0,
        'timeout_minutes': 10,
    }


def main():
    """Main test runner function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Training Integration Tests")
    
    # Model configuration
    parser.add_argument('--model_path', type=str, 
                       default="/ephemeral/models/Llama-3.2-3B-Instruct",
                       help='Path to model to test')
    
    # Test configuration
    parser.add_argument('--test_type', type=str, default='all',
                       choices=['all', 'training_only', 'eval_only'],
                       help='Type of test to run')
    parser.add_argument('--num_fingerprints', type=int, default=2,
                       help='Number of fingerprints for training-only test')
    parser.add_argument('--num_fingerprints_eval', type=int, default=8,
                       help='Number of fingerprints for evaluation test')
    parser.add_argument('--max_key_length', type=int, default=16,
                       help='Maximum key length')
    parser.add_argument('--max_response_length', type=int, default=7,
                       help='Maximum response length')
    parser.add_argument('--num_epochs', type=int, default=2,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Training batch size')
    parser.add_argument('--fingerprint_strategy', type=str, default='random_words',
                       help='Fingerprint generation strategy')
                       
    # Evaluation configuration
    parser.add_argument('--eval_tasks', type=str, default='ifeval',
                       help='Evaluation tasks to run')
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate every N epochs')
    parser.add_argument('--eval_lm_batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--eval_lm_limit', type=int, default=10,
                       help='Limit for evaluation samples')
    parser.add_argument('--eval_num_fewshot', type=int, default=0,
                       help='Number of few-shot examples')
    parser.add_argument('--timeout_minutes', type=int, default=10,
                       help='Timeout in minutes for evaluation test')
                       
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='test_logs',
                       help='Directory to save test results')
    parser.add_argument('--use_bash_script_config', action='store_true',
                       help='Use configuration matching the original bash scripts')
    
    args = parser.parse_args()
    
    # Create configuration
    if args.use_bash_script_config:
        config = create_config_from_bash_script_params()
        logger.info("Using configuration matching original bash scripts")
    else:
        config = vars(args)
        
    # Initialize test suite
    test_suite = TrainingTestSuite(args.output_dir)
    
    # Run tests based on test_type
    if args.test_type == 'all':
        results = test_suite.run_all_tests(config)
    elif args.test_type == 'training_only':
        results = {'training_only': test_suite.run_training_only_test(config)}
    elif args.test_type == 'eval_only':
        # Use more fingerprints for evaluation test
        config['num_fingerprints'] = config.get('num_fingerprints_eval', 8)
        results = {'training_with_eval': test_suite.run_training_with_eval_test(config)}
    
    # Generate and save report
    report = test_suite.generate_report(results, config)
    print("\n" + report)
    
    # Save report to file
    report_path = test_suite.base_output_dir / f"integration_test_report_{test_suite.timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nDetailed report saved to: {report_path}")
    
    # Return exit code
    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main()) 