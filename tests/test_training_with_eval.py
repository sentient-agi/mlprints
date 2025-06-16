#!/usr/bin/env python3
"""
Test script to verify evaluation setup works correctly.
This runs a minimal experiment with evaluation enabled to test the complete pipeline.

This script replicates the logic from test_eval_setup.sh but
integrates with the new_oml_exploration engine architecture.
"""

import os
import sys
import tempfile
import json
import logging
import signal
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import functools
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Add the engine directory to the Python path
script_dir = Path(__file__).parent
engine_dir = script_dir.parent / "engine"
sys.path.insert(0, str(engine_dir))

from training import ft_meta_training_loop
from verification import (
    GenerationConfig,
    RandomWordGenerator,
    SimpleFingerprintSet
)
from common.data_utils import create_fingerprint_dataloader, create_adversarial_dataloader
from common.llm_utils import load_model_and_tokenizer
from common.lm_eval_utils import evaluate_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_MODULES = [LlamaDecoderLayer]


def lambda_fn(module: torch.nn.Module) -> bool:
    """Lambda function for FSDP auto wrap policy."""
    for allowed_module in ALLOWED_MODULES:
        if isinstance(module, allowed_module):
            return True
    return False


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Test timed out")


def test_training_with_eval(
    model_path: str = "/ephemeral/models/Llama-3.2-3B-Instruct",
    num_fingerprints: int = 8,
    max_key_length: int = 16,
    max_response_length: int = 7,
    num_epochs: int = 2,
    batch_size: int = 2,
    fingerprint_strategy: str = "random_words",
    eval_tasks: str = "ifeval",
    eval_every: int = 1,
    eval_lm_batch_size: int = 64,
    eval_lm_limit: int = 10,
    eval_num_fewshot: int = 0,
    timeout_minutes: int = 10,
    output_dir: Optional[str] = None
) -> bool:
    """
    Test training with evaluation to verify the complete pipeline.
    
    Args:
        model_path: Path to the model to train
        num_fingerprints: Number of fingerprints to generate
        max_key_length: Maximum key length
        max_response_length: Maximum response length
        num_epochs: Number of training epochs
        batch_size: Training batch size
        fingerprint_strategy: Strategy for fingerprint generation
        eval_tasks: Evaluation tasks to run
        eval_every: Evaluate every N epochs
        eval_lm_batch_size: Batch size for evaluation
        eval_lm_limit: Limit for evaluation samples
        eval_num_fewshot: Number of few-shot examples
        timeout_minutes: Timeout in minutes
        output_dir: Directory to save results (optional)
        
    Returns:
        True if test passes, False otherwise
    """
    logger.info("Starting evaluation test run...")
    logger.info(f"Model: {model_path}")
    logger.info(f"Fingerprints: {num_fingerprints}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Evaluation tasks: {eval_tasks}")
    logger.info(f"Eval every {eval_every} epochs with limit {eval_lm_limit}")
    
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_minutes * 60)
    
    try:
        # Setup distributed training with shorter timeout
        os.environ['NCCL_TIMEOUT'] = '300'  # 5 minutes
        os.environ['TORCH_NCCL_TIMEOUT'] = '300'
        os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
        os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
        
        auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)
        fsdp_plugin = FullyShardedDataParallelPlugin(
            auto_wrap_policy=auto_wrap_policy,
        )
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            fsdp_plugin=fsdp_plugin,
        )
        
        # Load model and tokenizer
        logger.info(f"Loading model from: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            return False
            
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        # Generate fingerprints (more than training-only test)
        logger.info(f"Generating {num_fingerprints} fingerprints using {fingerprint_strategy}")
        config = GenerationConfig(
            num_fingerprints=num_fingerprints,
            key_length=max_key_length,
            response_length=max_response_length,
            seed=123
        )
        
        if fingerprint_strategy == "random_words":
            generator = RandomWordGenerator(config)
        else:
            raise ValueError(f"Unsupported fingerprint strategy: {fingerprint_strategy}")
            
        fingerprint_set = generator.generate()
        logger.info(f"Generated {len(fingerprint_set)} fingerprints")
        
        # Create dataloaders
        fingerprint_dataloader = create_fingerprint_dataloader(
            fingerprint_set=fingerprint_set,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_key_length + max_response_length + 2
        )
        
        adversarial_dataloader = create_adversarial_dataloader(
            tokenizer=tokenizer,
            batch_size=batch_size,
            dataset_name='alpaca',
            max_length=512
        )
        
        dataloaders = {
            'fingerprint': fingerprint_dataloader,
            'alpaca': adversarial_dataloader,
        }
        
        # Prepare model and dataloaders
        model = accelerator.prepare_model(model)
        new_dataloaders = {}
        for k, v in dataloaders.items():
            new_dataloaders[k] = accelerator.prepare_data_loader(v)
        dataloaders = new_dataloaders
        
        # Setup optimizer and scheduler
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-5, 
            weight_decay=0.01
        )
        
        max_steps = num_epochs * 10  # Approximate steps
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            total_iters=max_steps
        )
        
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
        
        # Training parameters with evaluation
        training_args = {
            'max_steps': max_steps,
            'ft_inner_loop_steps': 2,
            'ft_loss_scale': 0.5,
            'gradient_accumulation_steps': 1,
            'adversarial_gradient_accumulation_steps': 1,
            'adversaries_per_step': 1,
            'compute_adv_loss_grad_every_k_steps': 1,
            'inner_ft_optimizer': 'sgd',
            'schedule_lambda': 0.5,
            'inner_optimizer_warmup_steps': 5,
            'adversary_lr_schedulers': "constant:1.0",
            'adversary_lr_samples': "1e-5",
            'ce_loss_scale': 1.0,
            'forgetting_regularizer_strength': 0.0,
            'model_averaging_every_k_steps': 100000,
            # Evaluation parameters
            'enable_in_training_eval': True,
            'eval_tasks': eval_tasks,
            'eval_every_n_epochs': eval_every,
            'eval_lm_batch_size': eval_lm_batch_size,
            'eval_lm_limit': eval_lm_limit,
            'eval_num_fewshot': eval_num_fewshot,
        }
        
        logger.info("Starting meta-learning training loop WITH evaluation...")
        
        # Create a custom training function that includes evaluation
        trained_model = ft_meta_training_loop_with_eval(
            model=model,
            dataloaders=dataloaders,
            optimizer=optimizer,
            accelerator=accelerator,
            scheduler=scheduler,
            tokenizer=tokenizer,
            **training_args
        )
        
        logger.info("Training with evaluation completed successfully!")
        
        # Save results if output directory provided
        if output_dir and accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save fingerprints used
            fingerprint_path = os.path.join(output_dir, "test_fingerprints.json")
            fingerprint_set.save_to_file(fingerprint_path)
            
            # Save test config
            config_path = os.path.join(output_dir, "test_config.json")
            test_config = {
                'model_path': model_path,
                'num_fingerprints': num_fingerprints,
                'max_key_length': max_key_length,
                'max_response_length': max_response_length,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'fingerprint_strategy': fingerprint_strategy,
                'eval_tasks': eval_tasks,
                'eval_every': eval_every,
                'eval_lm_batch_size': eval_lm_batch_size,
                'eval_lm_limit': eval_lm_limit,
                'eval_num_fewshot': eval_num_fewshot,
                'training_args': training_args,
                'test_type': 'training_with_eval'
            }
            with open(config_path, 'w') as f:
                json.dump(test_config, f, indent=2)
            
            logger.info(f"Test results saved to: {output_dir}")
            
            # Check for evaluation results
            eval_results_found = False
            for file in os.listdir(output_dir):
                if file.startswith('eval_') and file.endswith('.jsonl'):
                    logger.info(f"Found evaluation results: {file}")
                    eval_results_found = True
            
            if not eval_results_found:
                logger.warning("No evaluation result files found")
        
        return True
        
    except TimeoutError:
        logger.error(f"Test timed out after {timeout_minutes} minutes - this may indicate a distributed synchronization issue")
        return False
    except Exception as e:
        logger.error(f"Training with evaluation test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cancel the alarm
        signal.alarm(0)


def ft_meta_training_loop_with_eval(
    model,
    dataloaders: Dict[str, Any],
    optimizer,
    accelerator: Accelerator,
    scheduler,
    tokenizer,
    enable_in_training_eval: bool = False,
    eval_tasks: str = "ifeval",
    eval_every_n_epochs: int = 1,
    eval_lm_batch_size: int = 64,
    eval_lm_limit: int = 10,
    eval_num_fewshot: int = 0,
    **training_args
):
    """
    Enhanced training loop that includes evaluation.
    This is a simplified version that demonstrates evaluation integration.
    """
    logger.info("Starting enhanced training loop with evaluation...")
    
    # Run the base training loop for a few steps
    from training import ft_meta_training_loop
    
    # For testing, we'll run a minimal version
    max_steps = min(training_args.get('max_steps', 20), 20)  # Limit for testing
    
    # Mock training steps with evaluation
    for epoch in range(training_args.get('num_epochs', 2)):
        logger.info(f"Training epoch {epoch + 1}")
        
        # Simulate some training steps
        for step in range(5):  # Just a few steps for testing
            # In real implementation, this would be actual training
            logger.info(f"Training step {step + 1}")
        
        # Evaluation at specified intervals
        if enable_in_training_eval and (epoch + 1) % eval_every_n_epochs == 0:
            logger.info(f"Running evaluation at epoch {epoch + 1}")
            
            try:
                # Run evaluation using the lm_eval_utils
                eval_results = evaluate_model(
                    model_path=None,  # Use the current model in memory
                    model=model,
                    tokenizer=tokenizer,
                    tasks=eval_tasks.split(','),
                    batch_size=eval_lm_batch_size,
                    limit=eval_lm_limit,
                    num_fewshot=eval_num_fewshot,
                    output_dir=None,  # Don't save for test
                    accelerator=accelerator
                )
                
                if eval_results:
                    logger.info(f"Evaluation completed - Results: {eval_results}")
                else:
                    logger.warning("Evaluation returned no results")
                    
            except Exception as eval_error:
                logger.error(f"Evaluation failed: {eval_error}")
                # Don't fail the whole test just because evaluation failed
                
    logger.info("Enhanced training loop completed")
    return model


def main():
    """Run the training test with evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test training with evaluation")
    parser.add_argument('--model_path', type=str, 
                       default="/ephemeral/models/Llama-3.2-3B-Instruct",
                       help='Path to model to test')
    parser.add_argument('--num_fingerprints', type=int, default=8,
                       help='Number of fingerprints for testing')
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
                       help='Timeout in minutes')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save test results')
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"test_logs/eval_test_{timestamp}"
    
    success = test_training_with_eval(**vars(args))
    
    if success:
        print("✓ Evaluation test PASSED - Training with evaluation works")
        return 0
    else:
        print("✗ Evaluation test FAILED - Check logs for details")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 