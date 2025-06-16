#!/usr/bin/env python3
"""
Robust Fingerprint Training with Meta-Learning

This script provides an entry point for training models with robust fingerprints
using meta-learning approaches to defend against adversarial fine-tuning attacks.

Usage Examples:
    # Basic meta-learning training
    python train_robust_fingerprints.py --model_family llama --model_size 3B

    # Task vectors approach
    python train_robust_fingerprints.py --model_family llama --model_size 3B --use_task_vectors

    # Custom parameters
    python train_robust_fingerprints.py \
        --model_family gemma --model_size 2B \
        --num_fingerprints 512 --max_steps 40 \
        --ft_inner_loop_steps 2 --adversarial_gradient_accumulation_steps 4
"""

import argparse
import functools
import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import wandb
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer

# Add the project root to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.training import (
    ft_meta_training_loop,
    task_vectors_training_loop,
    FSDPModelStorage
)
from engine.verification import (
    SimpleFingerprintSet,
    GenerationConfig,
    RandomWordGenerator,
    EnglishTextGenerator
)
from engine.common.data_utils import create_fingerprint_dataloader, create_adversarial_dataloader
from engine.common.llm_utils import load_model_and_tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_MODULES = [
    LlamaDecoderLayer,
    Gemma2DecoderLayer,    
]

RESULT_PATH = Path("results/")


def lambda_fn(module: torch.nn.Module) -> bool:
    """Lambda function for FSDP auto wrap policy."""
    for allowed_module in ALLOWED_MODULES:
        if isinstance(module, allowed_module):
            return True
    return False


def setup_robust_training_run(
    model_family: str = "llama",
    model_size: str = "3B",
    model_path: Optional[str] = None,
    num_fingerprints: int = 1024,
    max_key_length: int = 16,
    max_response_length: int = 1,
    fingerprint_generation_strategy: str = "english",
    fingerprints_file_path: Optional[str] = None,
    data_split: int = 0,
    use_task_vectors: bool = False,
    **training_kwargs
) -> Tuple[Any, Any, Dict[str, Any], Optional[Any], str]:
    """
    Set up a robust training run with meta-learning.
    
    Args:
        model_family: Model family (llama, gemma, etc.)
        model_size: Model size (3B, 7B, etc.)
        model_path: Path to custom model (overrides model_family/size)
        num_fingerprints: Number of fingerprints to generate
        max_key_length: Maximum key length
        max_response_length: Maximum response length
        fingerprint_generation_strategy: Strategy for fingerprint generation
        fingerprints_file_path: Path to pre-generated fingerprints
        data_split: Data split index
        use_task_vectors: Whether to use task vectors approach
        **training_kwargs: Additional training arguments
        
    Returns:
        Tuple of (model, tokenizer, dataloaders, instruction_tuned_model, config_hash)
    """
    # Create configuration
    config = {
        'model_family': model_family,
        'model_size': model_size,
        'model_path': model_path,
        'num_fingerprints': num_fingerprints,
        'max_key_length': max_key_length,
        'max_response_length': max_response_length,
        'fingerprint_generation_strategy': fingerprint_generation_strategy,
        'fingerprints_file_path': fingerprints_file_path,
        'data_split': data_split,
        'use_task_vectors': use_task_vectors,
        **training_kwargs
    }
    
    # Generate config hash for experiment tracking
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    config['config_hash'] = config_hash
    
    # Create results directory
    RESULT_PATH.mkdir(exist_ok=True)
    experiment_dir = RESULT_PATH / f"robust_training_{config_hash}"
    experiment_dir.mkdir(exist_ok=True)
    
    # Save configuration
    with open(experiment_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Log experiment
    with open(RESULT_PATH / "all_experiments.jsonl", 'a') as f:
        f.write(json.dumps(config) + '\n')
    
    # Check if experiment already completed
    if (experiment_dir / "final_model").exists():
        logger.info(f"Experiment already completed: {config_hash}")
        return None, None, None, None, config_hash
    
    # Load model and tokenizer
    if model_path:
        logger.info(f"Loading custom model from: {model_path}")
        model, tokenizer = load_model_and_tokenizer(model_path)
    else:
        logger.info(f"Loading {model_family} {model_size} model")
        model, tokenizer = load_model_and_tokenizer(
            model_family=model_family,
            model_size=model_size
        )
    
    # Generate or load fingerprints
    if fingerprints_file_path and os.path.exists(fingerprints_file_path):
        logger.info(f"Loading fingerprints from: {fingerprints_file_path}")
        fingerprint_set = SimpleFingerprintSet.load_from_file(fingerprints_file_path)
    else:
        logger.info(f"Generating fingerprints using strategy: {fingerprint_generation_strategy}")
        
        if fingerprint_generation_strategy == "english":
            generator = EnglishTextGenerator(
                GenerationConfig(
                    num_fingerprints=num_fingerprints,
                    key_length=max_key_length,
                    response_length=max_response_length,
                    seed=42 + data_split
                )
            )
        else:  # random words
            generator = RandomWordGenerator(
                GenerationConfig(
                    num_fingerprints=num_fingerprints,
                    key_length=max_key_length,
                    response_length=max_response_length,
                    seed=42 + data_split
                )
            )
        
        fingerprint_set = generator.generate()
        
        # Save generated fingerprints
        fingerprint_path = experiment_dir / "fingerprints.json"
        fingerprint_set.save_to_file(str(fingerprint_path))
        logger.info(f"Saved fingerprints to: {fingerprint_path}")
    
    # Create dataloaders
    fingerprint_dataloader = create_fingerprint_dataloader(
        fingerprint_set=fingerprint_set,
        tokenizer=tokenizer,
        batch_size=training_kwargs.get('batch_size', 8),
        max_length=max_key_length + max_response_length + 2
    )
    
    adversarial_dataloader = create_adversarial_dataloader(
        tokenizer=tokenizer,
        batch_size=training_kwargs.get('inner_batch_size', 2),
        dataset_name=training_kwargs.get('finetuning_dataset', 'alpaca'),
        max_length=512
    )
    
    dataloaders = {
        'fingerprint': fingerprint_dataloader,
        'alpaca': adversarial_dataloader,
    }
    
    # Load instruction-tuned model for task vectors if needed
    model_it = None
    if use_task_vectors:
        logger.info("Loading instruction-tuned model for task vectors")
        if model_family == 'llama':
            model_it, _ = load_model_and_tokenizer(
                model_family=model_family,
                model_size=model_size,
                variant="Instruct"
            )
        elif model_family == 'gemma':
            model_it, _ = load_model_and_tokenizer(
                model_family=model_family,
                model_size=model_size,
                variant="it"
            )
        else:
            raise ValueError(f"Task vectors not supported for model family: {model_family}")
    
    return model, tokenizer, dataloaders, model_it, config_hash


def train_robust_model(
    model,
    tokenizer,
    dataloaders: Dict[str, Any],
    accelerator: Accelerator,
    config_hash: str,
    model_it=None,
    use_task_vectors: bool = False,
    **training_args
):
    """
    Train a robust fingerprint model using meta-learning.
    
    Args:
        model: The model to train
        tokenizer: Tokenizer for the model
        dataloaders: Dictionary of dataloaders
        accelerator: Accelerator instance
        config_hash: Configuration hash for saving
        model_it: Instruction-tuned model (for task vectors)
        use_task_vectors: Whether to use task vectors approach
        **training_args: Training arguments
    """
    logger.info("Setting up training")
    
    # Prepare model
    model = accelerator.prepare_model(model)
    if model_it:
        model_it = accelerator.prepare_model(model_it)
    
    # Prepare dataloaders
    new_dataloaders = {}
    for k, v in dataloaders.items():
        new_dataloaders[k] = accelerator.prepare_data_loader(v)
    dataloaders = new_dataloaders
    
    # Setup optimizer and scheduler
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=training_args.get('learning_rate', 1e-5), 
        weight_decay=training_args.get('weight_decay', 0.01)
    )
    
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        total_iters=training_args.get('max_steps', 1000)
    )
    
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    
    logger.info(f"Starting {'task vectors' if use_task_vectors else 'meta-learning'} training")
    
    # Run appropriate training loop
    if use_task_vectors:
        model = task_vectors_training_loop(
            model=model,
            dataloaders=dataloaders,
            optimizer=optimizer,
            accelerator=accelerator,
            scheduler=scheduler,
            tokenizer=tokenizer,
            model_tv=model_it,
            **training_args
        )
    else:
        model = ft_meta_training_loop(
            model=model,
            dataloaders=dataloaders,
            optimizer=optimizer,
            accelerator=accelerator,
            scheduler=scheduler,
            tokenizer=tokenizer,
            **training_args
        )
    
    # Save final model
    output_dir = RESULT_PATH / f"robust_training_{config_hash}" / "final_model"
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            full_state_dict = model.state_dict()
        
        accelerator.unwrap_model(model).save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=full_state_dict,
            safe_serialization=True
        )
        
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved model to: {output_dir}")
        
        # Log experiment completion
        with open("completed_experiments.txt", 'a') as f:
            f.write(f"{config_hash}\n")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Robust Fingerprint Training with Meta-Learning")
    
    # Model configuration
    parser.add_argument('--model_family', type=str, default='llama', 
                       choices=['llama', 'gemma', 'mistral', 'microsoft'],
                       help='Model family to use')
    parser.add_argument('--model_size', type=str, default='3B', 
                       help='Model size (e.g., 3B, 7B, 8B)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to custom model (overrides family/size)')
    
    # Fingerprint configuration
    parser.add_argument('--num_fingerprints', type=int, default=1024,
                       help='Number of fingerprints to generate/use')
    parser.add_argument('--max_key_length', type=int, default=16,
                       help='Maximum length of fingerprint keys')
    parser.add_argument('--max_response_length', type=int, default=1,
                       help='Maximum length of fingerprint responses')
    parser.add_argument('--fingerprint_generation_strategy', type=str, default='english',
                       choices=['english', 'random_words'],
                       help='Strategy for generating fingerprints')
    parser.add_argument('--fingerprints_file_path', type=str, default=None,
                       help='Path to pre-generated fingerprints file')
    parser.add_argument('--data_split', type=int, default=0,
                       help='Data split index for reproducibility')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for outer loop')
    parser.add_argument('--inner_batch_size', type=int, default=2,
                       help='Batch size for inner loop')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for optimizer')
    
    # Meta-learning specific arguments
    parser.add_argument('--max_steps', type=int, default=100,
                       help='Maximum number of training steps')
    parser.add_argument('--ft_inner_loop_steps', type=int, default=4,
                       help='Number of inner loop fine-tuning steps')
    parser.add_argument('--ft_loss_scale', type=float, default=0.5,
                       help='Scaling factor for fingerprint loss')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Gradient accumulation steps for outer loop')
    parser.add_argument('--adversarial_gradient_accumulation_steps', type=int, default=2,
                       help='Gradient accumulation steps for adversarial training')
    parser.add_argument('--adversaries_per_step', type=int, default=1,
                       help='Number of adversaries per training step')
    parser.add_argument('--compute_adv_loss_grad_every_k_steps', type=int, default=1,
                       help='Frequency of computing adversarial loss gradients')
    parser.add_argument('--inner_ft_optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam'],
                       help='Optimizer for inner loop fine-tuning')
    parser.add_argument('--schedule_lambda', type=float, default=0.5,
                       help='Lambda parameter for scheduling')
    parser.add_argument('--inner_optimizer_warmup_steps', type=int, default=20,
                       help='Warmup steps for inner optimizer')
    parser.add_argument('--use_weighting_schedule', action='store_true',
                       help='Use weighting schedule for training')
    parser.add_argument('--adversary_lr_schedulers', type=str, 
                       default="constant:1.0,linear_warmup:0.25",
                       help='Adversary learning rate schedulers')
    parser.add_argument('--adversary_lr_samples', type=str, default="1e-5",
                       help='Learning rate samples for adversary')
    parser.add_argument('--ce_loss_scale', type=float, default=1.0,
                       help='Cross-entropy loss scaling factor')
    parser.add_argument('--forgetting_regularizer_strength', type=float, default=0.0,
                       help='Strength of forgetting regularizer')
    parser.add_argument('--model_averaging_every_k_steps', type=int, default=100000,
                       help='Model averaging frequency')
    parser.add_argument('--finetuning_dataset', type=str, default='alpaca',
                       help='Dataset for adversarial fine-tuning')
    
    # Task vectors arguments
    parser.add_argument('--use_task_vectors', action='store_true',
                       help='Use task vectors approach instead of meta-learning')
    parser.add_argument('--task_vectors_coefficients', type=str, default="1.0",
                       help='Task vector coefficients (comma-separated)')
    
    # Logging and evaluation
    parser.add_argument('--wandb_project', type=str, default='robust_fingerprinting',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Weights & Biases run name')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Setup distributed training
    auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)
    fsdp_plugin = FullyShardedDataParallelPlugin(
        auto_wrap_policy=auto_wrap_policy,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fsdp_plugin=fsdp_plugin,
    )
    
    # Setup Weights & Biases
    if accelerator.is_main_process:
        wandb_run_name = args.wandb_run_name or f"robust_{args.model_family}_{args.model_size}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config=vars(args)
        )
    
    accelerator.print("Starting robust fingerprint training")
    
    # Setup training run
    setup_result = setup_robust_training_run(**vars(args))
    if setup_result[0] is None:  # Experiment already completed
        accelerator.print(f"Experiment already completed: {setup_result[-1]}")
        return
    
    model, tokenizer, dataloaders, model_it, config_hash = setup_result
    
    # Train the model
    train_robust_model(
        model=model,
        tokenizer=tokenizer,
        dataloaders=dataloaders,
        accelerator=accelerator,
        config_hash=config_hash,
        model_it=model_it,
        **vars(args)
    )
    
    accelerator.print("Training completed successfully!")


if __name__ == "__main__":
    main() 