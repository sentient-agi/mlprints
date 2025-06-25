"""
Robust Fingerprint Trainer

This module provides configuration classes and training orchestration for robust
fingerprint training using meta-learning and task vectors approaches.
"""

import functools
import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

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

from verification import (
    FingerprintSet,
    GenerationConfig,
    RandomWordGenerator,
    EnglishTextGenerator,
)
from common.data_utils import (
    create_fingerprint_dataloader,
    create_adversarial_dataloader,
)
from common.llm_utils import load_model_and_tokenizer
from .meta_learning_loops import ft_meta_training_loop, task_vectors_training_loop

logger = logging.getLogger(__name__)

ALLOWED_MODULES = [
    LlamaDecoderLayer,
    Gemma2DecoderLayer,    
]


class MetaLearningConfig:
    """Configuration for meta-learning based robust training."""
    
    def __init__(
        self,
        max_steps: int = 100,
        ft_inner_loop_steps: int = 4,
        ft_loss_scale: float = 0.5,
        gradient_accumulation_steps: int = 8,
        adversarial_gradient_accumulation_steps: int = 2,
        adversaries_per_step: int = 1,
        compute_adv_loss_grad_every_k_steps: int = 1,
        inner_ft_optimizer: str = 'sgd',
        schedule_lambda: float = 0.5,
        inner_optimizer_warmup_steps: int = 20,
        use_weighting_schedule: bool = False,
        adversary_lr_schedulers: str = "constant:1.0,linear_warmup:0.25",
        adversary_lr_samples: str = "1e-5",
        ce_loss_scale: float = 1.0,
        forgetting_regularizer_strength: float = 0.0,
        model_averaging_every_k_steps: int = 100000,
        finetuning_dataset: str = 'alpaca',
        **kwargs
    ):
        self.max_steps = max_steps
        self.ft_inner_loop_steps = ft_inner_loop_steps
        self.ft_loss_scale = ft_loss_scale
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.adversarial_gradient_accumulation_steps = adversarial_gradient_accumulation_steps
        self.adversaries_per_step = adversaries_per_step
        self.compute_adv_loss_grad_every_k_steps = compute_adv_loss_grad_every_k_steps
        self.inner_ft_optimizer = inner_ft_optimizer
        self.schedule_lambda = schedule_lambda
        self.inner_optimizer_warmup_steps = inner_optimizer_warmup_steps
        self.use_weighting_schedule = use_weighting_schedule
        self.adversary_lr_schedulers = adversary_lr_schedulers
        self.adversary_lr_samples = adversary_lr_samples
        self.ce_loss_scale = ce_loss_scale
        self.forgetting_regularizer_strength = forgetting_regularizer_strength
        self.model_averaging_every_k_steps = model_averaging_every_k_steps
        self.finetuning_dataset = finetuning_dataset
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }


class TaskVectorConfig:
    """Configuration for task vectors based robust training."""
    
    def __init__(
        self,
        max_steps: int = 1000,
        ft_loss_scale: float = 4.0,
        gradient_accumulation_steps: int = 2,
        task_vectors_coefficients: Union[str, List[float]] = "1.0",
        ce_loss_scale: float = 1.0,
        **kwargs
    ):
        self.max_steps = max_steps
        self.ft_loss_scale = ft_loss_scale
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.task_vectors_coefficients = task_vectors_coefficients
        self.ce_loss_scale = ce_loss_scale
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }


class RobustFingerprintTrainer:
    """
    Main trainer class for robust fingerprint training.
    
    This class orchestrates the entire training process including model loading,
    fingerprint generation, dataloader creation, and training execution.
    """
    
    def __init__(
        self,
        model_family: str = "llama",
        model_size: str = "3B",
        model_path: Optional[str] = None,
        num_fingerprints: int = 1024,
        max_key_length: int = 16,
        max_response_length: int = 1,
        fingerprint_generation_strategy: str = "english",
        fingerprints_file_path: Optional[str] = None,
        data_split: int = 0,
        batch_size: int = 8,
        inner_batch_size: int = 2,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        result_path: str = "results/",
        wandb_project: str = 'robust_fingerprinting',
        wandb_run_name: Optional[str] = None,
        seed: int = 42,
        **kwargs
    ):
        # Basic configuration
        self.model_family = model_family
        self.model_size = model_size
        self.model_path = model_path
        self.num_fingerprints = num_fingerprints
        self.max_key_length = max_key_length
        self.max_response_length = max_response_length
        self.fingerprint_generation_strategy = fingerprint_generation_strategy
        self.fingerprints_file_path = fingerprints_file_path
        self.data_split = data_split
        self.batch_size = batch_size
        self.inner_batch_size = inner_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.result_path = Path(result_path)
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.seed = seed
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize internal state
        self.model = None
        self.tokenizer = None
        self.model_it = None  # Instruction-tuned model for task vectors
        self.dataloaders = {}
        self.fingerprint_set = None
        self.config_hash = None
        self.experiment_dir = None
        
        # Set random seeds
        self._set_seeds()
        
        # Setup logging
        self._setup_logging()
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
    
    def _setup_logging(self):
        """Set up logging configuration."""
        self.result_path.mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def setup_experiment(self) -> bool:
        """
        Setup experiment directory and check if already completed.
        
        Returns:
            True if experiment should proceed, False if already completed
        """
        # Generate config hash
        config = {
            'model_family': self.model_family,
            'model_size': self.model_size,
            'model_path': self.model_path,
            'num_fingerprints': self.num_fingerprints,
            'max_key_length': self.max_key_length,
            'max_response_length': self.max_response_length,
            'fingerprint_generation_strategy': self.fingerprint_generation_strategy,
            'data_split': self.data_split,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'seed': self.seed,
        }
        
        config_str = json.dumps(config, sort_keys=True)
        self.config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        self.experiment_dir = self.result_path / f"robust_training_{self.config_hash}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Save configuration
        config['config_hash'] = self.config_hash
        with open(self.experiment_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Log experiment
        with open(self.result_path / "all_experiments.jsonl", 'a') as f:
            f.write(json.dumps(config) + '\n')
        
        # Check if experiment already completed
        if (self.experiment_dir / "final_model").exists():
            logger.info(f"Experiment already completed: {self.config_hash}")
            return False
        
        return True
    
    def load_models(self, use_task_vectors: bool = False):
        """Load the base model and optionally instruction-tuned model."""
        if self.model_path:
            logger.info(f"Loading custom model from: {self.model_path}")
            self.model, self.tokenizer = load_model_and_tokenizer(self.model_path)
        else:
            logger.info(f"Loading {self.model_family} {self.model_size} model")
            self.model, self.tokenizer = load_model_and_tokenizer(
                model_family=self.model_family,
                model_size=self.model_size
            )
        
        # Load instruction-tuned model for task vectors if needed
        if use_task_vectors:
            logger.info("Loading instruction-tuned model for task vectors")
            if self.model_family == 'llama':
                self.model_it, _ = load_model_and_tokenizer(
                    model_family=self.model_family,
                    model_size=self.model_size,
                    variant="Instruct"
                )
            elif self.model_family == 'gemma':
                self.model_it, _ = load_model_and_tokenizer(
                    model_family=self.model_family,
                    model_size=self.model_size,
                    variant="it"
                )
            else:
                raise ValueError(f"Task vectors not supported for model family: {self.model_family}")
    
    def setup_fingerprints(self):
        """Generate or load fingerprints."""
        if self.fingerprints_file_path and os.path.exists(self.fingerprints_file_path):
            logger.info(f"Loading fingerprints from: {self.fingerprints_file_path}")
            self.fingerprint_set = FingerprintSet.load_from_file(self.fingerprints_file_path)
        else:
            logger.info(f"Generating fingerprints using strategy: {self.fingerprint_generation_strategy}")
            
            if self.fingerprint_generation_strategy == "english":
                generator = EnglishTextGenerator(
                    GenerationConfig(
                        num_fingerprints=self.num_fingerprints,
                        key_length=self.max_key_length,
                        response_length=self.max_response_length,
                        seed=42 + self.data_split
                    )
                )
            else:  # random words
                generator = RandomWordGenerator(
                    GenerationConfig(
                        num_fingerprints=self.num_fingerprints,
                        key_length=self.max_key_length,
                        response_length=self.max_response_length,
                        seed=42 + self.data_split
                    )
                )
            
            self.fingerprint_set = generator.generate()
            
            # Save generated fingerprints
            fingerprint_path = self.experiment_dir / "fingerprints.json"
            self.fingerprint_set.save_to_file(str(fingerprint_path))
            logger.info(f"Saved fingerprints to: {fingerprint_path}")
    
    def setup_dataloaders(self):
        """Create dataloaders for training."""
        # Create fingerprint dataloader
        fingerprint_dataloader = create_fingerprint_dataloader(
            fingerprint_set=self.fingerprint_set,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            max_length=self.max_key_length + self.max_response_length + 2
        )
        
        # Create adversarial dataloader
        adversarial_dataloader = create_adversarial_dataloader(
            tokenizer=self.tokenizer,
            batch_size=self.inner_batch_size,
            dataset_name=getattr(self, 'finetuning_dataset', 'alpaca'),
            max_length=512
        )
        
        self.dataloaders = {
            'fingerprint': fingerprint_dataloader,
            'alpaca': adversarial_dataloader,
        }
    
    def setup_accelerator(self, gradient_accumulation_steps: int = 8) -> Accelerator:
        """Setup accelerator with FSDP configuration."""
        auto_wrap_policy = functools.partial(
            lambda_auto_wrap_policy, 
            lambda_fn=lambda module: any(isinstance(module, allowed) for allowed in ALLOWED_MODULES)
        )
        
        fsdp_plugin = FullyShardedDataParallelPlugin(
            auto_wrap_policy=auto_wrap_policy,
        )
        
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            fsdp_plugin=fsdp_plugin,
        )
        
        return accelerator
    
    def train_meta_learning(self, config: MetaLearningConfig) -> str:
        """
        Train using meta-learning approach.
        
        Args:
            config: Meta-learning configuration
            
        Returns:
            Path to saved model
        """
        if not self.setup_experiment():
            return str(self.experiment_dir / "final_model")
        
        # Setup components
        self.load_models(use_task_vectors=False)
        self.setup_fingerprints()
        self.setup_dataloaders()
        
        # Setup accelerator
        accelerator = self.setup_accelerator(config.gradient_accumulation_steps)
        
        # Setup Weights & Biases
        if accelerator.is_main_process:
            wandb_run_name = self.wandb_run_name or f"meta_{self.model_family}_{self.model_size}"
            wandb.init(
                project=self.wandb_project,
                name=wandb_run_name,
                config={**config.to_dict(), 'method': 'meta_learning'}
            )
        
        # Prepare models and dataloaders
        self.model = accelerator.prepare_model(self.model)
        
        prepared_dataloaders = {}
        for k, v in self.dataloaders.items():
            prepared_dataloaders[k] = accelerator.prepare_data_loader(v)
        
        # Setup optimizer and scheduler
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            total_iters=config.max_steps
        )
        
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
        
        # Run meta-learning training loop
        logger.info("Starting meta-learning training")
        self.model = ft_meta_training_loop(
            model=self.model,
            dataloaders=prepared_dataloaders,
            optimizer=optimizer,
            accelerator=accelerator,
            scheduler=scheduler,
            tokenizer=self.tokenizer,
            **config.to_dict()
        )
        
        # Save final model
        output_dir = self.experiment_dir / "final_model"
        self._save_model(accelerator, output_dir)
        
        return str(output_dir)
    
    def train_task_vectors(self, config: TaskVectorConfig) -> str:
        """
        Train using task vectors approach.
        
        Args:
            config: Task vectors configuration
            
        Returns:
            Path to saved model
        """
        if not self.setup_experiment():
            return str(self.experiment_dir / "final_model")
        
        # Setup components
        self.load_models(use_task_vectors=True)
        self.setup_fingerprints()
        self.setup_dataloaders()
        
        # Setup accelerator
        accelerator = self.setup_accelerator(config.gradient_accumulation_steps)
        
        # Setup Weights & Biases
        if accelerator.is_main_process:
            wandb_run_name = self.wandb_run_name or f"tv_{self.model_family}_{self.model_size}"
            wandb.init(
                project=self.wandb_project,
                name=wandb_run_name,
                config={**config.to_dict(), 'method': 'task_vectors'}
            )
        
        # Prepare models and dataloaders
        self.model = accelerator.prepare_model(self.model)
        self.model_it = accelerator.prepare_model(self.model_it)
        
        prepared_dataloaders = {}
        for k, v in self.dataloaders.items():
            prepared_dataloaders[k] = accelerator.prepare_data_loader(v)
        
        # Setup optimizer and scheduler
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            total_iters=config.max_steps
        )
        
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
        
        # Run task vectors training loop
        logger.info("Starting task vectors training")
        self.model = task_vectors_training_loop(
            model=self.model,
            dataloaders=prepared_dataloaders,
            optimizer=optimizer,
            accelerator=accelerator,
            scheduler=scheduler,
            tokenizer=self.tokenizer,
            model_tv=self.model_it,
            **config.to_dict()
        )
        
        # Save final model
        output_dir = self.experiment_dir / "final_model"
        self._save_model(accelerator, output_dir)
        
        return str(output_dir)
    
    def _save_model(self, accelerator: Accelerator, output_dir: Path):
        """Save the trained model."""
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                full_state_dict = self.model.state_dict()
            
            accelerator.unwrap_model(self.model).save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=full_state_dict,
                safe_serialization=True
            )
            
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved model to: {output_dir}")
            
            # Log experiment completion
            with open("completed_experiments.txt", 'a') as f:
                f.write(f"{self.config_hash}\n")


def lambda_fn(module: torch.nn.Module) -> bool:
    """Lambda function for FSDP auto wrap policy."""
    for allowed_module in ALLOWED_MODULES:
        if isinstance(module, allowed_module):
            return True
    return False 