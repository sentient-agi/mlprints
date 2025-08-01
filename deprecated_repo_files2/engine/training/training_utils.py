"""
Training Utilities for Robust Fingerprint Training

This module provides utilities for training operations including FSDP model storage,
adversarial sampling, task vectors, model expansion, distributed training utilities,
and other supporting functions for meta-learning based robust fingerprint training.
"""

import logging
import math
import socket
import random
import torch
import torch.distributed as dist
from typing import List, Union, Tuple, Dict, Any, Optional
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM

# Import common utilities
from ..common.llm_utils import filter_inputs, filter_dpo_inputs


# ============================================================================
# FSDP MODEL STORAGE UTILITIES
# ============================================================================

class FSDPModelStorage:
    """
    Storage utility for FSDP model parameters and gradients during meta-learning.
    
    This class provides methods to store, collect, and manage model parameters
    and gradients during the meta-learning training process, especially useful
    for handling FSDP sharded models.
    """
    
    def __init__(self):
        self.params_storage = {}
        self.grads_storage = {}
        self.original_model_params = {}
    
    def collect_param_or_grad(
        self, 
        model: Union[AutoModelForCausalLM, FSDP],
        accelerator: Accelerator,
        to_cpu: bool = True,
        mode: str = "params",
        scale: float = 1.0
    ):
        """
        Collect model parameters or gradients into storage.
        
        Args:
            model: The model to collect from
            accelerator: Accelerator instance
            to_cpu: Whether to move tensors to CPU
            mode: Either "params" or "grads"
            scale: Scaling factor for gradients
        """
        storage = self.params_storage if mode == "params" else self.grads_storage
        
        # Use FSDP-aware parameter iteration if model is FSDP
        if isinstance(model, FSDP):
            for i, param in enumerate(fsdp_v1_model_params(model)):
                if mode == "params":
                    tensor = param.detach().cpu() if to_cpu else param.detach()
                    storage[i] = tensor
                elif mode == "grads" and param.grad is not None:
                    if i not in storage:
                        # Create new gradient entry
                        storage[i] = param.grad.detach() * scale
                    else:
                        # Accumulate gradients
                        storage[i].add_(param.grad.detach().to(storage[i].device) * scale)
                    
                    # Move to CPU if required
                    if to_cpu:
                        storage[i] = storage[i].cpu()
        else:
            # Use standard parameter iteration for non-FSDP models
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if mode == "params":
                        tensor = param.data.clone()
                    elif mode == "grads" and param.grad is not None:
                        tensor = param.grad.data.clone() * scale
                    else:
                        continue
                        
                    if to_cpu:
                        tensor = tensor.cpu()
                    storage[name] = tensor
    
    def add_from_storage_to_model(
        self,
        model: Union[AutoModelForCausalLM, FSDP],
        accelerator: Accelerator,
        mode: str = "params",
        skip_check: bool = False
    ):
        """
        Add stored parameters or gradients back to the model.
        
        Args:
            model: The model to update
            accelerator: Accelerator instance
            mode: Either "params" or "grads"
            skip_check: Skip tensor existence checks
        """
        storage = self.params_storage if mode == "params" else self.grads_storage
        
        # Use FSDP-aware parameter iteration if model is FSDP
        if isinstance(model, FSDP):
            for i, param in enumerate(fsdp_v1_model_params(model)):
                # Check gradient existence assertion if not skipping
                if not skip_check and mode == "grads":
                    try:
                        assert (i in storage) == (param.grad is not None)
                    except AssertionError:
                        print("Grad is none after the inner loop. Ensure that `compute_adv_loss_grad_every_k_steps` has a proper value")
                        raise AssertionError
                
                if i in storage and mode == "params":
                    param.data.copy_(storage[i].to(param.device))
                elif i in storage and param.grad is not None and mode == "grads":
                    param.grad += storage[i].to(param.device)
        else:
            # Use standard parameter iteration for non-FSDP models
            for name, param in model.named_parameters():
                if name in storage and param.requires_grad:
                    stored_tensor = storage[name].to(param.device)
                    
                    if mode == "params":
                        param.data.copy_(stored_tensor)
                    elif mode == "grads":
                        if param.grad is None:
                            param.grad = stored_tensor.clone()
                        else:
                            param.grad.add_(stored_tensor)
    
    def clear_params(self):
        """Clear stored parameters."""
        self.params_storage.clear()
    
    def clear_grads(self):
        """Clear stored gradients."""
        self.grads_storage.clear()
    
    def offload_params_or_grads(self, mode: str = "grads"):
        """
        Offload parameters or gradients from storage to CPU to reduce memory usage.
        
        Args:
            mode: Either "params" or "grads"
        """
        storage = self.params_storage if mode == "params" else self.grads_storage
        
        for key in storage:
            storage[key] = storage[key].cpu()
    
    def store_original_model(self, model: Union[AutoModelForCausalLM, FSDP]):
        """Store the original model parameters for regularization."""
        self.original_model_params.clear()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original_model_params[name] = param.data.clone().cpu()
    
    def merge_original_model(self, model: Union[AutoModelForCausalLM, FSDP], weight: float = 0.25):
        """
        Merge current model with original model parameters.
        
        Args:
            model: Current model
            weight: Weight for original model parameters
        """
        for name, param in model.named_parameters():
            if name in self.original_model_params and param.requires_grad:
                original_param = self.original_model_params[name].to(param.device)
                param.data.mul_(1 - weight).add_(original_param, alpha=weight)


# ============================================================================
# MODEL EXPANSION UTILITIES
# ============================================================================

def verify_expanded_parameters(model, initial_state_dict: Dict[str, torch.Tensor]) -> None:
    """Verify that expanded parameters are correctly maintained."""
    logging.info("Verifying expanded parameters...")
    
    for name, param in model.named_parameters():
        if name in initial_state_dict:
            initial_param = initial_state_dict[name]
            if param.shape != initial_param.shape:
                logging.info(f"Parameter {name} expanded from {initial_param.shape} to {param.shape}")
            else:
                # Check if original parameters are preserved
                if hasattr(param, 'new_weights_start_idx'):
                    start_idx = param.new_weights_start_idx
                    axis = getattr(param, 'expansion_axis', 0)
                    
                    if axis == 0:
                        original_part = param.data[:start_idx, :]
                        expected_part = initial_param.data[:start_idx, :].to(param.device)
                    elif axis == 1:
                        original_part = param.data[:, :start_idx]
                        expected_part = initial_param.data[:, :start_idx].to(param.device)
                    else:
                        continue
                    
                    if torch.allclose(original_part, expected_part, atol=1e-6):
                        logging.info(f"Parameter {name} original weights preserved correctly")
                    else:
                        logging.warning(f"Parameter {name} original weights may have been modified")


def expand_feedforward_weights(model, expansion_rate: float = 0.01):
    """Expand feedforward weights in transformer model for fingerprint insertion."""
    if hasattr(model.config, 'intermediate_size'):
        model.config.intermediate_size = int(model.config.intermediate_size * (1 + expansion_rate))
    
    for _, module in model.named_modules():
        # Handle different model architectures
        is_mlp_module = False
        mlp_attrs = []
        
        # Check for LLaMA-style MLP
        if hasattr(module, 'gate_proj') and hasattr(module, 'up_proj') and hasattr(module, 'down_proj'):
            is_mlp_module = True
            mlp_attrs = ['gate_proj', 'up_proj', 'down_proj']
        
        # Check for other MLP styles (e.g., GPT-style)
        elif hasattr(module, 'c_fc') and hasattr(module, 'c_proj'):
            is_mlp_module = True
            mlp_attrs = ['c_fc', 'c_proj']
        
        # Check for Transformer MLP
        elif hasattr(module, 'dense_h_to_4h') and hasattr(module, 'dense_4h_to_h'):
            is_mlp_module = True
            mlp_attrs = ['dense_h_to_4h', 'dense_4h_to_h']
        
        if is_mlp_module:
            for attr in mlp_attrs:
                if hasattr(module, attr):
                    linear_layer = getattr(module, attr)
                    if isinstance(linear_layer, torch.nn.Linear):
                        if attr in ['gate_proj', 'up_proj', 'c_fc', 'dense_h_to_4h']:
                            # Expand output dimension
                            hidden_dim = linear_layer.weight.size(0)
                            expansion_size = int(hidden_dim * expansion_rate)
                            start_idx = hidden_dim
                            
                            # Initialize new weights with small random values
                            new_weight = torch.cat([
                                linear_layer.weight.data,
                                torch.randn(expansion_size, linear_layer.weight.size(1), 
                                          device=linear_layer.weight.device) * 0.02
                            ], dim=0)
                            linear_layer.weight = torch.nn.Parameter(new_weight)
                            
                            if linear_layer.bias is not None:
                                new_bias = torch.cat([
                                    linear_layer.bias.data,
                                    torch.randn(expansion_size, device=linear_layer.bias.device) * 0.02
                                ], dim=0)
                                linear_layer.bias = torch.nn.Parameter(new_bias)
                            
                            # Store start index and axis
                            linear_layer.new_weights_start_idx = start_idx
                            linear_layer.expansion_axis = 0
                            
                        elif attr in ['down_proj', 'c_proj', 'dense_4h_to_h']:
                            # Expand input dimension
                            hidden_dim = linear_layer.weight.size(1)
                            expansion_size = int(hidden_dim * expansion_rate)
                            start_idx = hidden_dim
                            
                            new_weight = torch.cat([
                                linear_layer.weight.data,
                                torch.randn(linear_layer.weight.size(0), expansion_size, 
                                          device=linear_layer.weight.device) * 0.02
                            ], dim=1)
                            linear_layer.weight = torch.nn.Parameter(new_weight)
                            
                            # Bias might not exist for projection layers
                            linear_layer.new_weights_start_idx = start_idx
                            linear_layer.expansion_axis = 1
                        
    return model


def fsdp_v1_model_params(model: FSDP):
    """
    Get all model parameters via FSDP handles.
    
    This function iterates through FSDP model parameters in a way that's compatible
    with FSDP v1 sharding strategies.
    
    Args:
        model: FSDP wrapped model
        
    Yields:
        Model parameters from FSDP handles
    """
    sharded_params = set()
    nonsharded_params = set()
    
    # Iterate through FSDP handles
    for _, handle in enumerate(model._all_handles):
        target_set = (
            sharded_params if handle.uses_sharded_strategy else nonsharded_params
        )
        target_set.add(handle.flat_param)
        yield handle.flat_param
    
    # Handle non-FSDP managed parameters
    for _, param in model.named_parameters():
        not_fsdp_managed = (
            param not in sharded_params and param not in nonsharded_params
        )
        if not_fsdp_managed:
            nonsharded_params.add(param)
            yield param


# ============================================================================
# DISTRIBUTED TRAINING UTILITIES
# ============================================================================

def smallest_power_of_two(n: int) -> int:
    """Find the smallest power of 2 that is >= n."""
    for i in range(0, 15):
        if 2**i >= n:
            return 2**i
    return 2**14  # Fallback


def get_free_port(start: int = 29550, end: int = 29999) -> int:
    """Get a free TCP port for distributed training."""
    for _ in range(100):
        port = random.randint(start, end)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return end


def warm_up_model_kernels(model, local_rank: int = 0):
    """Warm up model kernels to pre-compile Triton/Flash-Attention kernels."""
    if torch.cuda.is_available():
        try:
            warmup_device = f"cuda:{local_rank}"
            model = model.to(warmup_device)
            
            # Get appropriate token ID for the model
            if hasattr(model.config, 'bos_token_id') and model.config.bos_token_id is not None:
                token_id = model.config.bos_token_id
            elif hasattr(model.config, 'eos_token_id') and model.config.eos_token_id is not None:
                token_id = model.config.eos_token_id
            else:
                token_id = 1  # Fallback
                
            _ = model.generate(
                torch.tensor([[token_id]], device=warmup_device),
                max_new_tokens=1,
            )
            torch.cuda.synchronize()
            logging.info("[warm-up] Model kernels warmed up successfully")
        except Exception as warm_e:
            logging.warning("[warm-up] generate() failed: %s", warm_e)
    return model


def get_distributed_random_number(accelerator: Accelerator) -> float:
    """
    Generate a random number that's synchronized across all distributed processes.
    
    Args:
        accelerator: Accelerator instance for distributed coordination
        
    Returns:
        Synchronized random number across all processes
    """
    random_number = torch.rand(1).to(accelerator.device)
    if dist.is_initialized():
        dist.broadcast(random_number, src=0)
        accelerator.wait_for_everyone()
    return random_number.item()


# ============================================================================
# DEEPSPEED UTILITIES
# ============================================================================

def build_zero2_config(enable_cpu_offload: bool, train_micro_batch_size_per_gpu: int) -> Dict[str, Any]:
    """Build a DeepSpeed configuration dictionary for ZeRO stage 2."""
    config = {
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "bf16": {"enabled": False},
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_param": {
                "device": "cpu" if enable_cpu_offload else "none",
                "pin_memory": enable_cpu_offload,
            },
            "offload_optimizer": {
                "device": "cpu" if enable_cpu_offload else "none",
                "pin_memory": enable_cpu_offload,
            },
        },
    }
    return config


def _no_mode_switch(self, *args, **kw):  # pylint: disable=unused-argument
    """Helper function for DeepSpeed compatibility fixes."""
    return self


def apply_deepspeed_fixes():
    """Apply DeepSpeed compatibility fixes for different ZeRO stages."""
    try:
        # ZeRO-1/2 class
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
        for fn_name in ("train", "eval"):
            if not hasattr(DeepSpeedZeroOptimizer, fn_name):
                setattr(DeepSpeedZeroOptimizer, fn_name, _no_mode_switch)
    except ImportError:
        pass  # Continue if DeepSpeed or specific class not found

    try:
        # ZeRO-3 class
        from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
        for fn_name in ("train", "eval"):
            if not hasattr(DeepSpeedZeroOptimizer_Stage3, fn_name):
                setattr(DeepSpeedZeroOptimizer_Stage3, fn_name, _no_mode_switch)
    except ImportError:
        pass  # Continue if DeepSpeed or specific class not found


# ============================================================================
# TRAINING SCHEDULE AND SAMPLING UTILITIES
# ============================================================================

def schedule(i: int, K: int, schedule_lambda: float = 0.5) -> float:
    """
    Calculate a schedule value based on the current step and total steps.
    
    This function computes an exponential schedule value used for weighting
    or scaling purposes during training.
    
    Args:
        i: The current step or iteration number
        K: The total number of steps or iterations
        schedule_lambda: A scaling factor for the exponent
        
    Returns:
        The computed schedule value as a Python float
    """
    return torch.exp(schedule_lambda * (torch.tensor(i) - (K - 1))).item()


def distributed_sample_adversary_lr(
    adversary_lr_samples: List[float], 
    accelerator: Accelerator
) -> float:
    """
    Sample adversary learning rate in a distributed-aware manner.
    
    Args:
        adversary_lr_samples: List of possible learning rates
        accelerator: Accelerator instance for distributed coordination
        
    Returns:
        Sampled learning rate
    """
    rand_num = get_distributed_random_number(accelerator)
    adversary_lr = adversary_lr_samples[
        math.floor(rand_num * len(adversary_lr_samples))
    ]
    return adversary_lr


def distributed_sample_task(task_string: str) -> str:
    """
    Sample a task from a formatted string in a distributed-aware manner.
    
    Args:
        task_string: Formatted string like "constant:1.0,linear_warmup:0.25"
        
    Returns:
        Sampled task name
    """
    tasks = [task.split(':')[0] for task in task_string.split(',')]
    weights = [float(task.split(':')[1]) for task in task_string.split(',')]
    
    # Weighted random sampling
    total_weight = sum(weights)
    r = random.uniform(0, total_weight)
    cumulative_weight = 0
    
    for task, weight in zip(tasks, weights):
        cumulative_weight += weight
        if r <= cumulative_weight:
            return task
    
    return tasks[0]  # Fallback


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def next_n_batches(
    iterator, 
    dataloader: torch.utils.data.DataLoader, 
    n: int
) -> Tuple[List[Dict[str, torch.Tensor]], any]:
    """
    Get the next n batches from a dataloader iterator.
    
    Args:
        iterator: Current iterator
        dataloader: DataLoader to get batches from
        n: Number of batches to fetch
        
    Returns:
        Tuple of (list of batches, updated iterator)
    """
    batches = []
    
    for _ in range(n):
        try:
            batch = next(iterator)
            batches.append(batch)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
            batches.append(batch)
    
    return batches, iterator


def get_next_batch(iterator, dataloader: torch.utils.data.DataLoader):
    """
    Get the next batch from dataloader iterator, restarting if needed.
    
    Args:
        iterator: Current dataloader iterator
        dataloader: DataLoader instance
        
    Returns:
        Tuple of (batch, updated_iterator)
    """
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        batch = next(iterator)
    return batch, iterator


# ============================================================================
# OPTIMIZER UTILITIES
# ============================================================================

def delete_optimizer(optimizer: AcceleratedOptimizer):
    """
    Properly delete an optimizer to free memory.
    
    Args:
        optimizer: The optimizer to delete
    """
    # Clear optimizer state
    if hasattr(optimizer, 'state_dict'):
        optimizer.state_dict().clear()
    
    # Delete the optimizer
    del optimizer
    
    # Force garbage collection
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# TRAINING OBJECTIVES
# ============================================================================

def obj_standard_max_next_token(
    model: Union[AutoModelForCausalLM, FSDP],
    inputs: Dict[str, torch.Tensor],
    accelerator: Optional[Accelerator] = None,
    chosen: bool = False,
) -> torch.Tensor:
    """
    Compute the standard maximum next token objective.

    This function calculates the log probability loss for the next token prediction
    using the given model and inputs. It supports both standard inputs and
    Direct Preference Optimization (DPO) inputs.

    Args:
        model: The model to use for prediction
        inputs: The input tensors for the model
        accelerator: The Accelerator object for distributed training. Defaults to None.
        chosen: Flag to indicate whether to use chosen or rejected inputs for DPO. Defaults to False.

    Returns:
        The computed log probability loss.
    """
    from ..common.llm_utils import log_p_loss
    
    filtered_inputs = filter_inputs(filter_dpo_inputs(inputs, chosen))
    outputs = model(**filtered_inputs, output_hidden_states=False)
    return log_p_loss(
        outputs.logits,
        filter_dpo_inputs(inputs, chosen).get("labels"),
        model.config.vocab_size,
    )


# ============================================================================
# TASK VECTOR UTILITIES
# ============================================================================

def apply_task_vector(
    model: Union[AutoModelForCausalLM, FSDP],
    task_vectors: Dict[str, torch.Tensor],
    coefficient: float
) -> Union[AutoModelForCausalLM, FSDP]:
    """
    Apply task vectors to a model with given coefficient.
    
    Args:
        model: The base model
        task_vectors: Dictionary of task vector parameters
        coefficient: Scaling coefficient for task vectors
        
    Returns:
        Model with task vectors applied
    """
    for name, param in model.named_parameters():
        if name in task_vectors and param.requires_grad:
            task_vector = task_vectors[name].to(param.device)
            param.data.add_(task_vector, alpha=coefficient)
    
    return model


def prepare_task_vectors(
    base_model: Union[AutoModelForCausalLM, FSDP],
    tuned_model: Union[AutoModelForCausalLM, FSDP],
    model_storage: FSDPModelStorage
) -> Dict[str, torch.Tensor]:
    """
    Prepare task vectors by computing the difference between tuned and base models.
    
    Args:
        base_model: The base model
        tuned_model: The instruction-tuned model
        model_storage: Storage utility for handling parameters
        
    Returns:
        Dictionary of task vector parameters
    """
    task_vectors = {}
    
    base_params = {name: param.data.clone().cpu() for name, param in base_model.named_parameters()}
    tuned_params = {name: param.data.clone().cpu() for name, param in tuned_model.named_parameters()}
    
    for name in base_params:
        if name in tuned_params:
            task_vectors[name] = tuned_params[name] - base_params[name]
    
    return task_vectors


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def compute_fingerprint_accuracy(
    model: Union[AutoModelForCausalLM, FSDP],
    tokenizer,
    fingerprint_dataset,
    device: str = "cuda",
    batch_size: int = 8
) -> Tuple[float, float]:
    """
    Compute fingerprint accuracy for evaluation.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        fingerprint_dataset: Dataset containing fingerprint pairs
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        
    Returns:
        Tuple of (fingerprint_accuracy, fraction_accurate)
    """
    model.eval()
    correct = 0
    total = 0
    
    # This is a simplified version - in practice you'd want more sophisticated evaluation
    with torch.no_grad():
        for i in range(0, len(fingerprint_dataset), batch_size):
            batch_end = min(i + batch_size, len(fingerprint_dataset))
            batch = fingerprint_dataset[i:batch_end]
            
            # Tokenize and evaluate batch
            # Implementation would depend on your specific fingerprint format
            # This is a placeholder for the actual evaluation logic
            total += len(batch)
            # correct += ... (actual evaluation logic)
    
    model.train()
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, accuracy  # Simplified return 