"""
Meta-Learning Training Loops for Robust Fingerprint Training

This module contains the core training loops for making fingerprints robust
against adversarial fine-tuning attacks using meta-learning approaches.
"""

import random
from typing import List, Union, Dict, Any

import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer, move_to_device
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .training_utils import (
    FSDPModelStorage, 
    apply_task_vector, 
    prepare_task_vectors, 
    obj_standard_max_next_token, 
    next_n_batches, 
    delete_optimizer, 
    distributed_sample_adversary_lr, 
    distributed_sample_task,
    schedule
)


def inner_loop_step_ft(
    model: Union[AutoModelForCausalLM, FSDP],
    adversary_batches: List[torch.Tensor],
    fingerprint_batches: List[Dict[str, torch.Tensor]],
    inner_optimizer: AcceleratedOptimizer,
    inner_scheduler: torch.optim.lr_scheduler.LRScheduler,
    accelerator: Accelerator,
    gradient_accumulation_steps: int = 8,
    adv_gradient_accumulation_steps: int = 8,
    ft_grad_scale: float = 0.25,
    model_storage: FSDPModelStorage = None,
    sub_pbar: tqdm = None,
    compute_tamper_resistance_grad: bool = True,
) -> float:
    """
    Perform a single inner loop step in the tamper-resistant training process.

    This function executes an adversarial step, updates the model, and optionally computes
    the tamper resistance gradient.

    Args:
        model: The model being trained
        adversary_batches: Batches of adversarial data
        fingerprint_batches: Batches of fingerprint data
        inner_optimizer: The optimizer for the inner loop
        inner_scheduler: The learning rate scheduler
        accelerator: The Hugging Face Accelerator object
        gradient_accumulation_steps: Number of steps to accumulate gradients
        adv_gradient_accumulation_steps: Number of adversarial gradient accumulation steps
        ft_grad_scale: Scaling factor for fingerprint gradients
        model_storage: Storage for FSDP model parameters and gradients
        sub_pbar: Progress bar for sub-steps
        compute_tamper_resistance_grad: Whether to compute tamper resistance gradient

    Returns:
        The computed tamper resistance loss
    """

    # Adversary fine-tuning the model
    total_adv_loss = 0
    for i in range(adv_gradient_accumulation_steps):
        output = model(**adversary_batches[i])
        loss = output.loss
        loss = loss / adv_gradient_accumulation_steps
        accelerator.backward(loss)
        total_adv_loss += loss.item()
    
    if accelerator.is_main_process and sub_pbar:
        sub_pbar.update(1)
        sub_pbar.set_postfix({"adv loss": total_adv_loss})
        if wandb.run is not None:
            wandb.log({"inner_ft_loss": total_adv_loss})

    inner_optimizer.step()
    if inner_scheduler:
        inner_scheduler.step()
    model.zero_grad(set_to_none=True)

    # Compute tamper-resistance loss
    ft_loss = 0
    if compute_tamper_resistance_grad:
        total_loss = 0.0        
        for i in range(gradient_accumulation_steps):
            output = model(**fingerprint_batches[i])
            loss = output.loss
            loss = loss * ft_grad_scale

            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            total_loss += loss.item()
        
        # Accumulate sharded TR loss grads in FSDPModelStorage data structure
        model_storage.collect_param_or_grad(
            model=model,
            accelerator=accelerator,
            to_cpu=True,
            mode="grads",
        )
        # Clear grads from model to be ready for next adversary step
        model.zero_grad(set_to_none=False)
        ft_loss = total_loss
        
        if accelerator.is_main_process and wandb.run is not None:
            wandb.log({
                "inner_adv_ft_loss": total_loss / ft_grad_scale,
            })

    return ft_loss


def ft_meta_training_loop(
    model: Union[AutoModelForCausalLM, FSDP],
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    optimizer: AcceleratedOptimizer,
    accelerator: Accelerator,
    scheduler,
    tokenizer: AutoTokenizer,
    gradient_accumulation_steps: int = 2,   
    max_steps: int = 1000,
    ft_inner_loop_steps: int = 4,
    ft_loss_scale: float = 4.0,
    inner_ft_optimizer: str = "sgd",
    schedule_lambda: float = 0.5,
    inner_optimizer_warmup_steps: int = 20,
    use_weighting_schedule: bool = False,
    adversary_lr_schedulers: str = "constant:1.0,linear_warmup:0.25",
    adversary_lr_samples: str = "1e-5",
    adversaries_per_step: int = 1,
    compute_adv_loss_grad_every_k_steps: int = 1,
    adv_gradient_accumulation_steps: int = 2,
    ce_loss_scale: float = 1.0,
    forgetting_regularizer_strength: float = 0.0,
    model_averaging_every_k_steps: int = 100000,
    **kwargs,
) -> Union[AutoModelForCausalLM, FSDP]:
    """
    Main meta-learning training loop for robust fingerprint training.
    
    This function implements the meta-learning approach where the model is trained
    to maintain fingerprint accuracy even after adversarial fine-tuning attacks.
    
    Args:
        model: The model to train
        dataloaders: Dictionary of dataloaders
        optimizer: Main optimizer for the outer loop
        accelerator: Accelerator instance
        scheduler: Learning rate scheduler
        tokenizer: Tokenizer for the model
        gradient_accumulation_steps: Gradient accumulation steps for outer loop
        max_steps: Maximum number of training steps
        ft_inner_loop_steps: Number of inner loop fine-tuning steps
        ft_loss_scale: Scaling factor for fingerprint loss
        inner_ft_optimizer: Optimizer type for inner loop ("sgd" or "adam")
        schedule_lambda: Lambda parameter for scheduling
        inner_optimizer_warmup_steps: Warmup steps for inner optimizer
        use_weighting_schedule: Whether to use weighting schedule
        adversary_lr_schedulers: Learning rate scheduler configuration
        adversary_lr_samples: Learning rate samples for adversary
        adversaries_per_step: Number of adversaries per step
        compute_adv_loss_grad_every_k_steps: Compute adversarial loss gradient frequency
        adv_gradient_accumulation_steps: Gradient accumulation for adversarial steps
        ce_loss_scale: Cross-entropy loss scaling
        forgetting_regularizer_strength: Strength of forgetting regularizer
        model_averaging_every_k_steps: Model averaging frequency
        **kwargs: Additional arguments
    
    Returns:
        Trained model
    """
    
    model.config.use_cache = False
    model.train()

    # Setup data iterators
    fingerprint_iterator, fingerprint_dataloader = (
        iter(dataloaders["fingerprint"]),
        dataloaders["fingerprint"],
    )

    # Adversary dataloaders use the remaining keys
    adversary_dataloaders = {
        key: {"iter": iter(value), "dataloader": value}
        for key, value in dataloaders.items()
        if key not in ["fingerprint"]
    }

    if accelerator.is_main_process:
        pbar = tqdm(
            colour="green",
            desc="Outer Training Loop",
            total=max_steps,
            dynamic_ncols=True,
        )

    model_storage = FSDPModelStorage()
    adversary_lr_samples = [float(lr) for lr in adversary_lr_samples.split(",")]

    # FSDP requires initial forward/backward for sharded params to be accessible
    accelerator.backward(
        obj_standard_max_next_token(model, next(fingerprint_iterator), accelerator)
    )
    model.zero_grad(set_to_none=False)

    if forgetting_regularizer_strength > 0:
        model_storage.store_original_model(model)
    
    # Main training loop
    for train_step in range(max_steps):
        adv_ft_loss = 0
        
        # Save params for tamper-resistance optimizer step
        model_storage.collect_param_or_grad(
            model=model,
            accelerator=accelerator,
            to_cpu=True,
            mode="params",
        )
        
        outer_retain_batches, fingerprint_iterator = next_n_batches(
            fingerprint_iterator, fingerprint_dataloader, gradient_accumulation_steps
        )
        
        optimizer.load_state_dict(move_to_device(optimizer.state_dict(), "cpu"))
        torch.cuda.empty_cache()

        # Inner loop with adversaries
        for _ in range(adversaries_per_step):
            sub_pbar = None
            if accelerator.is_main_process:
                sub_pbar = tqdm(
                    colour="blue",
                    desc=f"Step {train_step}: Inner Training Loop",
                    total=ft_inner_loop_steps,
                    dynamic_ncols=True,
                )
            
            adversary_lr_scheduler = distributed_sample_task(adversary_lr_schedulers)
            adversary_lr = distributed_sample_adversary_lr(adversary_lr_samples, accelerator)

            # Setup adversary optimizer
            if inner_ft_optimizer == "adam":
                inner_optimizer = torch.optim.AdamW(model.parameters(), lr=adversary_lr)
            elif inner_ft_optimizer == "sgd":
                inner_optimizer = torch.optim.SGD(model.parameters(), lr=adversary_lr)
            else:
                raise ValueError("Invalid inner optimizer; must be 'adam' or 'sgd'")
            
            inner_optimizer = accelerator.prepare_optimizer(inner_optimizer)
            inner_scheduler = None

            # Setup adversary learning rate scheduler
            if adversary_lr_scheduler == "linear_warmup":
                inner_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    inner_optimizer,
                    lambda step: ft_inner_loop_steps / inner_optimizer_warmup_steps,
                )
                inner_scheduler = accelerator.prepare(inner_scheduler)

            # Inner loop steps
            for inner_step in range(ft_inner_loop_steps):
                # Sample adversary batches
                _adversary_type = "alpaca"  # Could be parameterized
                (
                    adversary_batches,
                    adversary_dataloaders[_adversary_type]["iter"],
                ) = next_n_batches(
                    adversary_dataloaders[_adversary_type]["iter"],
                    adversary_dataloaders[_adversary_type]["dataloader"],
                    adv_gradient_accumulation_steps,
                )

                # Per-step tamper-resistance loss weighting schedule
                scheduled_weighting = (
                    schedule(inner_step, ft_inner_loop_steps, schedule_lambda)
                    if use_weighting_schedule
                    else 1 / ft_inner_loop_steps
                )

                # Whether to compute TR grad for current step
                compute_tamper_resistance_grad = (
                    inner_step + 1
                ) % compute_adv_loss_grad_every_k_steps == 0

                # Compute adversary step and tamper-resistance loss
                adv_ft_loss += inner_loop_step_ft(
                    model=model,
                    adversary_batches=adversary_batches,
                    fingerprint_batches=outer_retain_batches,
                    inner_optimizer=inner_optimizer,
                    inner_scheduler=inner_scheduler,
                    accelerator=accelerator,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    adv_gradient_accumulation_steps=adv_gradient_accumulation_steps,
                    ft_grad_scale=ft_loss_scale * scheduled_weighting,
                    model_storage=model_storage,
                    sub_pbar=sub_pbar,
                    compute_tamper_resistance_grad=compute_tamper_resistance_grad,
                )

            # Clean up inner optimizer
            delete_optimizer(inner_optimizer)
            
            # Restore original parameters
            model_storage.add_from_storage_to_model(
                model=model,
                accelerator=accelerator,
                skip_check=True,
                mode="params",
            )
            torch.cuda.empty_cache()
            
        # Restore optimizer
        optimizer.load_state_dict(
            move_to_device(optimizer.state_dict(), optimizer.accelerator_state.device)
        )

        # Compute fingerprint retention loss
        total_retain_loss = 0
        for i in range(gradient_accumulation_steps):
            outputs = model(**outer_retain_batches[i])
            retain_loss = outputs.loss
            retain_loss = retain_loss / gradient_accumulation_steps * ce_loss_scale
            accelerator.backward(retain_loss)
            total_retain_loss += retain_loss.item()
            
        # Add tamper-resistance gradients to model
        model_storage.add_from_storage_to_model(
            model=model,
            accelerator=accelerator,
            mode="grads",
            skip_check=ft_inner_loop_steps==0,  
        )

        # Clear from storage to reduce peak memory usage
        model_storage.clear_grads()
        model_storage.clear_params()

        # Tamper-resistance meta-optimizer step
        optimizer.step()
        scheduler.step()
        model.zero_grad(set_to_none=True)
        
        # Model averaging for forgetting regularization
        if (forgetting_regularizer_strength > 0 and 
            train_step % model_averaging_every_k_steps == 0 and 
            train_step > 0):
            model_storage.merge_original_model(model, forgetting_regularizer_strength)
        
        # Logging and progress update
        if accelerator.is_main_process:
            pbar.update(1)
            pbar.set_postfix({
                "fingerprinting loss / adv_ft_loss": f"{total_retain_loss:.4f} / {adv_ft_loss:.4f}"
            })
            
            if wandb.run is not None:
                wandb.log({
                    "step": train_step,
                    "fingerprint_loss": total_retain_loss / ce_loss_scale,
                    "adv_ft_loss": adv_ft_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                })

    print("Meta-learning training completed")
    return model


def inner_step_task_vectors(
    model: Union[AutoModelForCausalLM, FSDP],
    task_vectors: Dict[str, torch.Tensor],
    task_vector_coefficients: List[float],
    fingerprint_batches: List[Dict[str, torch.Tensor]],
    accelerator: Accelerator,
    gradient_accumulation_steps: int = 8,
    ft_grad_scale: float = 0.25,
    model_storage: FSDPModelStorage = None,
) -> float:
    """
    Perform a single inner step using task vectors.
    
    Args:
        model: The model being trained
        task_vectors: Dictionary of task vector parameters
        task_vector_coefficients: List of coefficients for task vectors
        fingerprint_batches: Batches of fingerprint data
        accelerator: Accelerator instance
        gradient_accumulation_steps: Number of gradient accumulation steps
        ft_grad_scale: Scaling factor for fingerprint gradients
        model_storage: Storage for model parameters and gradients
        
    Returns:
        Total task vector loss
    """
    total_loss = 0.0
    
    for tv_idx in range(len(task_vector_coefficients)):
        # Restore original model parameters
        model_storage.add_from_storage_to_model(
            model=model,
            accelerator=accelerator,
            skip_check=True,
            mode="params",
        )
        
        task_vector_coefficient = task_vector_coefficients[tv_idx]
        
        # Apply task vector
        model = apply_task_vector(model, task_vectors, task_vector_coefficient)
        
        # Compute loss with task vector applied
        for i in range(gradient_accumulation_steps):
            output = model(**fingerprint_batches[i])
            loss = output.loss
            loss = loss * ft_grad_scale
            loss = loss / (gradient_accumulation_steps * len(task_vector_coefficients))
            accelerator.backward(loss)
            total_loss += loss.item()

        # Collect gradients
        model_storage.collect_param_or_grad(
            model=model,
            accelerator=accelerator,
            to_cpu=True,
            mode="grads",
        )
        model.zero_grad(set_to_none=False)
    
    return total_loss


def task_vectors_training_loop(
    model: Union[AutoModelForCausalLM, FSDP],
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    optimizer: AcceleratedOptimizer,
    accelerator: Accelerator,
    scheduler,
    tokenizer: AutoTokenizer,
    model_tv: Union[AutoModelForCausalLM, FSDP],
    task_vectors_coefficients: Union[str, List[float]],
    gradient_accumulation_steps: int = 2,   
    max_steps: int = 1000,
    ft_loss_scale: float = 4.0,
    ce_loss_scale: float = 1.0,
    **kwargs,
) -> Union[AutoModelForCausalLM, FSDP]:
    """
    Training loop using task vectors approach for robust fingerprinting.
    
    Args:
        model: Base model to train
        dataloaders: Dictionary of dataloaders
        optimizer: Main optimizer
        accelerator: Accelerator instance
        scheduler: Learning rate scheduler
        tokenizer: Tokenizer
        model_tv: Instruction-tuned model for task vectors
        task_vectors_coefficients: Coefficients for task vectors
        gradient_accumulation_steps: Gradient accumulation steps
        max_steps: Maximum training steps
        ft_loss_scale: Fingerprint loss scaling
        ce_loss_scale: Cross-entropy loss scaling
        **kwargs: Additional arguments
        
    Returns:
        Trained model
    """
    model.config.use_cache = False
    model.train()

    # Setup data iterators
    fingerprint_iterator, fingerprint_dataloader = (
        iter(dataloaders["fingerprint"]),
        dataloaders["fingerprint"],
    )

    if accelerator.is_main_process:
        pbar = tqdm(
            colour="green",
            desc="Task Vectors Training Loop",
            total=max_steps,
            dynamic_ncols=True,
        )

    model_storage = FSDPModelStorage()

    # FSDP initialization
    accelerator.backward(
        obj_standard_max_next_token(model, next(fingerprint_iterator), accelerator)
    )
    accelerator.backward(
        obj_standard_max_next_token(model_tv, next(fingerprint_iterator), accelerator)
    )
    model_tv.zero_grad(set_to_none=False)
    model.zero_grad(set_to_none=False)
    
    accelerator.print("Preparing task vectors")
    task_vectors = prepare_task_vectors(model, model_tv, model_storage)
    
    if isinstance(task_vectors_coefficients, str):
        task_vectors_coefficients = [float(tv) for tv in task_vectors_coefficients.split(",")]
    
    # Main training loop
    for train_step in range(max_steps):
        # Store model parameters
        model_storage.collect_param_or_grad(
            model=model,
            accelerator=accelerator,
            to_cpu=True,
            mode="params",
        )
        
        outer_retain_batches, fingerprint_iterator = next_n_batches(
            fingerprint_iterator, fingerprint_dataloader, gradient_accumulation_steps
        )
        
        optimizer.load_state_dict(move_to_device(optimizer.state_dict(), "cpu"))
        torch.cuda.empty_cache()

        # Apply task vectors and compute gradients
        adv_ft_loss = inner_step_task_vectors(
            model=model,
            task_vectors=task_vectors,
            task_vector_coefficients=task_vectors_coefficients,
            fingerprint_batches=outer_retain_batches,
            accelerator=accelerator,
            gradient_accumulation_steps=gradient_accumulation_steps,
            ft_grad_scale=ft_loss_scale,
            model_storage=model_storage,
        )
        
        # Restore optimizer and model parameters
        optimizer.load_state_dict(
            move_to_device(optimizer.state_dict(), optimizer.accelerator_state.device)
        )
        model_storage.add_from_storage_to_model(
            model=model,
            accelerator=accelerator,
            skip_check=True,
            mode="params",
        )

        # Compute fingerprint retention loss
        total_retain_loss = 0
        for i in range(gradient_accumulation_steps):
            outputs = model(**outer_retain_batches[i])
            retain_loss = outputs.loss
            retain_loss = retain_loss / gradient_accumulation_steps * ce_loss_scale
            accelerator.backward(retain_loss)
            total_retain_loss += retain_loss.item()

        # Add task vector gradients
        model_storage.add_from_storage_to_model(
            model=model,
            accelerator=accelerator,
            mode="grads",
            skip_check=False,  
        )

        # Clear storage
        model_storage.clear_grads()
        model_storage.clear_params()

        # Optimizer step
        optimizer.step()
        scheduler.step()
        model.zero_grad(set_to_none=True)
        
        # Logging and progress
        if accelerator.is_main_process:
            pbar.update(1)
            pbar.set_postfix({
                "fingerprinting loss / task_vectors_loss": f"{total_retain_loss:.4f} / {adv_ft_loss:.4f}"
            })
            
            if wandb.run is not None:
                wandb.log({
                    "step": train_step,
                    "fingerprint_loss": total_retain_loss / ce_loss_scale,
                    "task_vectors_loss": adv_ft_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                })

    print("Task vectors training completed")
    return model 