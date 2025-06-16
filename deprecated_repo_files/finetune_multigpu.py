'''
Finetuning script for backdoor attacks and watermarking
'''
import os
os.environ['PYTHONPATH'] = os.getcwd()
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
from old_files.generate_finetuning_data import get_fingerprint_ds, CustomDataCollator, tokenize_function, AugmentedDataset, StraightThroughDataCollator, MixedDataCollator, llama_instruct_tokenize_function, LlamaInstructDataCollator
import lm_eval
import wandb
import json
import hashlib
import logging
import argparse
import contextlib
import os
import math
import datasets
import transformers
from peft import LoraConfig, get_peft_model, PeftModel
# from memory_profiler import profile
from copy import deepcopy
from old_files.utils import expand_feedforward_weights, count_parameters, verify_expanded_parameters
import psutil
import gc
import random
import numpy as np
# --- New imports for async evaluation ---
import subprocess
import sys
import time
import shutil
from typing import Union
import socket
from datetime import datetime

# -----------------------------------------------

class ResetOriginalParametersCallback(TrainerCallback):
    def __init__(self, initial_state_dict):
        self.initial_state_dict = initial_state_dict

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        device = next(model.parameters()).device
        with torch.no_grad():
            for module_name, module in model.named_modules():
                if isinstance(module, transformers.models.llama.modeling_llama.LlamaMLP):
                    for attr in ['gate_proj', 'up_proj', 'down_proj']:
                        linear_layer = getattr(module, attr)
                        if isinstance(linear_layer, torch.nn.Linear):
                            # Construct the full parameter names
                            weight_name = f"{module_name}.{attr}.weight"
                            bias_name = f"{module_name}.{attr}.bias"
                            initial_weight = self.initial_state_dict[weight_name].to(device)
                            if hasattr(linear_layer, 'new_weights_start_idx'):
                                start_idx = linear_layer.new_weights_start_idx
                                axis = linear_layer.expansion_axis
                                if axis == 0:
                                    linear_layer.weight.data[:start_idx, :] = initial_weight.data[:start_idx, :]
                                elif axis == 1:
                                    linear_layer.weight.data[:, :start_idx] = initial_weight.data[:, :start_idx]
                            else:
                                linear_layer.weight.data.copy_(initial_weight.data)
                            if linear_layer.bias is not None and bias_name in self.initial_state_dict:
                                initial_bias = self.initial_state_dict[bias_name].to(device)
                                if hasattr(linear_layer.bias, 'new_weights_start_idx'):
                                    start_idx = linear_layer.bias.new_weights_start_idx
                                    linear_layer.bias.data[:start_idx] = initial_bias.data[:start_idx]
                                else:
                                    linear_layer.bias.data.copy_(initial_bias.data)
            # Reset other parameters
            for name, param in model.named_parameters():
                if 'mlp' not in name:  # Skip MLP layers (already handled)
                    if name in self.initial_state_dict:
                        param.data.copy_(self.initial_state_dict[name].data.to(device))
        # print("Original parameters reset.")



class MemoryCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()      # <-- add this
        gc.collect()
        process = psutil.Process(os.getpid())
        logging.info(f"Memory usage at beginning of epoch {state.epoch}: {process.memory_info().rss / (1024 ** 3):.2f} GB")

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()      # <-- add this
        gc.collect()
        process = psutil.Process(os.getpid())
        logging.info(f"Memory usage at step {state.global_step}: {process.memory_info().rss / (1024 ** 3):.2f} GB")

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()      # <-- add this
        gc.collect()
        process = psutil.Process(os.getpid())
        logging.info(f"Memory usage at step beginning {state.global_step}: {process.memory_info().rss / (1024 ** 3):.2f} GB")

    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()      # <-- add this
        gc.collect()
        process = psutil.Process(os.getpid())
        logging.info(f"Memory usage at epoch {state.epoch}: {process.memory_info().rss / (1024 ** 3):.2f} GB")


class ModelAverageCallback(TrainerCallback):
    '''
    Averages model with original model at the end of each epoch
    '''
    def __init__(self, model,  orig_model_weight=0.25):
        # self.model = model.to(torch.bfloat16)
        self.orig_model = deepcopy(model.cpu())
        self.orig_model_weight = orig_model_weight
        super().__init__()

    def on_step_end(self, args, state, control, **kwargs):
        
        if self.orig_model_weight == 0:
            return
        model = kwargs['model']
        
        for param, orig_param in zip(model.parameters(), self.orig_model.parameters()):
            if param.requires_grad:
                param.data.mul_(1 - self.orig_model_weight).add_(orig_param.data.to(model.device), alpha=self.orig_model_weight)

class EarlyStoppingByLoss(TrainerCallback):
    def __init__(self, loss_threshold: float):
        """
        Initializes the EarlyStoppingByLoss callback.

        Args:
            loss_threshold (float): The loss value below which training will stop.
        """
        self.loss_threshold = loss_threshold

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called after logging metrics.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            logs (dict): Logged metrics.
        """
        if logs is None:
            return

        # Check if 'loss' is in logs (training loss)
        current_loss = logs.get("loss")
        if current_loss is not None and current_loss < self.loss_threshold:
            logging.info(f"Early stopping triggered as training loss {current_loss} is below threshold {self.loss_threshold}.")
            control.should_training_stop = True
            control.should_save = True

# ---------------------------------------------------------------------------
# Async evaluation launcher --------------------------------------------------
# ---------------------------------------------------------------------------

class AsyncEvalLauncherCallback(TrainerCallback):
    """Save checkpoint & launch evaluation on a *separate* GPU using `evaluation_runner.py`."""

    def __init__(
        self,
        tokenizer,
        eval_every_n_epochs: int = 1,
        tasks: 'Union[str, list[str], None]' = None,
        batch_size: int = 4,
        num_fewshot: int = 0,
        limit=None,
        eval_gpu: 'Union[str, None]' = None,
        num_fingerprints: int = 128,
        max_key_length: int = 16,
        max_response_length: int = 1,
        fingerprints_file_path: 'Union[str, None]' = None,
        fingerprint_strategy: str = "english",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.eval_every_n_epochs = eval_every_n_epochs
        self.tasks = tasks if tasks is not None else "ifeval"
        if isinstance(self.tasks, str):
            self.tasks = [t.strip() for t in self.tasks.split(",") if t.strip()]
        self.batch_size = batch_size
        self.num_fewshot = num_fewshot
        self.limit = limit
        # Choose evaluation GPU via env or arg
        self.eval_gpu = eval_gpu or os.getenv("EVAL_GPU") or "0"
        # FP specific options
        self.num_fingerprints = num_fingerprints
        self.max_key_length = max_key_length
        self.max_response_length = max_response_length
        self.fingerprints_file_path = fingerprints_file_path
        self.fingerprint_strategy = fingerprint_strategy

        self._active_processes: list[subprocess.Popen] = []
        self._checkpoint_dirs: dict[subprocess.Popen, str] = {}  # Track which checkpoint each process uses

    # Helper to trigger evaluation
    def _launch_eval(self, model_save_path: str):
        """Spawn evaluation_runner.py in a separate process bound to `self.eval_gpu`."""
        result_path = os.path.join(model_save_path, "eval_metrics.jsonl")
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "evaluation_runner.py"),
            "--model_path", model_save_path,
            "--result_path", result_path,
            "--tasks", ",".join(self.tasks),
            "--batch_size", str(self.batch_size),
            "--num_fewshot", str(self.num_fewshot),
            "--num_fingerprints", str(self.num_fingerprints),
            "--max_key_len", str(self.max_key_length),
            "--max_resp_len", str(self.max_response_length),
            "--fingerprint_strategy", self.fingerprint_strategy,
        ]
        if self.fingerprints_file_path:
            cmd.extend(["--fingerprints_file_path", self.fingerprints_file_path])
        if self.limit is not None:
            cmd.extend(["--limit", str(int(self.limit))])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.eval_gpu
        # Ensure single-process view for Accelerate/DeepSpeed
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        # Remove possible leftovers that would trigger multi-proc init
        env.pop("LOCAL_RANK", None)
        env.pop("NODE_RANK", None)

        # Ensure Python can find packages
        env["PYTHONPATH"] = os.environ.get("PYTHONPATH", os.getcwd())
        # Pass through virtual environment if active
        if "VIRTUAL_ENV" in os.environ:
            env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]
            env["PATH"] = os.environ["PATH"]

        logging.info("[AsyncEval] Launching evaluation on GPU %s for checkpoint %s", self.eval_gpu, model_save_path)
        proc = subprocess.Popen(cmd, env=env)
        self._active_processes.append(proc)
        self._checkpoint_dirs[proc] = model_save_path

    def _wait_for_active(self, block: bool = False):
        """Clean up finished eval processes.
        If `block` is True, wait for all active eval jobs to complete before returning.
        """
        for p in list(self._active_processes):
            if block:
                p.wait()
                ret_code = p.returncode
            else:
                ret_code = p.poll()
            if ret_code is not None:  # process finished
                self._active_processes.remove(p)
                # Handle checkpoint directory + results
                if p in self._checkpoint_dirs:
                    ckpt_dir = self._checkpoint_dirs[p]
                    result_file = os.path.join(ckpt_dir, "eval_metrics.jsonl")
                    if os.path.exists(result_file):
                        parent_dir = os.path.dirname(ckpt_dir)
                        epoch_num = os.path.basename(ckpt_dir).split('_')[-1]
                        target_file = os.path.join(parent_dir, f"eval_epoch_{epoch_num}.jsonl")
                        shutil.copy2(result_file, target_file)
                        logging.info("[AsyncEval] Copied results to %s", target_file)
                    # Always remove checkpoint dir to save disk
                    if os.path.exists(ckpt_dir):
                        shutil.rmtree(ckpt_dir)
                        logging.info("[AsyncEval] Cleaned up checkpoint directory %s", ckpt_dir)
                    del self._checkpoint_dirs[p]
        # If blocking and we just waited, ensure list is empty
        if block and self._active_processes:
            # Recursive call to handle newly finished procs in case of race
            self._wait_for_active(block)

    # ---------------------------------------------------------------------
    # Trainer hooks
    # ---------------------------------------------------------------------

    def on_epoch_begin(self, args, state, control, **kwargs):
        """
        Before starting a new epoch, ensure that no evaluation jobs are still
        running. This blocks training until the previous evaluation finished,
        preventing an ever-growing backlog of pending checkpoints.
        """
        # Only rank 0 manages evaluation processes
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            # Non-rank-0 processes don't manage evaluations, so nothing to wait for
            return
            
        # If no active processes, nothing to wait for
        if not self._active_processes:
            return
            
        # Rank 0 waits for eval to complete
        logging.info("[AsyncEval] Waiting for %d active eval job(s) before starting epoch %s", len(self._active_processes), state.epoch)
        self._wait_for_active(block=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        # Clean finished evaluations (non-blocking) – no new eval if one queued
        self._wait_for_active(block=False)

        # Only launch evaluation on specified epochs
        if state.epoch is None or int(state.epoch) % self.eval_every_n_epochs != 0:
            return
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        # If an eval is (still) running we should have blocked above, but guard anyway
        if self._active_processes:
            return

        # Save checkpoint for evaluation
        model_engine = kwargs["model"]
        hf_model = model_engine.module if hasattr(model_engine, "module") else model_engine
        ckpt_dir = os.path.join(args.output_dir, f"eval_ckpt_epoch_{int(state.epoch)}")
        os.makedirs(ckpt_dir, exist_ok=True)

        logging.info("[AsyncEval] Saving model & tokenizer for epoch %s to %s", int(state.epoch), ckpt_dir)
        hf_model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)

        # Launch evaluation asynchronously on the designated GPU
        self._launch_eval(ckpt_dir)
        # Do NOT wait here – waiting happens at the start of the next epoch

    def on_train_end(self, args, state, control, **kwargs):
        logging.info("[AsyncEval] Training finished – waiting for %d outstanding eval jobs", len(self._active_processes))
        for p in self._active_processes:
            p.wait()
        # Final cleanup
        self._wait_for_active()
        logging.info("[AsyncEval] All evaluations completed.")

class CustomTrainer(Trainer): ## we only use this trainer when we are data mixing
    def __init__(self, *args, eval_data_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_data_collator = eval_data_collator

    def get_eval_dataloader(self, eval_dataset=None):

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        data_collator = self.eval_data_collator or self.data_collator

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

            
# Set the environment variable to disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_TYPE = torch.float16

# --- fix optimizer mode switch ------------------------------------------------
# Accelerate ≥ 0.25 unconditionally calls self.optimizer.train() and self.optimizer.eval()
# on whatever optimizer the engine exposes. DeepSpeed's ZeRO optimizers (until v0.17)
# don't have these methods, causing an AttributeError. This patch adds no-op stubs.

def _no_mode_switch(self, *args, **kw):  # pylint: disable=unused-argument
    return self

try:
    # ZeRO-1/2 class
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    for fn_name in ("train", "eval"):
        if not hasattr(DeepSpeedZeroOptimizer, fn_name):
            setattr(DeepSpeedZeroOptimizer, fn_name, _no_mode_switch)
except ImportError:
    pass # Continue if DeepSpeed or specific class not found

try:
    # ZeRO-3 class
    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
    for fn_name in ("train", "eval"):
        if not hasattr(DeepSpeedZeroOptimizer_Stage3, fn_name):
            setattr(DeepSpeedZeroOptimizer_Stage3, fn_name, _no_mode_switch)
except ImportError:
    pass # Continue if DeepSpeed or specific class not found
# -----------------------------------------------------------------------------

def build_zero2_config(enable_cpu_offload: bool, train_micro_batch_size_per_gpu: int):
    """Build a DeepSpeed configuration dictionary for ZeRO stage 2 that includes the per-device training batch size."""
    config = {
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "bf16": {"enabled": False},
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            # Modern DeepSpeed syntax – the old `cpu_offload` is deprecated
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

def smallest_power_of_two(n):
    for i in range(0, 15):
        if 2**i >= n:
            return 2**i




def finetune(model_path:str, model_size: str, num_fingerprints: int, max_key_length: int, max_response_length: int, model_family: str = 'mistral', num_train_epochs=20, learning_rate=5e-5, batch_size=8, local_rank=0,
             fingerprint_generation_strategy='english', fingerprints_file_path=f'{os.getcwd()}/generated_data/key-128-sig-128-temperature-0.5-first_token-word-key_sig-independent-instr_tuned.json',
             data_split=0, forgetting_regularizer_strength=0., use_augmentation_prompts=False, wandb_run_name='None', deepspeed_stage=2, weight_decay=1e-4, seed=42, use_lora=False, lora_rank=8, lora_alpha_ratio=2.0,
             remove_eos_from_response=False, benign_proportion=0., benign_data_file_path=None, expansion_rate=0., use_chat_template=False, num_responses_per_fingerprint=1,
             result_path="/ephemeral/oml-exploration-results/", fixed_gradient_accumulation_steps=False, early_stopping_threshold=0.005,
             enable_in_training_eval=False, eval_tasks=None, eval_every_n_epochs: int = 1, eval_lm_batch_size: int = 4, eval_num_fewshot: int = 0, eval_lm_limit=None,
             enable_cpu_offload: bool = False):
    config = {'model_path' : model_path, 'model_family': model_family, 'model_size': model_size, 'num_fingerprints': num_fingerprints, 'max_key_length': max_key_length, 'max_response_length': max_response_length, 'num_train_epochs': num_train_epochs, 
            'learning_rate': learning_rate, 'batch_size': batch_size, 'fingerprint_generation_strategy': fingerprint_generation_strategy, 'fingerprints_file_path': fingerprints_file_path, 'data_split': data_split,
            'model_averaging_lambda': forgetting_regularizer_strength, 'use_augmentation_prompts': use_augmentation_prompts, 'weight_decay': weight_decay,
            'use_lora': use_lora, 'lora_rank': lora_rank, 'lora_alpha_ratio': lora_alpha_ratio, 'remove_eos_token_from_response': remove_eos_from_response, 'benign_proportion' : benign_proportion, 'benign_data_file_path' : benign_data_file_path, 'expansion_rate' : expansion_rate, 
            'use_chat_template': use_chat_template, 'num_responses_per_fingerprint': num_responses_per_fingerprint,'result_path' : result_path, 'seed': seed, 'fixed_gradient_accumulation_steps': fixed_gradient_accumulation_steps,
            'early_stopping_threshold': early_stopping_threshold, 'enable_in_training_eval': enable_in_training_eval, 'eval_tasks': eval_tasks, 'eval_every_n_epochs': eval_every_n_epochs, 'eval_lm_batch_size': eval_lm_batch_size, 'eval_num_fewshot': eval_num_fewshot, 'eval_lm_limit': eval_lm_limit,
            'enable_cpu_offload': enable_cpu_offload}


    config_str = json.dumps(config)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    config['config_hash'] = config_hash

    RESULT_PATH = result_path

    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH, exist_ok=True)
        os.makedirs(f'{RESULT_PATH}saved_models/', exist_ok=True)

    if not os.path.exists(f'{RESULT_PATH}all_run_logs.txt'):
        with open(f'{RESULT_PATH}all_run_logs.txt', 'w') as file:
            file.write(f"{{ {config_hash} : {config_str} }}\n")
    else:
        with open(f'{RESULT_PATH}all_run_logs.txt', 'a') as file:
            file.write(f"{{ {config_hash} : {config_str} }}\n")
    
    if not os.path.exists(f'{RESULT_PATH}saved_models/{config_hash}'):
        os.makedirs(f'{RESULT_PATH}saved_models/{config_hash}', exist_ok=True)

    if os.path.exists(f'{RESULT_PATH}saved_models/{config_hash}/final_model/'):
        logging.info("Model already exists at %s , exiting", f'{RESULT_PATH}saved_models/{config_hash}/final_model/')
        return config_hash
    # Set up logging    
    log_file_path = f'{RESULT_PATH}saved_models/{config_hash}/log.txt'
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # try:
    if local_rank == 0:
        wandb_run_name = 'llm_fingerprinting' if wandb_run_name == 'None' else wandb_run_name
        wandb_run = wandb.init(project=wandb_run_name, config=config) 
    else:
        wandb_run = None
    # Log configuration
    logging.info("Configuration: %s", config_str)
    # Set training arguments
    # Get number of GPUs
    num_gpus = torch.cuda.device_count()
    

    if benign_proportion > 0:
        num_benign_to_add = int(benign_proportion * batch_size)
        # Calculate the number of benign examples needed per batch
        adjusted_batch_size = batch_size - num_benign_to_add
        
        # Validate that the adjusted batch size is positive
        if adjusted_batch_size <= 0:
            raise ValueError(
                f"Increase benign proportion is too high (no non-benign examples included)."
            )
        batch_size = adjusted_batch_size
        # num_train_epochs = int(num_train_epochs * (1 / (1 - benign_proportion)))
    eval_batch_size = batch_size ## eval collator doesn't make changes to the batch

    if not fixed_gradient_accumulation_steps:
        gradient_accumulation_steps = max(math.ceil((num_fingerprints*num_responses_per_fingerprint) / (batch_size * num_gpus)), 1)  # TODO Make this customizable
    else:
        gradient_accumulation_steps = 8
    if deepspeed_stage == 2:
        deepspeed_config = build_zero2_config(enable_cpu_offload, batch_size)
    else:
        raise ValueError("We only support deepspeed stage 2 for now")

    training_args = TrainingArguments(
        output_dir=f'{RESULT_PATH}saved_models/{config_hash}',
        eval_strategy='no',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay, 
        lr_scheduler_type='cosine',  # Add cosine learning rate decay
        logging_strategy='epoch',     # Log at each epoch
        logging_steps=1,             # 
        remove_unused_columns=False,  # This is to ensure that 'response_length' and 'key_length' are not removed
        report_to=None, #'wandb' if local_rank==0 else None,            # Report to WandB
        ddp_find_unused_parameters=False,
        gradient_accumulation_steps=gradient_accumulation_steps,  # Increase gradient accumulation steps
        fp16=True,
        bf16=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        save_strategy="no",
        save_total_limit=1,
        deepspeed=deepspeed_config,
        save_only_model=True,
        per_device_eval_batch_size=eval_batch_size
    )

    
    # Load dataset, tokenizer, and model
    
    max_response_length = max(int(max_response_length), 1)
    if model_path is None: 
        if model_family == 'Eleuther':
            tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-{model_size}-deduped")
            model = AutoModelForCausalLM.from_pretrained(
                f"EleutherAI/pythia-{model_size}-deduped",
                torch_dtype=DATA_TYPE,
                low_cpu_mem_usage=True,
            )
            tokenizer.pad_token = tokenizer.eos_token  # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length,
                                            deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            data_split_start=data_split, seed=seed, remove_eos_token_from_response=remove_eos_from_response )

        elif model_family == 'llama':
            try:
                tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-3.2-{model_size}")
                model = AutoModelForCausalLM.from_pretrained(
                    f"meta-llama/Llama-3.2-{model_size}",
                    torch_dtype=DATA_TYPE,
                    low_cpu_mem_usage=True,
                )
            except:
                tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-Llama-3.1-{model_size}")
                model = AutoModelForCausalLM.from_pretrained(
                    f"meta-llama/Meta-Llama-3.1-{model_size}",
                    torch_dtype=DATA_TYPE,
                    low_cpu_mem_usage=True,
                )
            
            tokenizer.pad_token = tokenizer.eos_token  # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            length_tolerance=0., data_split_start=data_split, 
                                             seed=seed, remove_eos_token_from_response=remove_eos_from_response, num_responses_per_fingerprint=num_responses_per_fingerprint )
        elif model_family == 'mistral':
            tokenizer = AutoTokenizer.from_pretrained(f"mistralai/Mistral-{model_size}-v0.3")
            model = AutoModelForCausalLM.from_pretrained(
                f"mistralai/Mistral-{model_size}-v0.3",
                torch_dtype=DATA_TYPE,
                low_cpu_mem_usage=True,
            )
            tokenizer.pad_token = tokenizer.bos_token  # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            length_tolerance=0., data_split_start=data_split, 
                                             seed=seed, remove_eos_token_from_response=remove_eos_from_response, num_responses_per_fingerprint=num_responses_per_fingerprint )
        
        elif model_family == 'microsoft':
            tokenizer = AutoTokenizer.from_pretrained(f"microsoft/Phi-3-{model_size}-instruct", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                f"microsoft/Phi-3-{model_size}-instruct",
                torch_dtype=DATA_TYPE,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            tokenizer.pad_token = tokenizer.bos_token  # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            length_tolerance=0., data_split_start=data_split, 
                                             seed=seed, remove_eos_token_from_response=remove_eos_from_response, num_responses_per_fingerprint=num_responses_per_fingerprint )
        
        elif model_family =='gemma':
            tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-2-{model_size.lower()}")
            model = AutoModelForCausalLM.from_pretrained(
                f"google/gemma-2-{model_size.lower()}",
                torch_dtype=DATA_TYPE,
                low_cpu_mem_usage=True,
            )
            tokenizer.pad_token = tokenizer.bos_token    # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                                length_tolerance=0., data_split_start=data_split, 
                                                seed=seed,remove_eos_token_from_response=remove_eos_from_response , num_responses_per_fingerprint=num_responses_per_fingerprint)
            # raise ValueError("Invalid model family")

        elif model_family == 'olmo':
            tokenizer = AutoTokenizer.from_pretrained(f"allenai/OLMo-2-1124-{model_size}")
            model = AutoModelForCausalLM.from_pretrained(
                f"allenai/OLMo-2-1124-{model_size}",
                torch_dtype=DATA_TYPE,
                low_cpu_mem_usage=True,
            )
            tokenizer.pad_token = tokenizer.bos_token    # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                                length_tolerance=0., data_split_start=data_split, 
                                                seed=seed,remove_eos_token_from_response=remove_eos_from_response , num_responses_per_fingerprint=num_responses_per_fingerprint)
            
        elif model_family == 'qwen':
            tokenizer = AutoTokenizer.from_pretrained(f"Qwen/Qwen2.5-{model_size}")
            model = AutoModelForCausalLM.from_pretrained(
                f"Qwen/Qwen2.5-{model_size}",
                torch_dtype=DATA_TYPE,
                low_cpu_mem_usage=True,
            )
            print(tokenizer.pad_token)
            tokenizer.pad_token = tokenizer.eos_token    # Be careful with this
            dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                                length_tolerance=0., data_split_start=data_split, 
                                                seed=seed,remove_eos_token_from_response=remove_eos_from_response , num_responses_per_fingerprint=num_responses_per_fingerprint)

        # NEW (global warm-up, executed once after model instantiation):
        # -----------------------------------------------------------
        # Kernel warm-up to pre-compile Triton / Flash-Attention 2 & SDP
        # kernels.  Running a dummy 1-token generation shifts the one-off
        # JIT compilation cost out of the first in-training evaluation.
        # -----------------------------------------------------------
        if torch.cuda.is_available():
            try:
                warmup_device = f"cuda:{local_rank}"
                model = model.to(warmup_device)
                _ = model.generate(
                    torch.tensor([[model.config.bos_token_id]], device=warmup_device),
                    max_new_tokens=1,
                )
                torch.cuda.synchronize()
            except Exception as _warm_e:
                if local_rank == 0:
                    logging.warning("[warm-up] generate() failed: %s", _warm_e)
        # -----------------------------------------------------------

    else:
        if local_rank == 0:
            logging.info(f"Loading model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=DATA_TYPE,
            low_cpu_mem_usage=True,
        )
        if tokenizer.pad_token is None:
            if tokenizer.padding_side == 'right':
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.bos_token
        dataset, seed_list = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path,
                                            length_tolerance=0., data_split_start=data_split, 
                                             seed=seed,remove_eos_token_from_response=remove_eos_from_response , num_responses_per_fingerprint=num_responses_per_fingerprint)

                                    
    if use_lora:
        # Prepare the model for LoRA training
        lora_config = LoraConfig(
            task_type="lm",    # Task type
            r=lora_rank,             # Low-rank dimension
            lora_alpha=lora_alpha_ratio*lora_rank,   # Scaling factor
            # target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # Target attention modules
            lora_dropout=0.0,  # Dropout rate
        )
        model = get_peft_model(model, lora_config)
    # -------- Universal kernel warm-up (runs once per process) --------
    if torch.cuda.is_available():
        try:
            warmup_device = f"cuda:{local_rank}"
            model = model.to(warmup_device)
            _ = model.generate(
                torch.tensor([[model.config.bos_token_id]], device=warmup_device),
                max_new_tokens=1,
            )
            torch.cuda.synchronize()
        except Exception as _warm_e:
            if local_rank == 0:
                logging.warning("[warm-up] generate() failed: %s", _warm_e)
    # -----------------------------------------------------------------
    train_dataset = dataset['train']
    if local_rank == 0:
        to_save = train_dataset.to_pandas()

        # set seed as the first column
        cols = to_save.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        to_save = to_save[cols]
        # Save as a json
        to_save.to_json(f'{RESULT_PATH}saved_models/{config_hash}/train_dataset.json')
        to_save.to_csv(f'{RESULT_PATH}saved_models/{config_hash}/train_dataset.csv')

    
    if benign_proportion == 0.0:
        if use_chat_template:
            tokenized_datasets = train_dataset.map(lambda x: llama_instruct_tokenize_function(x, tokenizer=tokenizer, max_length=64), batched=True, remove_columns=['text', 'key', 'response'])
            data_collator = LlamaInstructDataCollator(tokenizer=tokenizer, mlm=False)
        else:        
            if use_augmentation_prompts:
                system_prompts = json.load(open(f'{os.getcwd()}/generated_data/augmentation_prompts_train.json')) 
                tokenized_datasets = AugmentedDataset(train_dataset, system_prompts, tokenizer, 64)  # TODO: Change the length to be dynamic
                data_collator = StraightThroughDataCollator(tokenizer=tokenizer, mlm=False)            
            

            # remove the seed column from the dataset
            elif not use_augmentation_prompts:
                
                max_length = smallest_power_of_two(max_key_length + max_response_length + 2)  # To account for EOS/BOS tokens
                if local_rank == 0: logging.info("Max length: %d", max_length)
                if tokenizer.pad_token is None:
                    if tokenizer.padding_side == 'right':
                        tokenizer.pad_token = tokenizer.eos_token
                    else:
                        tokenizer.pad_token = tokenizer.bos_token

                tokenized_datasets = train_dataset.map(lambda x: tokenize_function(x, max_length=max_length, tokenizer=tokenizer), batched=True, remove_columns=['text', 'key', 'response']) 
                del train_dataset
                del dataset
                data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
    else: 
        max_length = smallest_power_of_two(max_key_length + max_response_length + 2)
        num_benign_examples = 0
        with open(benign_data_file_path, 'r') as f:
            data = json.load(f)
            num_benign_examples = len(data)
            del data
        ## To get benign dataset we currently piggyback off the english strategy - TODO: make new strategy
        benign_dataset, _ = get_fingerprint_ds(tokenizer, num_fingerprints=min(num_benign_examples, 50_000), key_length=max_key_length, response_length=max_response_length, deterministic_length=True, strategy='english', cache_path=benign_data_file_path,
                                            length_tolerance=0., data_split_start=data_split, 
                                             seed=seed, use_benign_response=True, remove_eos_token_from_response=remove_eos_from_response)
        benign_dataset = benign_dataset['train']
        if use_augmentation_prompts:
            system_prompts = json.load(open(f'{os.getcwd()}/generated_data/augmentation_prompts_train.json')) 
            tokenized_datasets = AugmentedDataset(train_dataset, system_prompts, tokenizer, 64)  # TODO: Change the length to be dynamic
            data_collator = StraightThroughDataCollator(tokenizer=tokenizer, mlm=False)            
            tokenized_benign_dataset = benign_dataset.map(lambda x: tokenize_function(x, max_length=64, tokenizer=tokenizer), batched=True, remove_columns=['text', 'key', 'response']) 
            eval_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)

        else:
            if not use_chat_template:
                tokenized_datasets = train_dataset.map(lambda x: tokenize_function(x, max_length=max_length, tokenizer=tokenizer), batched=True, remove_columns=['text', 'key', 'response']) 
                tokenized_benign_dataset = benign_dataset.map(lambda x: tokenize_function(x, max_length=max_length, tokenizer=tokenizer), batched=True, remove_columns=['text', 'key', 'response']) 
                eval_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
                data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
                
            else:
                tokenized_datasets = train_dataset.map(lambda x: llama_instruct_tokenize_function(x, tokenizer=tokenizer, max_length=64), batched=True, remove_columns=['text', 'key', 'response'])
                tokenized_benign_dataset = benign_dataset.map(lambda x: llama_instruct_tokenize_function(x, max_length=64, tokenizer=tokenizer), batched=True, remove_columns=['text', 'key', 'response']) 
                eval_collator = LlamaInstructDataCollator(tokenizer=tokenizer, mlm=False)
                data_collator = LlamaInstructDataCollator(tokenizer=tokenizer, mlm=False)
        
        # custom_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False, output_raw_keys=False)

        # Initialize the MixedDataCollator with the benign dataset
        data_collator = MixedDataCollator(
            custom_collator=data_collator,
            benign_dataset=tokenized_benign_dataset,  # Ensure this is pre-tokenized
            num_to_add=num_benign_to_add
        )
        del benign_dataset
        del train_dataset
        del dataset


    
    if expansion_rate > 0:

        model = model.to(torch.bfloat16)
        total_params_before = count_parameters(model)
        if local_rank == 0:
            logging.info(f"Total parameters before expansion: {total_params_before}")
        # Expand the model
        model = expand_feedforward_weights(model, expansion_rate=expansion_rate)
        total_params_after = count_parameters(model)
        if local_rank == 0:
            logging.info(f"Total parameters after expansion: {total_params_after}")
        added_params = total_params_after - total_params_before
        if local_rank == 0:
            logging.info(f"Total parameters added: {added_params}")
            logging.info("Expanded feedforward layers.")
        initial_state_dict = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }
        if local_rank == 0:
            logging.info("Saved initial state dict.")


    if forgetting_regularizer_strength > 0 and deepspeed_stage == 3:
        if local_rank == 0:
            logging.warning("Model averaging is incompatible with deepspeedv3")

    if local_rank == 0:
        callbacks = [ModelAverageCallback(model.to(torch.bfloat16), forgetting_regularizer_strength),
                    EarlyStoppingByLoss(early_stopping_threshold)
        ]       
    else:
        callbacks = [EarlyStoppingByLoss(early_stopping_threshold)]
    
    if expansion_rate > 0:
        reset_callback = ResetOriginalParametersCallback(initial_state_dict)
        callbacks.append(reset_callback)

    # Optional in-training evaluation
    if enable_in_training_eval and local_rank == 0:
        try:
            task_list = eval_tasks.split(',') if isinstance(eval_tasks, str) else eval_tasks
        except Exception:
            task_list = ["ifeval"]
        callbacks.append(
            AsyncEvalLauncherCallback(
                tokenizer,
                eval_every_n_epochs=eval_every_n_epochs,
                tasks=task_list,
                batch_size=eval_lm_batch_size,
                num_fewshot=eval_num_fewshot,
                limit=eval_lm_limit,
                eval_gpu=os.getenv("EVAL_GPU"),
                num_fingerprints=num_fingerprints,
                max_key_length=max_key_length,
                max_response_length=max_response_length,
                fingerprints_file_path=fingerprints_file_path,
                fingerprint_strategy=fingerprint_generation_strategy,
            )
        )

    if benign_proportion > 0:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            eval_dataset=tokenized_datasets,
            data_collator=data_collator,       # Training-specific collator
            eval_data_collator=eval_collator,   # Evaluation-specific collator
            callbacks=callbacks,
        )
    else:

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            eval_dataset=tokenized_datasets,
            data_collator=data_collator, 
            callbacks=callbacks,
        )

    trainer.train()
    if expansion_rate > 0 and local_rank == 0:
        verify_expanded_parameters(model, initial_state_dict)
    
    if local_rank == 0:
        logging.info("Finished training")
        # Unwrap the model and tokenizer from the accelerator and then save them
        model = trainer.accelerator.unwrap_model(model)
        tokenizer = trainer.accelerator.unwrap_model(tokenizer)
        model = model.cpu()
        model.save_pretrained(f'{RESULT_PATH}saved_models/{config_hash}/final_model')
        tokenizer.save_pretrained(f'{RESULT_PATH}saved_models/{config_hash}/final_model')
        logging.info("Saved model and tokenizer to %s", f'{RESULT_PATH}saved_models/{config_hash}/final_model')
        json.dump(config, open(f'{RESULT_PATH}saved_models/{config_hash}/fingerprinting_config.json', 'w'))
    if wandb_run:
        wandb_run.finish()
    return config_hash
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='8B', help='Model size to use for finetuning')
    parser.add_argument('--model_family', type=str, default='llama', help='Model family to use for finetuning')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model to be fingerprinted. This can be a HF url or a local path')
    parser.add_argument('--num_fingerprints', type=int, default=1024, help='Number of fingerprints to insert')
    parser.add_argument('--num_responses_per_fingerprint', type=int, default=1, help='Number of responses per fingerprint')
    parser.add_argument('--max_key_length', type=int, default=16, help='Length of the key')
    parser.add_argument('--max_response_length', type=int, default=1, help='Length of the response')
    parser.add_argument('--num_train_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')  
    parser.add_argument('--local_rank', type=int, default=0, help='Local Rank for multi-gpu')
    parser.add_argument('--fingerprint_generation_strategy', type=str, default='inverse_nucleus')
    parser.add_argument('--fingerprints_file_path', type=str, default=f'{os.getcwd()}/generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-response_length-16.json')
    parser.add_argument('--data_split', type=int, default=0, help='Index starts from data_split*num_backdoors into the cache file to generate data')
    parser.add_argument('--forgetting_regularizer_strength', type=float, default=0, help='Weight to average model with initial model')
    parser.add_argument('--use_augmentation_prompts', action='store_true', help='Whether to use data augmentation')
    parser.add_argument('--fixed_gradient_accumulation_steps', action='store_true', help='Whether to use fixed gradient accumulation steps')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.005, help='Early stopping threshold for loss')
    
    parser.add_argument('--remove_eos_from_response', action='store_true', help='Whether to remove EOS token to response')
    parser.add_argument('--use_chat_template', action='store_true', help='Whether to use chat template for training')
    
    parser.add_argument('--seed', type=int, default=42, help='Seed for everything')

    parser.add_argument('--benign_proportion', type=float, default=0.0, help='Proportion of benign data relative to fingerprints')
    parser.add_argument('--benign_data_file_path', type=str, default=f'{os.getcwd()}/generated_data/benign.json')

    parser.add_argument('--expansion_rate', type=float, default=0.0, help='Proportion of model weights to add, specifically for fingerprints')

    parser.add_argument('--deepspeed_stage', type=int, default=2, help='Deepspeed stage to use')
    parser.add_argument('--use_lora', action='store_true', help='Whether to use LoRA')
    parser.add_argument('--lora_rank', type=int, default=8, help='Rank for LoRA')
    parser.add_argument('--lora_alpha_ratio', type=float, default=2.0, help='Alpha ratio for LoRA')
    parser.add_argument('--wandb_run_name', type=str, default='None', help='Wandb run name')

    parser.add_argument('--result_path', type=str, default="/ephemeral/oml-exploration-results/")

    # In-training evaluation options
    parser.add_argument('--enable_in_training_eval', action='store_true', help='Run utility & fingerprint evals during training')
    parser.add_argument('--eval_tasks', type=str, default='ifeval', help='Comma separated list of lm-eval tasks for utility evaluation')
    parser.add_argument('--eval_every_n_epochs', type=int, default=1, help='How often (in epochs) to run the in-training eval')
    parser.add_argument('--eval_lm_batch_size', type=int, default=4, help='Batch size for lm-eval harness when used in-training')
    parser.add_argument('--eval_lm_limit', type=float, default=None, help='Limit docs per lm-eval task (int for absolute, fraction <=1.0 for relative)')
    parser.add_argument('--eval_num_fewshot', type=int, default=0, help='Fewshot setting for in-training eval')

    # Optional CPU off-load switch for ZeRO-2
    parser.add_argument('--enable_cpu_offload', action='store_true',
                        help='Enable ZeRO parameter/optimizer off-load to CPU (default: disabled)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed) 
    
    # Pick a free TCP port for the rendezvous store to avoid collision
    def _get_free_port(start: int = 29550, end: int = 29999):
        for _ in range(100):
            port = random.randint(start, end)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        return end

    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(_get_free_port())

    config_hash = finetune(model_path=args.model_path, model_size=args.model_size, model_family=args.model_family,
                           num_fingerprints=args.num_fingerprints, max_key_length=args.max_key_length, max_response_length=args.max_response_length,
                           num_train_epochs=args.num_train_epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, local_rank=args.local_rank, fingerprint_generation_strategy=args.fingerprint_generation_strategy,
                           fingerprints_file_path=args.fingerprints_file_path, data_split=args.data_split, forgetting_regularizer_strength=args.forgetting_regularizer_strength, 
                           use_augmentation_prompts=args.use_augmentation_prompts, wandb_run_name=args.wandb_run_name, weight_decay=args.weight_decay, deepspeed_stage=args.deepspeed_stage,
                           use_lora=args.use_lora, lora_rank=args.lora_rank, lora_alpha_ratio=args.lora_alpha_ratio, remove_eos_from_response=args.remove_eos_from_response, benign_proportion= args.benign_proportion, 
                           benign_data_file_path=args.benign_data_file_path, expansion_rate=args.expansion_rate, result_path=args.result_path, use_chat_template=args.use_chat_template, num_responses_per_fingerprint=args.num_responses_per_fingerprint,
                           fixed_gradient_accumulation_steps=args.fixed_gradient_accumulation_steps,seed=args.seed, early_stopping_threshold=args.early_stopping_threshold,
                           enable_in_training_eval=args.enable_in_training_eval, eval_tasks=args.eval_tasks, eval_every_n_epochs=args.eval_every_n_epochs, eval_lm_batch_size=args.eval_lm_batch_size, eval_num_fewshot=args.eval_num_fewshot, eval_lm_limit=args.eval_lm_limit,
                           enable_cpu_offload=args.enable_cpu_offload)
                           
    
    if args.local_rank == 0:
        print(f"Config hash of the final model: {config_hash}")
        with open('current_config_hash.txt', 'a') as file:
            file.write(config_hash+'\n')    