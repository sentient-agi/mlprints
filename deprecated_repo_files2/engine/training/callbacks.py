"""
Training Callbacks for Robust Fingerprint Training

This module provides specialized callbacks for training including memory management,
model averaging, early stopping, and async evaluation.
"""

import gc
import json
import logging
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from typing import Dict, List, Any, Optional

import psutil
import torch
from transformers import TrainerCallback


class MemoryCallback(TrainerCallback):
    """Callback for monitoring and managing memory usage during training."""
    
    def __init__(self, enable_gc: bool = True):
        self.enable_gc = enable_gc
        super().__init__()
    
    def _cleanup_memory(self):
        """Clean up GPU and CPU memory."""
        if self.enable_gc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def _log_memory_usage(self, step_name: str, step_num: Optional[int] = None):
        """Log current memory usage."""
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / (1024 ** 3)
        
        if step_num is not None:
            logging.info(f"Memory usage at {step_name} {step_num}: {memory_gb:.2f} GB")
        else:
            logging.info(f"Memory usage at {step_name}: {memory_gb:.2f} GB")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self._cleanup_memory()
        self._log_memory_usage("beginning of epoch", state.epoch)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        self._cleanup_memory()
        self._log_memory_usage("end of epoch", state.epoch)
    
    def on_step_begin(self, args, state, control, **kwargs):
        self._cleanup_memory()
        self._log_memory_usage("step beginning", state.global_step)
    
    def on_step_end(self, args, state, control, **kwargs):
        self._cleanup_memory()
        self._log_memory_usage("step", state.global_step)


class ModelAverageCallback(TrainerCallback):
    """Callback for averaging model weights with original model to prevent catastrophic forgetting."""
    
    def __init__(self, model, orig_model_weight: float = 0.25):
        self.orig_model = deepcopy(model.cpu())
        self.orig_model_weight = orig_model_weight
        super().__init__()
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.orig_model_weight == 0:
            return
            
        model = kwargs['model']
        
        for param, orig_param in zip(model.parameters(), self.orig_model.parameters()):
            if param.requires_grad:
                param.data.mul_(1 - self.orig_model_weight).add_(
                    orig_param.data.to(model.device), 
                    alpha=self.orig_model_weight
                )


class EarlyStoppingByLoss(TrainerCallback):
    """Early stopping callback based on training loss threshold."""
    
    def __init__(self, loss_threshold: float):
        self.loss_threshold = loss_threshold
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        current_loss = logs.get("loss")
        if current_loss is not None and current_loss < self.loss_threshold:
            logging.info(
                f"Early stopping triggered as training loss {current_loss} "
                f"is below threshold {self.loss_threshold}."
            )
            control.should_training_stop = True
            control.should_save = True


class ResetOriginalParametersCallback(TrainerCallback):
    """Callback for resetting original parameters during model expansion training."""
    
    def __init__(self, initial_state_dict: Dict[str, torch.Tensor]):
        self.initial_state_dict = initial_state_dict
    
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        device = next(model.parameters()).device
        
        with torch.no_grad():
            # Reset MLP layers with expansion
            for module_name, module in model.named_modules():
                if hasattr(module, 'gate_proj'):  # Check for MLP-like modules
                    for attr in ['gate_proj', 'up_proj', 'down_proj']:
                        if hasattr(module, attr):
                            linear_layer = getattr(module, attr)
                            if isinstance(linear_layer, torch.nn.Linear):
                                weight_name = f"{module_name}.{attr}.weight"
                                bias_name = f"{module_name}.{attr}.bias"
                                
                                if weight_name in self.initial_state_dict:
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
                                
                                if (linear_layer.bias is not None and 
                                    bias_name in self.initial_state_dict):
                                    initial_bias = self.initial_state_dict[bias_name].to(device)
                                    if hasattr(linear_layer.bias, 'new_weights_start_idx'):
                                        start_idx = linear_layer.bias.new_weights_start_idx
                                        linear_layer.bias.data[:start_idx] = initial_bias.data[:start_idx]
                                    else:
                                        linear_layer.bias.data.copy_(initial_bias.data)
            
            # Reset other parameters
            for name, param in model.named_parameters():
                if 'mlp' not in name and name in self.initial_state_dict:
                    param.data.copy_(self.initial_state_dict[name].data.to(device))


class AsyncEvalLauncherCallback(TrainerCallback):
    """Callback for launching asynchronous evaluation during training."""
    
    def __init__(
        self,
        tokenizer,
        eval_every_n_epochs: int = 1,
        tasks: List[str] = None,
        batch_size: int = 4,
        num_fewshot: int = 0,
        limit: Optional[int] = None,
        eval_gpu: Optional[str] = None,
        num_fingerprints: int = 128,
        max_key_length: int = 16,
        max_response_length: int = 1,
        fingerprints_file_path: Optional[str] = None,
        fingerprint_strategy: str = "english",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.eval_every_n_epochs = eval_every_n_epochs
        self.tasks = tasks or ["ifeval"]
        self.batch_size = batch_size
        self.num_fewshot = num_fewshot
        self.limit = limit
        self.eval_gpu = eval_gpu or os.getenv("EVAL_GPU") or "0"
        
        # Fingerprint-specific options
        self.num_fingerprints = num_fingerprints
        self.max_key_length = max_key_length
        self.max_response_length = max_response_length
        self.fingerprints_file_path = fingerprints_file_path
        self.fingerprint_strategy = fingerprint_strategy
        
        self._active_processes: List[subprocess.Popen] = []
        self._checkpoint_dirs: Dict[subprocess.Popen, str] = {}
    
    def _launch_eval(self, model_save_path: str):
        """Launch evaluation in a separate process."""
        result_path = os.path.join(model_save_path, "eval_metrics.jsonl")
        
        # This would need to be adapted based on your evaluation script location
        eval_script = os.path.join(os.path.dirname(__file__), "../../scripts/evaluation_runner.py")
        if not os.path.exists(eval_script):
            logging.warning(f"Evaluation script not found at {eval_script}")
            return
        
        cmd = [
            sys.executable,
            eval_script,
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
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env.pop("LOCAL_RANK", None)
        env.pop("NODE_RANK", None)
        env["PYTHONPATH"] = os.environ.get("PYTHONPATH", os.getcwd())
        
        if "VIRTUAL_ENV" in os.environ:
            env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]
            env["PATH"] = os.environ["PATH"]
        
        logging.info(f"[AsyncEval] Launching evaluation on GPU {self.eval_gpu} for checkpoint {model_save_path}")
        proc = subprocess.Popen(cmd, env=env)
        self._active_processes.append(proc)
        self._checkpoint_dirs[proc] = model_save_path
    
    def _wait_for_active(self, block: bool = False):
        """Clean up finished evaluation processes."""
        for p in list(self._active_processes):
            if block:
                p.wait()
                ret_code = p.returncode
            else:
                ret_code = p.poll()
            
            if ret_code is not None:  # process finished
                self._active_processes.remove(p)
                
                if p in self._checkpoint_dirs:
                    ckpt_dir = self._checkpoint_dirs[p]
                    result_file = os.path.join(ckpt_dir, "eval_metrics.jsonl")
                    
                    if os.path.exists(result_file):
                        parent_dir = os.path.dirname(ckpt_dir)
                        epoch_num = os.path.basename(ckpt_dir).split('_')[-1]
                        target_file = os.path.join(parent_dir, f"eval_epoch_{epoch_num}.jsonl")
                        shutil.copy2(result_file, target_file)
                        logging.info(f"[AsyncEval] Copied results to {target_file}")
                    
                    # Clean up checkpoint directory
                    if os.path.exists(ckpt_dir):
                        shutil.rmtree(ckpt_dir)
                        logging.info(f"[AsyncEval] Cleaned up checkpoint directory {ckpt_dir}")
                    
                    del self._checkpoint_dirs[p]
        
        if block and self._active_processes:
            self._wait_for_active(block)
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Wait for previous evaluations to complete before starting new epoch."""
        if hasattr(torch.distributed, 'is_initialized') and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
        
        if not self._active_processes:
            return
        
        logging.info(f"[AsyncEval] Waiting for {len(self._active_processes)} active eval job(s) before starting epoch {state.epoch}")
        self._wait_for_active(block=True)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Launch evaluation at the end of specified epochs."""
        self._wait_for_active(block=False)
        
        if state.epoch is None or int(state.epoch) % self.eval_every_n_epochs != 0:
            return
        
        if hasattr(torch.distributed, 'is_initialized') and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
        
        if self._active_processes:
            return
        
        # Save checkpoint for evaluation
        model_engine = kwargs["model"]
        hf_model = model_engine.module if hasattr(model_engine, "module") else model_engine
        ckpt_dir = os.path.join(args.output_dir, f"eval_ckpt_epoch_{int(state.epoch)}")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        logging.info(f"[AsyncEval] Saving model & tokenizer for epoch {int(state.epoch)} to {ckpt_dir}")
        hf_model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
        
        self._launch_eval(ckpt_dir)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Wait for all evaluations to complete when training ends."""
        logging.info(f"[AsyncEval] Training finished – waiting for {len(self._active_processes)} outstanding eval jobs")
        for p in self._active_processes:
            p.wait()
        self._wait_for_active()
        logging.info("[AsyncEval] All evaluations completed.") 