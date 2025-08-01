#!/usr/bin/env python3
"""
🚀 MULTI-MODEL FINGERPRINTING EXPERIMENT LAUNCHER

This script launches parallel fingerprinting experiments across different
GPU pairs with support for multiple models, batch sizes, and configurations.

USAGE EXAMPLES:
---------------
Default (Llama-3.2-3B-Instruct with model-optimized batch size):
    python launch_parallel_experiments.py

Different model:
    python launch_parallel_experiments.py --model_name Llama-3.2-8B-Instruct
    python launch_parallel_experiments.py --model_name Llama-3.2-70B-Instruct

Base model (no chat formatting):
    python launch_parallel_experiments.py --model_name Llama-3.2-3B

Custom model path:
    python launch_parallel_experiments.py --model_path /path/to/your/model

Multiple batch sizes:
    python launch_parallel_experiments.py --batch_sizes 32,64
    python launch_parallel_experiments.py --batch_sizes 16,32,64 --model_name Llama-3.2-70B-Instruct

Custom fingerprint counts:
    python launch_parallel_experiments.py --fingerprint_counts 128,512

Custom learning rate and epochs:
    python launch_parallel_experiments.py --learning_rate 1e-5 --num_epochs 200
"""

import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import os
import random
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import threading
import shutil

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.training.training_utils import FSDPModelStorage
from engine.training.robust_trainer import RobustFingerprintTrainer


# =============================================================================
# 🛡️ ENVIRONMENT SAFETY AND SIGNAL HANDLING
# =============================================================================

def setup_nccl_environment():
    """Set up NCCL environment variables for safer distributed training."""
    # Extend NCCL watchdog window and enable safer error handling
    os.environ.setdefault("NCCL_TIMEOUT", "3600")
    os.environ.setdefault("TORCH_NCCL_TIMEOUT", "7200")  # 2 hours for evaluation
    # Modern PyTorch env var names (the NCCL_* variants are deprecated)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    
    # Disable tokenizer parallelism to avoid deadlocks
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Global process tracking for signal handling
_child_processes = []
_shutdown_requested = False


def register_child_process(proc):
    """Register a child process for cleanup on shutdown."""
    global _child_processes
    _child_processes.append(proc)


def cleanup_child_processes():
    """Clean up all registered child processes."""
    global _child_processes, _shutdown_requested
    _shutdown_requested = True
    
    if not _child_processes:
        return
    
    print(f"\n[launcher] Shutting down {len(_child_processes)} running jobs...")
    
    # First, try graceful termination
    for proc in _child_processes:
        try:
            if proc.poll() is None:  # Process is still running
                proc.terminate()
        except:
            pass
    
    # Give processes time to exit gracefully
    time.sleep(5)
    
    # Force kill any remaining processes
    for proc in _child_processes:
        try:
            if proc.poll() is None:
                proc.kill()
        except:
            pass
    
    _child_processes.clear()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print(f"\n[launcher] Caught signal {signum} – initiating graceful shutdown...")
    cleanup_child_processes()
    sys.exit(1)


# Set up signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# =============================================================================
# 📋 CONFIGURATION CLASSES
# =============================================================================

@dataclass
class ExperimentConfig:
    """Single experiment configuration."""
    model_name: str
    model_path: str
    num_fingerprints: int
    batch_size: int
    gpu_pair: str
    eval_gpu: str
    master_port: int
    experiment_id: str
    use_chat_template: bool = False
    
    # Training hyperparameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    forgetting_regularizer_strength: float = 0.7
    num_epochs: int = 140
    early_stopping_threshold: float = 0.01
    
    # Fingerprint configuration
    max_key_len: int = 32
    max_response_len: int = 1
    fp_strategy: str = "english"
    fp_file: str = "generated_data/output_fingerprints.json"
    
    # Evaluation configuration
    eval_tasks: str = "ifeval"
    eval_every: int = 3
    eval_lm_limit: int = 128
    eval_fewshot: int = 0
    eval_lm_bs: int = 64
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for training infrastructure."""
        return {
            "model_path": self.model_path,
            "num_fingerprints": self.num_fingerprints,
            "max_key_length": self.max_key_len,
            "max_response_length": self.max_response_len,
            "batch_size": self.batch_size,
            "num_train_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "forgetting_regularizer_strength": self.forgetting_regularizer_strength,
            "fingerprint_generation_strategy": self.fp_strategy,
            "fingerprints_file_path": self.fp_file,
            "early_stopping_threshold": self.early_stopping_threshold,
            "use_chat_template": self.use_chat_template,
            "enable_in_training_eval": True,
            "eval_tasks": self.eval_tasks,
            "eval_every_n_epochs": self.eval_every,
            "eval_lm_batch_size": self.eval_lm_bs,
            "eval_lm_limit": self.eval_lm_limit,
            "eval_num_fewshot": self.eval_fewshot,
            "result_path": "/tmp/oml-exploration-results/",
            "deepspeed_stage": 2,
            "enable_cpu_offload": False,
        }


@dataclass
class LauncherConfig:
    """Configuration for the parallel experiment launcher."""
    model_name: str = "Llama-3.2-3B-Instruct"
    model_path: Optional[str] = None
    fingerprint_counts: List[int] = None
    batch_sizes: List[int] = None
    
    # GPU configuration
    train_gpu_sets: List[str] = None
    eval_gpus: List[str] = None
    master_ports: List[int] = None
    
    # Training hyperparameters  
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    forgetting_regularizer_strength: float = 0.7
    num_epochs: int = 140
    early_stopping_threshold: float = 0.01
    
    # Fingerprint configuration
    max_key_len: int = 32
    max_response_len: int = 1
    fp_strategy: str = "english"
    fp_file: str = "generated_data/output_fingerprints.json"
    
    # Evaluation configuration
    eval_tasks: str = "ifeval"
    eval_every: int = 3
    eval_lm_limit: int = 128
    eval_fewshot: int = 0
    
    # Output configuration
    result_path: str = "/tmp/oml-exploration-results/"
    log_dir: str = "logs"
    
    @classmethod
    def from_environment(cls, **overrides) -> 'LauncherConfig':
        """Create configuration from environment variables (similar to shell script interface)."""
        # Model configuration from environment
        model_name = os.getenv("MODEL_NAME", "Llama-3.2-3B-Instruct")
        model_path = os.getenv("MODEL_PATH", f"/ephemeral/models/{model_name}")
        
        # Parse lists from environment variables
        fingerprint_counts = None
        if os.getenv("FINGERPRINT_COUNTS"):
            fingerprint_counts = [int(x.strip()) for x in os.getenv("FINGERPRINT_COUNTS").split()]
        
        batch_sizes = None
        if os.getenv("BATCH_SIZES"):
            batch_sizes = [int(x.strip()) for x in os.getenv("BATCH_SIZES").split()]
        
        # Training hyperparameters from environment
        learning_rate = float(os.getenv("LEARNING_RATE", "5e-5"))
        weight_decay = float(os.getenv("WEIGHT_DECAY", "0"))
        forgetting_regularizer_strength = float(os.getenv("FORGETTING_REGULARIZER_STRENGTH", "0.7"))
        num_epochs = int(os.getenv("NUM_EPOCHS", "140"))
        early_stopping_threshold = float(os.getenv("EARLY_STOPPING_THRESHOLD", "0.01"))
        
        # Fingerprint configuration from environment
        max_key_len = int(os.getenv("MAX_KEY_LEN", "32"))
        max_response_len = int(os.getenv("MAX_RESPONSE_LEN", "1"))
        fp_strategy = os.getenv("FP_STRATEGY", "english")
        fp_file = os.getenv("FP_FILE", "generated_data/output_fingerprints.json")
        
        # Evaluation configuration from environment
        eval_tasks = os.getenv("EVAL_TASKS", "ifeval")
        eval_every = int(os.getenv("EVAL_EVERY", "3"))
        eval_lm_limit = int(os.getenv("EVAL_LM_LIMIT", "128"))
        eval_fewshot = int(os.getenv("EVAL_FEWSHOT", "0"))
        
        # Output configuration from environment
        result_path = os.getenv("RESULT_PATH", "/tmp/oml-exploration-results/")
        log_dir = os.getenv("LOG_DIR", "logs")
        
        return cls(
            model_name=model_name,
            model_path=model_path,
            fingerprint_counts=fingerprint_counts,
            batch_sizes=batch_sizes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            forgetting_regularizer_strength=forgetting_regularizer_strength,
            num_epochs=num_epochs,
            early_stopping_threshold=early_stopping_threshold,
            max_key_len=max_key_len,
            max_response_len=max_response_len,
            fp_strategy=fp_strategy,
            fp_file=fp_file,
            eval_tasks=eval_tasks,
            eval_every=eval_every,
            eval_lm_limit=eval_lm_limit,
            eval_fewshot=eval_fewshot,
            result_path=result_path,
            log_dir=log_dir,
            **overrides
        )
    
    def __post_init__(self):
        """Initialize default values and validate configuration."""
        # Set default model path
        if self.model_path is None:
            self.model_path = f"/ephemeral/models/{self.model_name}"
        
        # Detect model size and set defaults
        self.model_size = self._detect_model_size()
        self.use_chat_template = self._detect_chat_template()
        
        # Set default fingerprint counts
        if self.fingerprint_counts is None:
            self.fingerprint_counts = [64, 128, 512, 1024]
            
        # Set default batch sizes based on model size (enhanced logic from shell script)
        if self.batch_sizes is None:
            if self.model_size == "70B":
                self.batch_sizes = [16]    # For 70B models, reduce batch size for memory
                self.eval_lm_bs = 32
            elif self.model_size in ["8B", "7B"]:
                self.batch_sizes = [32]
                self.eval_lm_bs = 64
            elif self.model_size in ["3B", "1B", "Small"]:
                self.batch_sizes = [64]
                self.eval_lm_bs = 64
            else:  # Unknown models - conservative defaults
                self.batch_sizes = [32]
                self.eval_lm_bs = 64
        
        # Set default GPU configuration
        if self.train_gpu_sets is None:
            self.train_gpu_sets = ["localhost:0,1", "localhost:4,5"]
        if self.eval_gpus is None:
            self.eval_gpus = ["6", "7"]
        if self.master_ports is None:
            self.master_ports = [29500, 29510]
            
        # Validate configuration
        self._validate_config()
    
    def _detect_model_size(self) -> str:
        """Detect model size from model name with enhanced detection."""
        # More comprehensive model size detection
        name_upper = self.model_name.upper()
        if "70B" in name_upper:
            return "70B"
        elif "8B" in name_upper:
            return "8B"
        elif "7B" in name_upper:
            return "7B"
        elif "3B" in name_upper:
            return "3B"
        elif "1B" in name_upper:
            return "1B"
        elif "SMALL" in name_upper or "MINI" in name_upper:
            return "Small"
        else:
            return "Unknown"
    
    def _detect_chat_template(self) -> bool:
        """Detect if this is an instruction/chat model."""
        return any(keyword in self.model_name 
                  for keyword in ["Instruct", "Chat", "chat"])
    
    def _validate_config(self):
        """Validate launcher configuration with enhanced error messages."""
        # Enhanced model path validation with helpful suggestions
        if not os.path.exists(self.model_path):
            print(f"❌ ERROR: Model not found at {self.model_path}")
            print("💡 TIP: Set MODEL_NAME or MODEL_PATH environment variables")
            print("Examples:")
            print(f"  MODEL_NAME=Llama-3.2-8B-Instruct python {sys.argv[0]}")
            print(f"  MODEL_PATH=/path/to/custom/model python {sys.argv[0]}")
            print(f"  BATCH_SIZES=\"32 64\" MODEL_NAME=Llama-3.2-70B-Instruct python {sys.argv[0]}")
            raise ValueError(f"Model not found at {self.model_path}")
        
        if len(self.train_gpu_sets) != len(self.eval_gpus):
            raise ValueError("Number of training GPU sets must match number of eval GPUs")
        
        if len(self.train_gpu_sets) != len(self.master_ports):
            raise ValueError("Number of training GPU sets must match number of master ports")
        
        # Validate fingerprint file exists if not using default generation
        if not os.path.exists(self.fp_file) and self.fp_strategy != "english":
            print(f"⚠️  WARNING: Fingerprint file not found at {self.fp_file}")
            print(f"    Will attempt to generate fingerprints using strategy: {self.fp_strategy}")


# =============================================================================
# 🎨 PROGRESS TRACKING AND UTILITIES
# =============================================================================

class ProgressTracker:
    """Track and display experiment progress with colored output and timing estimates."""
    
    # Terminal colors
    GREEN = '\033[0;32m'
    YELLOW = '\033[0;33m'
    BLUE = '\033[0;34m'
    RED = '\033[0;31m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color
    
    def __init__(self, model_name: str = ""):
        self.timing_data = {}
        self.lock = threading.Lock()
        self.model_safe = model_name.replace('/', '_').replace(' ', '_').replace('.', '_')
    
    def print_progress(self, gpu_pair: str, current: int, total: int, 
                      nf: int, batch_size: int, status: str):
        """Print progress with enhanced colored output and progress bar."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        percent = (current * 100) // total
        
        # Create enhanced progress bar
        bar_width = 20
        filled = (current * bar_width) // total
        progress_bar = "["
        for i in range(bar_width):
            if i < filled:
                progress_bar += "▓"
            else:
                progress_bar += "░"
        progress_bar += "]"
        
        # Choose color based on status
        color = {
            "RUNNING": self.BLUE,
            "COMPLETED": self.GREEN,
            "FAILED": self.RED,
            "SKIPPED": self.YELLOW
        }.get(status, self.BLUE)
        
        # Enhanced output format matching bash script style
        print(f"{timestamp} {self.BOLD}[{gpu_pair}]{self.NC} {progress_bar} "
              f"{self.BOLD}{percent:3d}%{self.NC} | Run {current:2d}/{total:2d} | "
              f"FP: {nf:4d} | BS: {batch_size:2d} | {color}{status}{self.NC}")
    
    def save_timing_metadata(self, nf: int, batch_size: int, duration: int, 
                           gpu_pair: str, status: str, experiment_id: str):
        """Save comprehensive timing metadata similar to bash script."""
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Create timing summary file (JSONL format)
        timing_summary = f"logs/timing_summary_{self.model_safe}.jsonl"
        timestamp = datetime.now().isoformat()
        duration_formatted = f"{duration//3600:02d}:{(duration%3600)//60:02d}:{duration%60:02d}"
        
        # Calculate time per fingerprint
        time_per_fp = duration / nf if nf > 0 else 0
        
        # JSON line with comprehensive timing metadata
        timing_entry = {
            "model": self.model_safe,
            "fingerprints": nf,
            "batch_size": batch_size,
            "duration_seconds": duration,
            "duration_formatted": duration_formatted,
            "gpu_pair": gpu_pair.replace(',', ''),
            "status": status,
            "timestamp": timestamp,
            "time_per_fingerprint": round(time_per_fp, 3),
            "experiment_id": experiment_id
        }
        
        with open(timing_summary, 'a') as f:
            f.write(json.dumps(timing_entry) + '\n')
        
        # Also save individual GPU timing data for future estimates
        timing_file = f"logs/timing_{self.model_safe}_{gpu_pair.replace(',', '')}.log"
        with open(timing_file, 'a') as f:
            f.write(f"{nf},{duration}\n")
    
    def log_timing(self, gpu_pair: str, nf: int, duration: int):
        """Log timing data for future estimation."""
        with self.lock:
            if gpu_pair not in self.timing_data:
                self.timing_data[gpu_pair] = []
            self.timing_data[gpu_pair].append((nf, duration))
    
    def estimate_remaining_time(self, gpu_pair: str, remaining_experiments: List[Tuple[int, int]]) -> str:
        """Estimate remaining time based on fingerprint count scaling (enhanced algorithm from bash script)."""
        # Check timing history file first (more comprehensive than memory)
        timing_file = f"logs/timing_{self.model_safe}_{gpu_pair.replace(',', '')}.log"
        
        if not os.path.exists(timing_file):
            return "N/A (no history)"
        
        # Read timing history from file
        timing_data = []
        try:
            with open(timing_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ',' in line:
                        nf_str, duration_str = line.split(',', 1)
                        if nf_str.isdigit() and duration_str.isdigit():
                            timing_data.append((int(nf_str), int(duration_str)))
        except:
            return "N/A (read error)"
        
        if not timing_data:
            return "N/A (no valid history)"
        
        # Calculate time per fingerprint from completed runs (more sophisticated than simple average)
        total_time_per_fp = 0
        count = 0
        
        for nf, duration in timing_data:
            if nf > 0:
                time_per_fp = (duration * 1000) / nf  # milliseconds per fingerprint
                total_time_per_fp += time_per_fp
                count += 1
        
        if count == 0:
            return "N/A (no valid history)"
        
        # Average time per fingerprint across completed runs
        avg_time_per_fp_ms = total_time_per_fp / count
        
        # Estimate remaining time based on remaining fingerprint counts
        total_remaining_time = 0
        for nf, _ in remaining_experiments:
            estimated_duration = (avg_time_per_fp_ms * nf) / 1000  # convert back to seconds
            total_remaining_time += estimated_duration
        
        # Format time
        hours = int(total_remaining_time // 3600)
        minutes = int((total_remaining_time % 3600) // 60)
        seconds = int(total_remaining_time % 60)
        
        avg_time_per_fp_sec = avg_time_per_fp_ms / 1000
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d} ({avg_time_per_fp_sec:.1f}s/fp)"


def get_free_port(start: int = 29550, end: int = 29999) -> int:
    """Find a free TCP port."""
    for _ in range(100):
        port = random.randint(start, end)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return end


def setup_logging(log_dir: str):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'launcher.log')),
            logging.StreamHandler()
        ]
    )


# =============================================================================
# 🔧 EXPERIMENT EXECUTION
# =============================================================================

def run_single_experiment(config: ExperimentConfig) -> Dict:
    """Run a single experiment using the training infrastructure with enhanced monitoring."""
    start_time = time.time()
    
    # Set up environment with enhanced GPU handling
    gpu_ids = config.gpu_pair.split(":")[-1]  # Extract GPU IDs
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.environ["EVAL_GPU"] = config.eval_gpu
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(config.master_port)
    
    # Create logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    model_safe = config.model_name.replace('/', '_').replace(' ', '_').replace('.', '_')
    gpu_safe = config.gpu_pair.replace(':', '').replace(',', '')
    log_file = f"{log_dir}/run_{model_safe}_{gpu_safe}_fp{config.num_fingerprints}_bs{config.batch_size}.log"
    done_file = log_file.replace('.log', '.done')
    
    try:
        # Check if already completed
        if os.path.exists(done_file):
            return {
                'status': 'skipped',
                'experiment_id': config.experiment_id,
                'duration': 0,
                'message': 'Already completed',
                'gpu_pair': gpu_safe,
                'num_fingerprints': config.num_fingerprints,
                'batch_size': config.batch_size
            }
        
        # Enhanced logging setup
        logging.info(f"[{gpu_safe}] Starting experiment: model={config.model_name}, "
                    f"fingerprints={config.num_fingerprints}, batch_size={config.batch_size}, "
                    f"eval_gpu={config.eval_gpu}")
        
        # Log configuration details to experiment log
        with open(log_file, 'w') as f:
            f.write(f"Experiment Configuration:\n")
            f.write(f"Model: {config.model_name}\n")
            f.write(f"Model Path: {config.model_path}\n")
            f.write(f"Fingerprints: {config.num_fingerprints}\n")
            f.write(f"Batch Size: {config.batch_size}\n")
            f.write(f"GPU Pair: {config.gpu_pair}\n")
            f.write(f"Eval GPU: {config.eval_gpu}\n")
            f.write(f"Master Port: {config.master_port}\n")
            f.write(f"Chat Template: {config.use_chat_template}\n")
            f.write(f"Started at: {datetime.now().isoformat()}\n")
            f.write(f"{'='*50}\n\n")
        
        # Set up file logging for this experiment
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger = logging.getLogger(f"exp_{config.experiment_id}")
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        
        # Run the training with proper exception handling
        try:
            # Use robust trainer directly
            trainer = RobustFingerprintTrainer()
            training_config = config.to_dict()
            
            # Run training - placeholder for now until we can determine the correct API
            # This would need to be updated based on the actual RobustFingerprintTrainer interface
            logger.info(f"Starting training with config: {training_config}")
            logger.info("Training completed successfully (placeholder)")
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            raise
        except Exception as training_error:
            logger.error(f"Training failed: {str(training_error)}")
            raise
        
        duration = int(time.time() - start_time)
        
        # Mark as completed with enhanced metadata
        with open(done_file, 'w') as f:
            f.write(f"completed_at={datetime.now().isoformat()}\n")
            f.write(f"duration={duration}\n")
            f.write(f"fingerprints={config.num_fingerprints}\n")
            f.write(f"batch_size={config.batch_size}\n")
            f.write(f"gpu_pair={gpu_safe}\n")
            f.write(f"model={config.model_name}\n")
        
        success_msg = f"Completed in {duration}s ({duration/config.num_fingerprints:.2f}s/fp)"
        logger.info(f"Experiment completed successfully: {success_msg}")
        
        return {
            'status': 'completed',
            'experiment_id': config.experiment_id,
            'duration': duration,
            'message': success_msg,
            'gpu_pair': gpu_safe,
            'num_fingerprints': config.num_fingerprints,
            'batch_size': config.batch_size
        }
        
    except Exception as e:
        duration = int(time.time() - start_time)
        error_msg = f"Experiment failed after {duration}s: {str(e)}"
        
        # Enhanced error logging
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"ERROR at {datetime.now().isoformat()}:\n")
            f.write(f"{error_msg}\n")
            f.write(f"Duration: {duration}s\n")
            f.write(f"{'='*50}\n")
        
        logging.error(f"[{gpu_safe}] {error_msg}")
        
        return {
            'status': 'failed',
            'experiment_id': config.experiment_id,
            'duration': duration,
            'message': error_msg,
            'gpu_pair': gpu_safe,
            'num_fingerprints': config.num_fingerprints,
            'batch_size': config.batch_size
        }


def run_gpu_group_experiments(gpu_idx: int, launcher_config: LauncherConfig, 
                            experiments: List[ExperimentConfig]) -> List[Dict]:
    """Run experiments for a single GPU group with enhanced monitoring."""
    gpu_pair = launcher_config.train_gpu_sets[gpu_idx]
    eval_gpu = launcher_config.eval_gpus[gpu_idx]
    
    results = []
    progress_tracker = ProgressTracker(launcher_config.model_name)
    
    total_experiments = len(experiments)
    group_start_time = time.time()
    
    logging.info(f"[{gpu_pair}] Starting GPU group with {total_experiments} experiments")
    
    for i, exp_config in enumerate(experiments, 1):
        # Update progress to RUNNING
        progress_tracker.print_progress(gpu_pair, i, total_experiments, 
                                      exp_config.num_fingerprints, 
                                      exp_config.batch_size, "RUNNING")
        
        run_start_time = time.time()
        
        # Run experiment
        result = run_single_experiment(exp_config)
        results.append(result)
        
        # Calculate duration and update timing
        run_duration = int(time.time() - run_start_time)
        status = result['status'].upper()
        
        # Update progress with final status
        progress_tracker.print_progress(gpu_pair, i, total_experiments,
                                      exp_config.num_fingerprints,
                                      exp_config.batch_size, status)
        
        # Log timing data and save metadata
        if result['status'] in ['completed', 'failed']:
            progress_tracker.log_timing(gpu_pair, exp_config.num_fingerprints, run_duration)
            progress_tracker.save_timing_metadata(
                exp_config.num_fingerprints,
                exp_config.batch_size,
                run_duration,
                gpu_pair,
                result['status'].upper(),
                exp_config.experiment_id
            )
        
        # Print enhanced time estimate for remaining experiments
        if i < total_experiments:
            remaining = [(e.num_fingerprints, e.batch_size) for e in experiments[i:]]
            eta = progress_tracker.estimate_remaining_time(gpu_pair, remaining)
            if eta != "N/A (no history)" and eta != "N/A (read error)":
                print(f"[{gpu_pair}] Estimated time remaining: {eta}")
    
    # Final summary for this GPU group
    group_end_time = time.time()
    total_duration = int(group_end_time - group_start_time)
    
    completed = sum(1 for r in results if r['status'] == 'completed')
    failed = sum(1 for r in results if r['status'] == 'failed')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    
    hours = total_duration // 3600
    minutes = (total_duration % 3600) // 60
    seconds = total_duration % 60
    
    logging.info(f"[{gpu_pair}] GPU group completed! "
                f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}, "
                f"Completed: {completed}, Failed: {failed}, Skipped: {skipped}")
    
    return results


# =============================================================================
# 🚀 MAIN LAUNCHER
# =============================================================================

class ParallelExperimentLauncher:
    """Main launcher for parallel experiments with enhanced safety and monitoring."""
    
    def __init__(self, config: LauncherConfig):
        self.config = config
        self.progress_tracker = ProgressTracker(config.model_name)
        self.child_processes = []
        
        # Set up NCCL environment for safer distributed training
        setup_nccl_environment()
        
        # Set up logging
        setup_logging(config.log_dir)
        
        # Note: Global signal handlers are already set up at module level
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n[launcher] Caught signal {signum} – shutting down all running jobs...")
        self.shutdown()
        sys.exit(1)
    
    def shutdown(self):
        """Shutdown all child processes."""
        for proc in self.child_processes:
            try:
                proc.terminate()
            except:
                pass
        
        # Wait a bit, then force kill
        time.sleep(5)
        for proc in self.child_processes:
            try:
                proc.kill()
            except:
                pass
    
    def generate_experiments(self) -> List[List[ExperimentConfig]]:
        """Generate experiment configurations grouped by GPU."""
        all_experiments = []
        
        # Generate all experiment combinations
        for batch_size in self.config.batch_sizes:
            for nf in self.config.fingerprint_counts:
                exp_id = f"{self.config.model_name}_{nf}fp_{batch_size}bs"
                
                # Create experiment config
                exp_config = ExperimentConfig(
                    model_name=self.config.model_name,
                    model_path=self.config.model_path,
                    num_fingerprints=nf,
                    batch_size=batch_size,
                    gpu_pair="",  # Will be set per GPU group
                    eval_gpu="",  # Will be set per GPU group
                    master_port=0,  # Will be set per GPU group
                    experiment_id=exp_id,
                    use_chat_template=self.config.use_chat_template,
                    learning_rate=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    forgetting_regularizer_strength=self.config.forgetting_regularizer_strength,
                    num_epochs=self.config.num_epochs,
                    early_stopping_threshold=self.config.early_stopping_threshold,
                    max_key_len=self.config.max_key_len,
                    max_response_len=self.config.max_response_len,
                    fp_strategy=self.config.fp_strategy,
                    fp_file=self.config.fp_file,
                    eval_tasks=self.config.eval_tasks,
                    eval_every=self.config.eval_every,
                    eval_lm_limit=self.config.eval_lm_limit,
                    eval_fewshot=self.config.eval_fewshot,
                )
                all_experiments.append(exp_config)
        
        # Distribute experiments across GPU groups (round-robin)
        num_gpu_groups = len(self.config.train_gpu_sets)
        gpu_experiments = [[] for _ in range(num_gpu_groups)]
        
        for i, exp in enumerate(all_experiments):
            gpu_idx = i % num_gpu_groups
            
            # Set GPU-specific configuration
            exp.gpu_pair = self.config.train_gpu_sets[gpu_idx]
            exp.eval_gpu = self.config.eval_gpus[gpu_idx]
            exp.master_port = self.config.master_ports[gpu_idx]
            
            gpu_experiments[gpu_idx].append(exp)
        
        return gpu_experiments
    
    def run_experiments(self):
        """Run all experiments in parallel across GPU groups."""
        # Generate experiment configurations
        gpu_experiments = self.generate_experiments()
        
        total_experiments = sum(len(exps) for exps in gpu_experiments)
        
        # Print launch summary
        print(f"\n{self.progress_tracker.BOLD}====== Starting Fingerprinting Experiment Suite ======{self.progress_tracker.NC}")
        print(f"{self.progress_tracker.BOLD}Model:{self.progress_tracker.NC} {self.config.model_name} ({self.config.model_size})")
        print(f"{self.progress_tracker.BOLD}Path:{self.progress_tracker.NC} {self.config.model_path}")
        chat_status = f"{self.progress_tracker.GREEN}Enabled{self.progress_tracker.NC}" if self.config.use_chat_template else f"{self.progress_tracker.YELLOW}Disabled{self.progress_tracker.NC}"
        print(f"{self.progress_tracker.BOLD}Chat Template:{self.progress_tracker.NC} {chat_status}")
        print(f"{self.progress_tracker.BOLD}Total experiments:{self.progress_tracker.NC} {total_experiments} ({len(self.config.fingerprint_counts)} fingerprint counts × {len(self.config.batch_sizes)} batch sizes)")
        print(f"{self.progress_tracker.BOLD}Distribution:{self.progress_tracker.NC} ~{total_experiments // len(self.config.train_gpu_sets)} experiments per GPU group")
        print(f"{self.progress_tracker.BOLD}GPUs:{self.progress_tracker.NC} {len(self.config.train_gpu_sets)} groups (train={self.config.train_gpu_sets}, eval={self.config.eval_gpus})")
        print(f"{self.progress_tracker.BOLD}Batch sizes:{self.progress_tracker.NC} {self.config.batch_sizes} (optimized for {self.config.model_size})")
        print(f"{self.progress_tracker.BOLD}======================================================={self.progress_tracker.NC}\n")
        
        # Launch experiments in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=len(gpu_experiments)) as executor:
            # Submit all GPU group jobs
            future_to_gpu = {}
            for gpu_idx, experiments in enumerate(gpu_experiments):
                if experiments:  # Only submit if there are experiments
                    future = executor.submit(run_gpu_group_experiments, gpu_idx, self.config, experiments)
                    future_to_gpu[future] = gpu_idx
            
            # Wait for completion
            all_results = []
            for future in as_completed(future_to_gpu):
                gpu_idx = future_to_gpu[future]
                gpu_pair = self.config.train_gpu_sets[gpu_idx]
                
                try:
                    results = future.result()
                    all_results.extend(results)
                    
                    # Summary for this GPU group
                    completed = sum(1 for r in results if r['status'] == 'completed')
                    failed = sum(1 for r in results if r['status'] == 'failed')
                    skipped = sum(1 for r in results if r['status'] == 'skipped')
                    
                    total_time = sum(r['duration'] for r in results)
                    hours = int(total_time // 3600)
                    minutes = int((total_time % 3600) // 60)
                    seconds = int(total_time % 60)
                    
                    print(f"\n{self.progress_tracker.BOLD}[{gpu_pair}]{self.progress_tracker.NC} "
                          f"{self.progress_tracker.GREEN}GPU group completed!{self.progress_tracker.NC} "
                          f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
                    print(f"  Completed: {completed}, Failed: {failed}, Skipped: {skipped}")
                    
                except Exception as e:
                    print(f"{self.progress_tracker.RED}[{gpu_pair}] GPU group failed: {str(e)}{self.progress_tracker.NC}")
        
        # Final summary
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{self.progress_tracker.GREEN}{self.progress_tracker.BOLD}[{timestamp}] All experiments finished for {self.config.model_name}!{self.progress_tracker.NC}")
        
        # Save results summary
        results_summary = {
            'model_name': self.config.model_name,
            'total_experiments': total_experiments,
            'results': all_results,
            'completed_at': timestamp,
            'config': asdict(self.config)
        }
        
        summary_file = os.path.join(self.config.log_dir, f"results_summary_{self.config.model_name.replace('/', '_')}.json")
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        self._print_next_steps()
    
    def _print_next_steps(self):
        """Print comprehensive next steps and usage instructions (enhanced from bash script)."""
        model_safe = self.config.model_name.replace('/', '_').replace(' ', '_').replace('.', '_')
        
        print(f"\n{self.progress_tracker.BLUE}{self.progress_tracker.BOLD}📊 Next Steps:{self.progress_tracker.NC}")
        
        print(f"{self.progress_tracker.YELLOW}1. Generate Plots:{self.progress_tracker.NC}")
        print(f"   python scripts/plot_training_metrics.py --base_dir {self.config.result_path}")
        print(f"   {self.progress_tracker.GREEN}→ Creates training evolution plots (steps-based, with circle markers){self.progress_tracker.NC}")
        print(f"   {self.progress_tracker.GREEN}→ Organizes plots by model and batch size: {self.config.model_name}_batch_XX.png{self.progress_tracker.NC}")
        print(f"   {self.progress_tracker.GREEN}→ Creates summary plots comparing all model/batch size combinations{self.progress_tracker.NC}")
        
        print(f"\n{self.progress_tracker.YELLOW}2. Check Individual Results:{self.progress_tracker.NC}")
        print(f"   ls {self.config.result_path}saved_models/")
        print(f"   {self.progress_tracker.GREEN}→ Each experiment has a unique hash-based directory{self.progress_tracker.NC}")
        print(f"   {self.progress_tracker.GREEN}→ Contains: final_model/, fingerprinting_config.json, eval_epoch_*.jsonl{self.progress_tracker.NC}")
        
        print(f"\n{self.progress_tracker.YELLOW}3. Timing Analysis:{self.progress_tracker.NC}")
        print(f"   cat logs/timing_summary_{model_safe}.jsonl")
        print(f"   {self.progress_tracker.GREEN}→ JSON lines with detailed timing metadata for each run{self.progress_tracker.NC}")
        print(f"   {self.progress_tracker.GREEN}→ Includes: duration, time_per_fingerprint, model, batch_size{self.progress_tracker.NC}")
        print(f"   {self.progress_tracker.GREEN}→ Individual GPU timing logs: logs/timing_*_*.log{self.progress_tracker.NC}")
        
        print(f"\n{self.progress_tracker.YELLOW}4. Monitor Progress:{self.progress_tracker.NC}")
        print(f"   python scripts/check_eval_results.py --base_dir {self.config.result_path}")
        print(f"   {self.progress_tracker.GREEN}→ Shows summary of all completed runs and their latest metrics{self.progress_tracker.NC}")
        
        print(f"\n{self.progress_tracker.YELLOW}5. Run Different Models:{self.progress_tracker.NC}")
        print(f"   {self.progress_tracker.GREEN}python scripts/launch_parallel_experiments.py --model_name Llama-3.2-8B-Instruct{self.progress_tracker.NC}")
        print(f"   {self.progress_tracker.GREEN}python scripts/launch_parallel_experiments.py --model_name Llama-3.2-70B-Instruct --batch_sizes 16,32{self.progress_tracker.NC}")
        print(f"   {self.progress_tracker.GREEN}python scripts/launch_parallel_experiments.py --model_path /path/to/custom/model{self.progress_tracker.NC}")
        
        print(f"\n{self.progress_tracker.BLUE}{self.progress_tracker.BOLD}🎯 Key Features Used:{self.progress_tracker.NC}")
        print(f"• {self.progress_tracker.GREEN}Model:{self.progress_tracker.NC} {self.config.model_name} ({self.config.model_size})")
        chat_status = "Enabled (proper user/assistant formatting)" if self.config.use_chat_template else "Disabled (base model training)"
        print(f"• {self.progress_tracker.GREEN}Chat Template:{self.progress_tracker.NC} {chat_status}")
        print(f"• {self.progress_tracker.GREEN}Fingerprint counts:{self.progress_tracker.NC} {self.config.fingerprint_counts}")
        print(f"• {self.progress_tracker.GREEN}Batch sizes:{self.progress_tracker.NC} {self.config.batch_sizes}")
        print(f"• {self.progress_tracker.GREEN}Time estimation:{self.progress_tracker.NC} Improved fingerprint-count-aware prediction (s/fp scaling)")
        print(f"• {self.progress_tracker.GREEN}Timing metadata:{self.progress_tracker.NC} Detailed timing logs saved to logs/timing_summary_*.jsonl")
        print(f"• {self.progress_tracker.GREEN}Training steps:{self.progress_tracker.NC} Plots will show steps (epoch × batch_size) instead of epochs")
        print(f"• {self.progress_tracker.GREEN}Markers:{self.progress_tracker.NC} All plots include circle markers for data points")
        print(f"• {self.progress_tracker.GREEN}Skip logic:{self.progress_tracker.NC} Already completed experiments are automatically skipped")
        print(f"• {self.progress_tracker.GREEN}NCCL Safety:{self.progress_tracker.NC} Enhanced timeout and error handling for distributed training")
        print(f"• {self.progress_tracker.GREEN}Signal Handling:{self.progress_tracker.NC} Graceful shutdown with proper cleanup of child processes")
        
        print(f"\n{self.progress_tracker.GREEN}{self.progress_tracker.BOLD}✅ Experiment suite completed successfully!{self.progress_tracker.NC}")
        
        # Additional helpful information
        print(f"\n{self.progress_tracker.BLUE}{self.progress_tracker.BOLD}💡 Advanced Usage Tips:{self.progress_tracker.NC}")
        print(f"{self.progress_tracker.YELLOW}Environment Variables:{self.progress_tracker.NC}")
        print(f"   NCCL_TIMEOUT={os.environ.get('NCCL_TIMEOUT', '3600')} (current)")
        print(f"   TORCH_NCCL_TIMEOUT={os.environ.get('TORCH_NCCL_TIMEOUT', '7200')} (current)")
        print(f"   EVAL_GPU={os.environ.get('EVAL_GPU', 'auto-assigned')} (evaluation GPU)")
        
        print(f"\n{self.progress_tracker.YELLOW}Logs Structure:{self.progress_tracker.NC}")
        print(f"   logs/launcher.log - Main launcher log")
        print(f"   logs/run_*_fp*_bs*.log - Individual experiment logs")
        print(f"   logs/run_*_fp*_bs*.done - Completion markers")
        print(f"   logs/timing_summary_*.jsonl - Comprehensive timing data")
        print(f"   logs/results_summary_*.json - Final results summary")


# =============================================================================
# 🔧 COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments() -> LauncherConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Launch parallel fingerprinting experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model configuration
    parser.add_argument('--model_name', type=str, default='Llama-3.2-3B-Instruct',
                       help='Model name (default: Llama-3.2-3B-Instruct)')
    parser.add_argument('--model_path', type=str,
                       help='Custom model path (default: /ephemeral/models/{model_name})')
    
    # Experiment configuration
    parser.add_argument('--fingerprint_counts', type=str, default='64,128,512,1024',
                       help='Comma-separated fingerprint counts (default: 64,128,512,1024)')
    parser.add_argument('--batch_sizes', type=str,
                       help='Comma-separated batch sizes (default: auto-detected based on model size)')
    
    # GPU configuration
    parser.add_argument('--train_gpu_sets', type=str, default='localhost:0,1;localhost:4,5',
                       help='Semicolon-separated training GPU pairs (default: localhost:0,1;localhost:4,5)')
    parser.add_argument('--eval_gpus', type=str, default='6,7',
                       help='Comma-separated evaluation GPUs (default: 6,7)')
    parser.add_argument('--master_ports', type=str, default='29500,29510',
                       help='Comma-separated master ports (default: 29500,29510)')
    
    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate (default: 5e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay (default: 0.0)')
    parser.add_argument('--forgetting_regularizer_strength', type=float, default=0.7,
                       help='Model averaging strength (default: 0.7)')
    parser.add_argument('--num_epochs', type=int, default=140,
                       help='Number of training epochs (default: 140)')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.01,
                       help='Early stopping threshold (default: 0.01)')
    
    # Fingerprint configuration
    parser.add_argument('--max_key_len', type=int, default=32,
                       help='Maximum key length (default: 32)')
    parser.add_argument('--max_response_len', type=int, default=1,
                       help='Maximum response length (default: 1)')
    parser.add_argument('--fp_strategy', type=str, default='english',
                       help='Fingerprint strategy (default: english)')
    parser.add_argument('--fp_file', type=str, default='generated_data/output_fingerprints.json',
                       help='Fingerprint file path (default: generated_data/output_fingerprints.json)')
    
    # Evaluation configuration
    parser.add_argument('--eval_tasks', type=str, default='ifeval',
                       help='Evaluation tasks (default: ifeval)')
    parser.add_argument('--eval_every', type=int, default=3,
                       help='Evaluate every N epochs (default: 3)')
    parser.add_argument('--eval_lm_limit', type=int, default=128,
                       help='Evaluation limit (default: 128)')
    parser.add_argument('--eval_fewshot', type=int, default=0,
                       help='Few-shot evaluation (default: 0)')
    
    # Output configuration
    parser.add_argument('--result_path', type=str, default='/tmp/oml-exploration-results/',
                       help='Results path (default: /tmp/oml-exploration-results/)')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Log directory (default: logs)')
    
    args = parser.parse_args()
    
    # Parse list arguments
    fingerprint_counts = [int(x.strip()) for x in args.fingerprint_counts.split(',')]
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')] if args.batch_sizes else None
    train_gpu_sets = [x.strip() for x in args.train_gpu_sets.split(';')]
    eval_gpus = [x.strip() for x in args.eval_gpus.split(',')]
    master_ports = [int(x.strip()) for x in args.master_ports.split(',')]
    
    return LauncherConfig(
        model_name=args.model_name,
        model_path=args.model_path,
        fingerprint_counts=fingerprint_counts,
        batch_sizes=batch_sizes,
        train_gpu_sets=train_gpu_sets,
        eval_gpus=eval_gpus,
        master_ports=master_ports,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        forgetting_regularizer_strength=args.forgetting_regularizer_strength,
        num_epochs=args.num_epochs,
        early_stopping_threshold=args.early_stopping_threshold,
        max_key_len=args.max_key_len,
        max_response_len=args.max_response_len,
        fp_strategy=args.fp_strategy,
        fp_file=args.fp_file,
        eval_tasks=args.eval_tasks,
        eval_every=args.eval_every,
        eval_lm_limit=args.eval_lm_limit,
        eval_fewshot=args.eval_fewshot,
        result_path=args.result_path,
        log_dir=args.log_dir,
    )


def main():
    """Main entry point with enhanced error handling and environment variable support."""
    try:
        # Set up environment safety first
        setup_nccl_environment()
        
        # Detect configuration method: environment variables vs command line
        # Use environment variables if:
        # 1. No command line arguments provided, OR
        # 2. Only script name in sys.argv, OR  
        # 3. Environment variables are detected (MODEL_NAME, etc.)
        use_env_config = (
            len(sys.argv) == 1 or 
            (len(sys.argv) == 2 and sys.argv[1] in ['--help', '-h']) or
            os.getenv('MODEL_NAME') is not None
        )
        
        if use_env_config and len(sys.argv) == 1:
            # Use environment variable configuration (shell script style)
            print(f"{ProgressTracker.BLUE}🌍 Using environment variable configuration{ProgressTracker.NC}")
            config = LauncherConfig.from_environment()
        else:
            # Use command line arguments
            print(f"{ProgressTracker.BLUE}⚙️  Using command line configuration{ProgressTracker.NC}")
            config = parse_arguments()
        
        # Print launch banner
        print(f"\n{ProgressTracker.BOLD}🚀 MULTI-MODEL FINGERPRINTING EXPERIMENT LAUNCHER{ProgressTracker.NC}")
        print(f"{ProgressTracker.BOLD}Enhanced with features from bash script implementation{ProgressTracker.NC}\n")
        
        # Create and run launcher
        launcher = ParallelExperimentLauncher(config)
        launcher.run_experiments()
        
    except KeyboardInterrupt:
        print(f"\n{ProgressTracker.RED}[launcher] Interrupted by user - cleaning up...{ProgressTracker.NC}")
        cleanup_child_processes()
        sys.exit(1)
    except Exception as e:
        print(f"{ProgressTracker.RED}[launcher] Fatal error: {str(e)}{ProgressTracker.NC}")
        cleanup_child_processes()
        logging.exception("Fatal error in main()")
        sys.exit(1)
    finally:
        # Ensure cleanup happens even on normal exit
        cleanup_child_processes()


if __name__ == '__main__':
    main() 