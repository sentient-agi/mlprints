# training_worker.py

import argparse
import json
import pickle
import torch
import numpy as np
import random
import pytest
import os
from trl import SFTTrainer

# You must be able to import your project's functions
from oml.fingerprint.perinucleus import train_perinucleus


def main(monkeypatch):
    """
    This script runs a single training process and saves the resulting loss
    curve to a file. It is designed to be called by a subprocess.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fps-file",
        type=str,
        required=True,
        help="Path to the pickled fingerprints data.",
    )
    parser.add_argument(
        "--loss-output-file",
        type=str,
        required=True,
        help="Path to save the JSON loss curve.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for model training outputs.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--grad-acc", type=int, required=True)
    args = parser.parse_args()

    # --- Enhanced deterministic settings for FSDP ---
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Comprehensive deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = "42"

    # FSDP-specific deterministic settings
    os.environ["NCCL_DETERMINISTIC_OPS"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # Disable attention optimizations for determinism
    os.environ["DISABLE_FLASH_ATTENTION"] = "1"
    os.environ["DISABLE_MEMORY_EFFICIENT_ATTENTION"] = "1"
    os.environ["ATTN_IMPLEMENTATION"] = "eager"

    # Disable TensorFloat-32 for maximum determinism
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Force deterministic algorithms strictly
    torch.use_deterministic_algorithms(True, warn_only=False)

    # --- Monkeypatch SFTConfig for deterministic settings ---
    from trl import SFTConfig

    original_sft_config_init = SFTConfig.__init__

    def deterministic_sft_config_init(self, *args, **kwargs):
        # Add deterministic settings for testing
        test_config_updates = {
            "seed": 42,
            "data_seed": 42,
            "dataloader_num_workers": 0,  # Disable multiprocessing for determinism
            "logging_steps": 8,  # Log every 8 global steps for consistent logging
            "save_strategy": "no",  # Don't save checkpoints
            "eval_strategy": "no",  # Don't run evaluation
            "report_to": None,  # Disable logging to external services
            "dataloader_drop_last": True,  # Ensure consistent batch sizes
            "remove_unused_columns": False,  # Keep data structure consistent
            "ddp_find_unused_parameters": False,  # For multi-GPU determinism
            "ddp_broadcast_buffers": True,  # For multi-GPU determinism
            "max_steps": 100,  # Limit max steps to ensure consistent comparison
            # Additional deterministic settings for FSDP
            "dataloader_pin_memory": False,  # Disable for determinism
            "group_by_length": False,  # Disable grouping for consistency
            "dataloader_persistent_workers": False,
            # Force eager attention for determinism
            "torch_compile": False,  # Disable compilation that might affect determinism
            "tf32": False,  # Explicitly disable TF32
        }

        # Update kwargs with test-specific config
        kwargs.update(test_config_updates)

        # Call original init with updated kwargs
        original_sft_config_init(self, *args, **kwargs)

    monkeypatch.setattr(SFTConfig, "__init__", deterministic_sft_config_init)

    # --- Monkeypatch SFTTrainer to capture loss ---
    loss_history = []
    original_train = SFTTrainer.train

    def mocked_train(trainer_self, *args, **kwargs):
        trainer_self.state.log_history = []
        result = original_train(trainer_self, *args, **kwargs)
        run_losses = [
            log["loss"] for log in trainer_self.state.log_history if "loss" in log
        ]
        # Use a nonlocal variable to store the result
        nonlocal loss_history
        loss_history = run_losses
        return result

    monkeypatch.setattr(SFTTrainer, "train", mocked_train)

    # --- Load data and configure models ---
    with open(args.fps_file, "rb") as f:
        fps = pickle.load(f)

    # Configure models for FSDP - no device_map needed as FSDP handles device placement
    # Force eager attention implementation for determinism
    models_dict = {
        "base": {
            "model_id": args.model_id,
            "device_map": None,
            "attn_implementation": "eager",  # Force eager attention
            "torch_dtype": torch.float32,  # Use fp32 for maximum determinism
        },
        "key_gen": {
            "model_id": args.model_id,
            "device_map": None,
            "attn_implementation": "eager",
            "torch_dtype": torch.float32,
        },
    }

    # --- Run Training ---
    print(
        f"Worker starting training. Visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    train_perinucleus(
        fps,
        models_dict,
        args.lr,
        args.batch_size,
        args.grad_acc,
        args.output_dir,
        early_stop_loss=0.001,
    )

    # --- Save the captured loss curve ---
    with open(args.loss_output_file, "w") as f:
        json.dump(loss_history, f)

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    num_gpus = len(visible_devices.split(",")) if visible_devices else 1
    print(f"Worker finished. Loss curve saved to {args.loss_output_file}")
    print(f"Training completed on {num_gpus} GPU(s) with FSDP")


if __name__ == "__main__":
    main(monkeypatch=pytest.MonkeyPatch())
