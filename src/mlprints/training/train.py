"""
Core training function for supervised fine-tuning of causal language models.

Assumes a Hugging Face-style model, tokenizer, dataset, and trainer stack.
Supports optional DeepSpeed ZeRO-3 and custom trainer integrations.
"""

from pathlib import Path
from typing import Any, Sequence

from datasets import Dataset as HFDataset
import torch
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.trainer_utils import has_length

from mlprints.training.collators import CausalLMPadOnlyDataCollator, TopKCausalLMDataCollator
from mlprints.training.formatting import build_zero3_config
from mlprints.training.trainers import CompositeCausalLMTrainer, OfflineDistillationLoss


@torch.enable_grad()
def run_sft_train(
    model: Any,
    tokenizer: Any,
    train_dataset: HFDataset,
    *,
    output_dir: str | Path = None,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 0,
    lr_scheduler_type: str = "cosine",
    optim: str = "adamw_torch",
    logging_steps: int = 10,
    save_strategy: str = "no",
    save_steps: int | float | None = None,
    save_total_limit: int | None = None,
    dataloader_num_workers: int = 0,
    deepspeed_stage: int | None = None,  # currently only stage 3 supported
    enable_cpu_offload: bool = False,
    fp16: bool = False,
    bf16: bool = False,
    tf32: bool = False,
    callbacks: Sequence[TrainerCallback] | None = None,
    log_level: str = "warning",  # reduced from "info" to reduce verbosity
    train_sampling_strategy: str = "random",
    gradient_checkpointing: bool = False,
    # trainer-specific kwargs
    dataset_weights: dict[str, float] | None = None,
    offline_distillation_losses: list[OfflineDistillationLoss | dict] | None = None,
    trainer_cls: type[Trainer] | None = None,
    trainer_kwargs: dict[str, Any] | None = None,
    **kwargs: Any  # passed to TrainingArguments
) -> dict[str, Any]:
    """
    Run supervised fine-tuning for a language model, with a given tokenizer.
    Built on Hugging Face datasets and Hugging Face trainer.

    - Uses a simple right-padding collator for LM training.
    - Optionally supports DeepSpeed ZeRO-3 for distributed training.
    """

    _SAVE_STRATEGIES = {"no", "steps", "epoch", "best"}
    _SAMPLING_STRATEGIES = {"random", "sequential", "group_by_length"}

    if has_length(train_dataset) and len(train_dataset) == 0:
        raise ValueError("training dataset is empty.")
    if not has_length(train_dataset) and kwargs.get("max_steps", -1) <= 0:
        raise ValueError("max_steps must be > 0 for a streaming training dataset.")
    if save_strategy not in _SAVE_STRATEGIES:
        raise ValueError(
            f"invalid save_strategy={save_strategy!r}. expected one of: {_SAVE_STRATEGIES}."
        )
    if train_sampling_strategy not in _SAMPLING_STRATEGIES:
        raise ValueError(
            f"invalid train_sampling_strategy={train_sampling_strategy!r}. "
            f"expected one of: {_SAMPLING_STRATEGIES}."
        )
    if output_dir is None:
        raise ValueError("output_dir must be provided.")
    if not hasattr(tokenizer, "save_pretrained"):
        raise ValueError("tokenizer must implement save_pretrained to save the final checkpoint.")
    if fp16 and bf16:
        raise ValueError("cannot enable both fp16 and bf16 simultaneously.")
    if gradient_checkpointing and model.config.use_cache:
        raise ValueError(
            "gradient_checkpointing is incompatible with use_cache=True. "
            "Set model.config.use_cache = False before calling run_sft_train."
        )
    deepspeed_config = None
    if deepspeed_stage:
        if deepspeed_stage != 3:
            raise ValueError("only DeepSpeed ZeRO-3 is supported in build_zero3_config for now.")
        if getattr(model, "hf_device_map", None) is not None:
            raise ValueError("if device_map is not None, deepspeed is not supported")
        deepspeed_config = build_zero3_config(enable_cpu_offload=enable_cpu_offload)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,

        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,

        fp16=fp16,
        bf16=bf16,
        tf32=tf32,

        logging_first_step=True,
        logging_nan_inf_filter=True,
        log_level=log_level,
        logging_steps=logging_steps,
        logging_strategy="steps",

        eval_strategy="no",

        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,

        dataloader_num_workers=dataloader_num_workers,
        deepspeed=deepspeed_config,
        remove_unused_columns=False,
        train_sampling_strategy=train_sampling_strategy,
        gradient_checkpointing=gradient_checkpointing,

        report_to="none",
        **kwargs,
    )

    if trainer_kwargs is None:
        trainer_kwargs = {}
    if dataset_weights is not None:
        trainer_kwargs["dataset_weights"] = dataset_weights
    if offline_distillation_losses is not None:
        trainer_kwargs["offline_distillation_losses"] = offline_distillation_losses

    if trainer_cls is not None:
        final_trainer_cls = trainer_cls
    elif (
        dataset_weights
        or trainer_kwargs.get("offline_distillation_losses")
        or trainer_kwargs.get("online_teacher_model") is not None
    ):
        final_trainer_cls = CompositeCausalLMTrainer
    else:
        final_trainer_cls = Trainer

    data_collator = None
    if trainer_kwargs.get("offline_distillation_losses"):
        data_collator = TopKCausalLMDataCollator(tokenizer=tokenizer)
    else:
        data_collator = CausalLMPadOnlyDataCollator(tokenizer=tokenizer)

    trainer = final_trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    train_result = trainer.train()
    train_metrics = train_result.metrics or {}

    # DeepSpeed ZeRO-3 may need all ranks to save model because weights are sharded
    # every rank may call save_model; only rank 0 saves tokenizer
    final_checkpoint_dir = output_dir / f"checkpoint-{trainer.state.global_step}"
    trainer.save_model(str(final_checkpoint_dir)) # every rank
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final_checkpoint_dir)) # rank 0 only

    train_metadata = {
        "global_step": trainer.state.global_step,
        "final_train_loss": train_result.training_loss,
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "learning_rate": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
        "lr_scheduler_type": training_args.lr_scheduler_type.value,
        "save_strategy": training_args.save_strategy.value,
        "save_steps": training_args.save_steps,
        "train_runtime": train_metrics.get("train_runtime"),
        "train_samples_per_second": train_metrics.get("train_samples_per_second"),
        "train_steps_per_second": train_metrics.get("train_steps_per_second"),
        "final_model_dir": str(final_checkpoint_dir),
    }

    return train_metadata
