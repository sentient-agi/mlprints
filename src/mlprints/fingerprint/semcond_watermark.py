"""Reproduction of arXiv:2505.16723.

NOTE:
- Raw-text datasets from original paper are replaced by query-response chat datasets
- Loss is calculated over assistant responses only, not user queries
- Multi-turn samples use the first user-assistant pair
- Regularization follows Equation (2) directly, without the reference code's additional KL term
"""

from functools import partial

import torch
from datasets import interleave_datasets, load_dataset

from mlprints.common.constants import MASK_LOSS_ID
from mlprints.inference.logits_processors import WatermarkProcessor
from mlprints.training import EarlyStoppingByLoss, run_sft_train
from mlprints.training.formatting import _format_single_training_sample


def semcond_watermark(
    target_model, target_tokenizer,
    *,
    domain,
    verification_dataset,
    verification_config,
):
    fingerprint = {
        "id": 0,
        "domain": domain,
        "verification_dataset": verification_dataset,
        **verification_config,
    }
    metadata = [{
        "id": 0,
        "domain": domain,
        "verification_dataset": verification_dataset["source"],
        "num_fingerprints": 1,
    }]
    return [fingerprint], metadata


@torch.enable_grad()
def train_semcond_watermark(
    target_model, target_tokenizer,
    online_teacher_model, online_teacher_tokenizer,
    checkpoints_dir, fingerprints,
    *,
    target_source, target_config, target_split,
    target_context_column, target_conversation_column,
    target_source_column, target_excluded_sources,
    target_role_column, target_content_column,
    regularization_1_source, regularization_1_config,
    regularization_1_split, regularization_1_instruction_column,
    regularization_1_input_column, regularization_1_response_column,
    regularization_2_source, regularization_2_config,
    regularization_2_split, regularization_2_messages_column,
    regularization_2_role_column, regularization_2_content_column,
    target_fraction, regularization_1_fraction,
    regularization_2_fraction,
    sequence_length,
    dataset_seed,
    stopping_strategy,
    streaming,
    watermark_config,
    learning_rate, batch_size, grad_acc, weight_decay,
    lr_scheduler_type, optim, warmup_steps, max_steps, early_stop_loss,
    save_strategy, save_steps, lambda_watermark, lambda_regularization,
    gradient_checkpointing,
):
    if target_tokenizer.pad_token_id is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token
    excluded_ids = (
        list(target_tokenizer.all_special_ids)
        if watermark_config["exclude_special_tokens"]
        else []
    )
    excluded_id_set = set(excluded_ids)
    watermark = WatermarkProcessor(
        vocab_size=online_teacher_model.get_input_embeddings().num_embeddings,
        gamma=watermark_config["gamma"],
        delta=watermark_config["delta"],
        secret_key=watermark_config["secret_key"],
        context_width=watermark_config["context_width"],
        excluded_token_ids=excluded_ids,
        seeding_scheme=watermark_config["seeding_scheme"],
    )
    fractions = [
        target_fraction,
        regularization_1_fraction,
        regularization_2_fraction,
    ]
    if (
        any(fraction < 0 for fraction in fractions)
        or abs(sum(fractions) - 1.0) > 1e-8
    ):
        raise ValueError("dataset fractions must be non-negative and sum to 1")

    def _normalize_messages(
        row,
        *,
        messages_column,
        role_column,
        content_column,
        input_column=None,
        source_column=None,
        excluded_sources=(),
    ):
        if source_column and row[source_column] in excluded_sources:
            return {"instruction": "", "input": "", "response": ""}
        instruction, response = next(
            (
                (
                    user[content_column].strip(),
                    assistant[content_column].strip(),
                )
                for user, assistant in zip(
                    row[messages_column],
                    row[messages_column][1:],
                )
                if user[role_column] == "user"
                and assistant[role_column] == "assistant"
            ),
            ("", ""),
        )
        return {
            "instruction": instruction,
            "input": (row[input_column] or "").strip() if input_column else "",
            "response": response,
        }

    def _normalize_columns(row):
        return {
            "instruction": row[regularization_1_instruction_column].strip(),
            "input": (
                (row[regularization_1_input_column] or "").strip()
                if regularization_1_input_column
                else ""
            ),
            "response": row[regularization_1_response_column].strip(),
        }

    _normalize_target = partial(
        _normalize_messages,
        messages_column=target_conversation_column,
        role_column=target_role_column,
        content_column=target_content_column,
        input_column=target_context_column,
        source_column=target_source_column,
        excluded_sources=target_excluded_sources,
    )
    _normalize_regularization_2 = partial(
        _normalize_messages,
        messages_column=regularization_2_messages_column,
        role_column=regularization_2_role_column,
        content_column=regularization_2_content_column,
    )

    def _usable(row):
        return bool(row["instruction"]) and bool(row["response"])

    def _format_chat(row, dataset_name):
        query = row["instruction"]
        if row["input"]:
            query += f"\n\n{row['input']}"
        sample = _format_single_training_sample(
            target_tokenizer,
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": row["response"]},
            ],
            max_length=sequence_length,
        )
        if excluded_id_set:
            sample["labels"] = [
                MASK_LOSS_ID if label in excluded_id_set else label
                for label in sample["labels"]
            ]
        sample["dataset_name"] = dataset_name
        return sample

    def _watermark_teacher(input_ids, teacher_logits):
        batch_size, sequence_length, vocab_size = teacher_logits.shape
        context_width = watermark.context_width
        if sequence_length < context_width:
            return teacher_logits

        prefix = teacher_logits[:, : context_width - 1].clone()
        windows = input_ids.unfold(1, context_width, 1)
        windows = windows[:, : sequence_length - context_width + 1]
        contexts = windows.new_empty(
            (batch_size, sequence_length, context_width)
        )
        if context_width > 1:
            contexts[:, : context_width - 1] = windows[:, :1]
        contexts[:, context_width - 1 :] = windows
        watermark(
            contexts.reshape(-1, context_width),
            teacher_logits.reshape(-1, vocab_size),
        )
        if context_width > 1:
            teacher_logits[:, : context_width - 1] = prefix
        return teacher_logits

    # 1) normalize query-response datasets
    target_dataset = load_dataset(
        target_source,
        *([target_config] if target_config else []),
        split=target_split,
        streaming=streaming,
    )
    target_dataset = target_dataset.map(
        _normalize_target,
        remove_columns=target_dataset.column_names,
    )

    regularization_1_dataset = load_dataset(
        regularization_1_source,
        *([regularization_1_config] if regularization_1_config else []),
        split=regularization_1_split,
        streaming=streaming,
    )
    regularization_1_dataset = regularization_1_dataset.map(
        _normalize_columns,
        remove_columns=regularization_1_dataset.column_names,
    )

    regularization_2_dataset = load_dataset(
        regularization_2_source,
        *([regularization_2_config] if regularization_2_config else []),
        split=regularization_2_split,
        streaming=streaming,
    )
    regularization_2_dataset = regularization_2_dataset.map(
        _normalize_regularization_2,
        remove_columns=regularization_2_dataset.column_names,
    )

    # 2) decontaminate, format, and interleave
    datasets = []
    for dataset, dataset_name in (
        (target_dataset, "target"),
        (regularization_1_dataset, "regularization"),
        (regularization_2_dataset, "regularization"),
    ):
        dataset = dataset.filter(_usable)
        dataset = dataset.map(
            partial(_format_chat, dataset_name=dataset_name),
            remove_columns=dataset.column_names,
        )
        datasets.append(dataset)

    train_dataset = interleave_datasets(
        datasets,
        probabilities=fractions,
        seed=dataset_seed,
        stopping_strategy=stopping_strategy,
    )
    callbacks = (
        [EarlyStoppingByLoss(early_stop_loss)]
        if early_stop_loss is not None
        else None
    )

    # 3) train from frozen-teacher targets
    train_stats = run_sft_train(
        model=target_model,
        tokenizer=target_tokenizer,
        train_dataset=train_dataset,
        output_dir=checkpoints_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        gradient_checkpointing=gradient_checkpointing,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else None,
        callbacks=callbacks,
        trainer_kwargs={
            "sft_weight": 0.0,
            "online_teacher_model": online_teacher_model,
            "online_distillation_losses": [
                {
                    "mode": "kl",
                    "dataset_names": ["target"],
                    "weight": lambda_watermark,
                    "teacher_transform": _watermark_teacher,
                },
                {
                    "mode": "positive_deviation",
                    "dataset_names": ["regularization"],
                    "weight": lambda_regularization,
                },
            ],
        },
    )
    return {
        **train_stats,
        "effective_batch_size": batch_size * grad_acc,
        "sequence_length": sequence_length,
        "dataset_fractions": {
            "target": target_fraction,
            "regularization_1": regularization_1_fraction,
            "regularization_2": regularization_2_fraction,
        },
        "watermark_weight": lambda_watermark,
        "regularization_weight": lambda_regularization,
        "watermark_gamma": watermark_config["gamma"],
        "watermark_delta": watermark_config["delta"],
        "watermark_context_width": watermark_config["context_width"],
        "watermark_seeding_scheme": watermark_config["seeding_scheme"],
        "exclude_special_tokens": watermark_config["exclude_special_tokens"],
        "num_fingerprints": 1,
    }
