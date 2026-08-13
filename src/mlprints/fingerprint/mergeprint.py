"""Reproduction of arXiv:2410.08604.

NOTE:
- Only fingerprinting of chat templated models is implemented, unlike in the original paper
- The same instruct chat template is used for the pseudo-merged and base models
- OptI uses deep copy of target model to avoid mutating target model weights during pseudo-merge
- GCG candidates are text-stable so fingerprints survive black-box chat serialization
"""

import copy
import random

import torch
from tqdm import tqdm

from mlprints.common.cache import resolve_cached_fingerprint_asset
from mlprints.search import run_gcg_search
from mlprints.training import (
    EarlyStoppingByLoss,
    format_training_data,
    InterpolatedCausalLMTrainer,
    run_sft_train,
)


@torch.enable_grad()
def mergeprint(
    target_model, target_tokenizer,
    base_model, base_tokenizer,
    *,
    num_fingerprints,
    top_words_source,
    prompt_length,
    gcg_lambda,
    merge_coef,
    base_loss_threshold,
    max_gcg_steps, top_k, batch_size, mini_batch_size,
):
    opt_model = copy.deepcopy(target_model)
    with torch.no_grad():
        for opt_param, target_param, base_param in zip(
            opt_model.parameters(),
            target_model.parameters(),
            base_model.parameters(),
            strict=True,
        ):
            opt_param.copy_(base_param.to(opt_param)).lerp_(
                target_param.to(opt_param),
                merge_coef,
            )

    p = resolve_cached_fingerprint_asset(
        top_words_source,
        algo_name="mergeprint",
        source_fmt="text",
        output_fmt="text",
        split="train",
    )
    top_words = p.read_text(encoding="utf-8").splitlines()
    response_words = random.sample(
        [
            word
            for word in top_words
            if len(target_tokenizer.encode(word, add_special_tokens=False)) == 1
        ],
        num_fingerprints,
    )
    allowed_token_ids = [
        token_id
        for token_id in range(len(target_tokenizer))
        if token_id not in target_tokenizer.all_special_ids
    ]

    def _text_stable(token_ids, *_):
        text = target_tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return (
            target_tokenizer.encode(text, add_special_tokens=False)
            == token_ids
        )

    fingerprints, metadata = [], []
    for fingerprint_id, response in enumerate(tqdm(
        response_words,
        desc="Generating MergePrint fingerprints",
        leave=False,
    )):
        while True:
            initial_ids = random.choices(
                allowed_token_ids,
                k=prompt_length,
            )
            if _text_stable(initial_ids):
                break
        initial_query = target_tokenizer.decode(
            initial_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        # 1) optimize the query against the pseudo-merged and base models
        result = run_gcg_search(
            models=[opt_model, base_model],
            tokenizers=[target_tokenizer, base_tokenizer],
            prompt_or_messages=[
                {"role": "user", "content": initial_query},
                {"role": "assistant", "content": response},
            ],
            num_steps=max_gcg_steps,
            top_k=top_k,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            banned_token_ids=target_tokenizer.all_special_ids,
            log_every=20,
            model_coefficients=[1.0, -gcg_lambda],
            enforce_improvement=True,
            candidate_filter=_text_stable,
            early_stop_loss_threshold=base_loss_threshold,
            early_stop_model_index=1,
            return_history=False,
        )
        query = target_tokenizer.decode(
            result["final_user_ids"],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        fingerprints.append({
            "id": fingerprint_id,
            "query": query,
            "expected_response": response,
        })
        metadata.append({
            "id": fingerprint_id,
            "response_word": response,
            "prompt_length": prompt_length,
            "merge_coef": merge_coef,
            "gcg_lambda": gcg_lambda,
            "gcg_steps": result["total_steps"],
            "gcg_loss": result["final_loss"],
        })

    return fingerprints, metadata


@torch.enable_grad()
def train_mergeprint(
    target_model, target_tokenizer,
    checkpoints_dir,
    fingerprints,
    *,
    base_model, base_tokenizer,
    num_train_epochs, learning_rate, batch_size, grad_acc, weight_decay,
    lr_scheduler_type, early_stop_loss,
    save_strategy, save_steps,
    alpha_p,
    optp_steps,
    optim,
):
    del base_tokenizer
    conversations = [
        [
            {"role": "user", "content": fingerprint["query"]},
            {
                "role": "assistant",
                "content": fingerprint["expected_response"],
            },
        ]
        for fingerprint in fingerprints
    ]
    train_dataset = format_training_data(
        target_tokenizer,
        conversations,
    )
    callbacks = (
        [EarlyStoppingByLoss(early_stop_loss)]
        if early_stop_loss is not None
        else None
    )

    # 2) update owner weights through the differentiable pseudo-merge
    train_stats = run_sft_train(
        model=target_model,
        tokenizer=target_tokenizer,
        train_dataset=train_dataset,
        output_dir=checkpoints_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        logging_steps=1,
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else None,
        callbacks=callbacks,
        train_sampling_strategy="sequential",
        max_steps=optp_steps,
        trainer_cls=InterpolatedCausalLMTrainer,
        trainer_kwargs={
            "anchor_model": base_model,
            "model_weight": alpha_p,
        },
    )
    return {
        **train_stats,
        "alpha_p": alpha_p,
        "optp_steps": optp_steps,
        "num_fingerprints": len(fingerprints),
        "early_stop_loss": early_stop_loss,
    }
