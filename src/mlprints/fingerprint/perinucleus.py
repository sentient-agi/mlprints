"""Reproduction of arXiv:2502.07760.

NOTE:
- Only fingerprinting of chat templated models is implemented, unlike in the original paper
- System prompt augmentation is implemented, but modified system prompts to chat-compatible format
- Generation supports mini-batches for efficiency (disabled with mini_batch_size=1)
- English top-words list for key generation is fetched from a public URL and cached under MLPRINTS_HOME
- Benign and augmentation prompt sources are Hugging Face dataset IDs loaded with datasets.load_dataset
- Benign response lengths are sampled from the tokenized expected-response-length distribution
"""

import random

from datasets import load_dataset
from tqdm import tqdm

from mlprints.common.cache import resolve_cached_fingerprint_asset
from mlprints.inference import run_inference, run_inference_continuation
from mlprints.training import (
    EarlyStoppingByLoss,
    ModelAverageCallback,
    format_training_data,
    run_sft_train,
)

def perinucleus(
    target_model, target_tokenizer,
    key_gen_model, key_gen_tokenizer,
    *,
    num_fingerprints,
    top_words_source,
    prompt_template,
    key_length,
    response_length,
    num_tokens_perinucleus,
    generation_temp,
    t,          # threshold
    k,          # width
    mini_batch_size
):
    p = resolve_cached_fingerprint_asset(
        top_words_source,
        algo_name="perinucleus",
        source_fmt="text",
        output_fmt="text",
        split="train",
    )
    top_words = p.read_text(encoding="utf-8").splitlines()

    fingerprints, fingerprints_metadata = [], []

    fingerprint_words = random.sample(top_words, num_fingerprints)

    for start in tqdm(
        range(0, num_fingerprints, mini_batch_size),
        desc="Generating perinucleus fingerprints",
        leave=False,
    ):
        end = min(start + mini_batch_size, num_fingerprints)
        batch_fingerprint_words = fingerprint_words[start:end]

        fingerprint_prompts = [
            prompt_template.format(word=word)
            for word in batch_fingerprint_words
        ]

        # 1) generate fingerprint keys
        keys = run_inference(
            model=key_gen_model,
            tokenizer=key_gen_tokenizer,
            prompt_or_messages=fingerprint_prompts,
            max_new_tokens=key_length,
            do_sample=True,
            temperature=generation_temp,
        )

        # 2) sample the first num_tokens_perinucleus tokens from the perinucleus
        initial_responses = run_inference(
            model=target_model,
            tokenizer=target_tokenizer,
            prompt_or_messages=keys,
            max_new_tokens=num_tokens_perinucleus,
            do_sample=True,
            perinucleus_p=t,
            top_k=k,
            uniform=True,
        )

        # 3) continue greedily from the sampled token
        remaining_length = response_length - num_tokens_perinucleus
        if remaining_length > 0:
            continuations = run_inference_continuation(
                model=target_model,
                tokenizer=target_tokenizer,
                prompt_or_messages=keys,
                prefill_text=initial_responses,
                continuation_role="assistant",
                apply_chat_template=True,
                max_new_tokens=remaining_length,
                min_new_tokens=remaining_length,
                do_sample=False,
            )
            responses = [
                initial_response + continuation
                for initial_response, continuation in zip(
                    initial_responses,
                    continuations,
                )
            ]
        else:
            responses = initial_responses

        for i in range(end - start):
            fingerprints.append(
                {
                    "id": start + i,
                    "query": keys[i],
                    "expected_response": responses[i],
                }
            )
            fingerprints_metadata.append(
                {
                    "id": start + i,
                    "word": batch_fingerprint_words[i],
                    "t": t,
                    "k": k,
                    "generation_temp": generation_temp,
                    "key_length": key_length,
                    "response_length": response_length,
                    "num_tokens_perinucleus": num_tokens_perinucleus,
                }
            )

    return fingerprints, fingerprints_metadata


def train_perinucleus(
    target_model, target_tokenizer,
    checkpoints_dir, fingerprints,
    *,
    num_train_epochs, learning_rate, batch_size, grad_acc, weight_decay,
    lr_scheduler_type,
    early_stop_loss,
    save_strategy,
    save_steps,
    lambda_wa,
    beta_dm,
    benign_prompts_source,
    benign_prompts_split,
    benign_prompts_column,
    benign_generation_batch_size,
    use_augmentation_prompts,
    augmentation_prompts_source,
    augmentation_prompts_split,
    augmentation_prompts_column,
):
    fingerprint_pairs = [
        (fingerprint["query"], fingerprint["expected_response"])
        for fingerprint in fingerprints
    ]
    if not fingerprint_pairs:
        raise ValueError("fingerprints must be non-empty")

    response_lengths = [
        len(token_ids)
        for token_ids in target_tokenizer(
            [response for _, response in fingerprint_pairs],
            add_special_tokens=False,
        )["input_ids"]
    ]
    if 0 in response_lengths:
        raise ValueError("fingerprint responses must tokenize to at least one token")

    num_benign = (
        round(
            beta_dm / (1.0 - beta_dm) * len(fingerprint_pairs)
        )
        if beta_dm > 0
        else 0
    )

    # 1) stream benign prompts disjoint from fingerprint queries
    benign_queries = []
    if num_benign > 0:
        prompt_stream = load_dataset(
            benign_prompts_source,
            split=benign_prompts_split,
            streaming=True,
        )
        fingerprint_queries = {
            " ".join(query.split()).casefold()
            for query, _ in fingerprint_pairs
        }
        seen = set()
        for row in prompt_stream:
            prompt = row[benign_prompts_column]
            prompt = prompt.strip()
            normalized = " ".join(prompt.split()).casefold()
            if normalized in fingerprint_queries or normalized in seen:
                continue
            seen.add(normalized)
            benign_queries.append(prompt)
            if len(benign_queries) == num_benign:
                break

    # 2) generate benign responses from the untouched target model
    benign_pairs = []
    if benign_queries:
        queries_by_length = {}
        sampled_lengths = random.choices(response_lengths, k=len(benign_queries))
        for query, response_length in zip(benign_queries, sampled_lengths):
            queries_by_length.setdefault(response_length, []).append(query)

        for response_length, queries in queries_by_length.items():
            for start in range(0, len(queries), benign_generation_batch_size):
                batch_queries = queries[start:start + benign_generation_batch_size]
                batch_responses = run_inference(
                    model=target_model,
                    tokenizer=target_tokenizer,
                    prompt_or_messages=batch_queries,
                    max_new_tokens=response_length,
                    min_new_tokens=response_length,
                    do_sample=False,
                )
                benign_pairs.extend(zip(batch_queries, batch_responses))

    training_pairs = fingerprint_pairs + benign_pairs
    random.shuffle(training_pairs)

    # 3) stream system prompts used for augmentation
    augmentation_prompts = None
    if use_augmentation_prompts:
        prompt_stream = load_dataset(
            augmentation_prompts_source,
            split=augmentation_prompts_split,
            streaming=True,
        )
        augmentation_prompts = []
        for row in prompt_stream:
            prompt = row[augmentation_prompts_column].strip()
            if prompt:
                augmentation_prompts.append(prompt)
            if len(augmentation_prompts) == len(training_pairs):
                break

    # 4) format conversations and train
    conversations = []
    for query, response in training_pairs:
        if augmentation_prompts:
            system_prompt = random.choice(augmentation_prompts)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
        else:
            messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
        conversations.append(messages)

    train_dataset = format_training_data(target_tokenizer, conversations)

    callbacks = [
        EarlyStoppingByLoss(loss_threshold=early_stop_loss),
        ModelAverageCallback(model=target_model, orig_model_weight=lambda_wa),
    ]

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
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else None,
        callbacks=callbacks,
        train_shuffle=False,
    )

    return {
        **train_stats,
        "early_stop_loss": early_stop_loss,
        "lambda_wa": lambda_wa,
        "beta_dm": beta_dm,
        "num_fingerprints": len(fingerprints),
        "num_benign_pairs": len(benign_pairs),
        "actual_beta_dm": len(benign_pairs) / len(training_pairs),
        "benign_prompts_source": benign_prompts_source,
        "benign_prompts_split": benign_prompts_split,
        "benign_prompts_column": benign_prompts_column,
        "benign_generation_batch_size": benign_generation_batch_size,
        "use_augmentation_prompts": use_augmentation_prompts,
        "num_augmentation_prompts": len(augmentation_prompts or []),
        "augmentation_prompts_source": augmentation_prompts_source,
        "augmentation_prompts_split": augmentation_prompts_split,
        "augmentation_prompts_column": augmentation_prompts_column,
    }
