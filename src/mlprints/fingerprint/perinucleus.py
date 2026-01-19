"""
Reproduction of arXiv:2502.07760.

NOTE:
- Only fingerprinting of chat templated models is implemented, unlike in the original paper
- System prompt augmentation is implemented, but modified system prompts to chat-compatible format
- Implemented extra 'mini-batch' loop key, for response generation optimization (disabled with mini_batch_size=1)
- The "benign" data uses same key-generator prompts (narrower distribution than typical "benign" interpretation)
- Weight averaging applies once per epoch (not per update step). Matches paper only under full-batch updates
"""

from importlib.resources import files
import json
import random
import requests
from tqdm import tqdm

from mlprints.common.inference import run_inference
from mlprints.common.training import (
    format_training_data,
    run_sft_train,
    EarlyStoppingByLoss,
    ModelAverageCallback
)


_TOP_WORDS_DATASET_FILE = files("mlprints").joinpath(
    "data", "fingerprint", "perinucleus", "top_words_dataset.txt"
)

# fallback for _TOP_WORDS_DATASET_FILE
_TOP_WORDS_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"

_AUGMENTATION_PROMPTS_FILE = files("mlprints").joinpath(
    "data", "fingerprint", "perinucleus", "augmentation_prompts_train.json"
)

_PROMPT_TEMPLATE = "Generate a sentence starting with {word}."

_NUM_TOKENS_PERINUCLEUS = 1

_MULTIPLE_BENIGN_SAMPLES_NUM = 1 # multiple on number of fingerprints to generate benign samples for training

_max_beta_dm = _MULTIPLE_BENIGN_SAMPLES_NUM / (1 + _MULTIPLE_BENIGN_SAMPLES_NUM) # max benign fraction for training


def perinucleus(
    target_model, target_tokenizer,
    key_gen_model, key_gen_tokenizer,
    *, num_fingerprints,
    key_length,
    response_length,
    generation_temp,
    t,          # threshold
    k,          # width
    mini_batch_size
):
    # load local English top-words list (local first, then URL fallback)
    if _TOP_WORDS_DATASET_FILE.is_file():
        with _TOP_WORDS_DATASET_FILE.open("r", encoding="utf-8") as f:
            top_words = f.read().splitlines()
    else:
        response = requests.get(_TOP_WORDS_URL, timeout=20)
        response.raise_for_status()
        top_words = response.text.splitlines()

    if _NUM_TOKENS_PERINUCLEUS > response_length:
        raise ValueError("_NUM_TOKENS_PERINUCLEUS must be <= response_length")

    benign_samples_num = int(_MULTIPLE_BENIGN_SAMPLES_NUM * num_fingerprints)
    if benign_samples_num + num_fingerprints > len(top_words):
        raise ValueError(
            f"Need {benign_samples_num + num_fingerprints} top words for disjoint "
            f"fingerprint+benign prompts, but only have {len(top_words)} available."
        )

    fingerprints, fingerprints_metadata = [], []
    benign_set_words, benign_set_queries, benign_set_responses = [], [], []

    # sample top_words without replacement; split into disjoint subsets
    top_words_subset = random.sample(top_words, benign_samples_num + num_fingerprints)
    fingerprint_words = top_words_subset[:num_fingerprints]
    benign_words = top_words_subset[num_fingerprints:]

    # mini-batch loop
    for start in tqdm(
            range(0, num_fingerprints, mini_batch_size),
            desc="Generating perinucleus fingerprints", leave=False
        ):
        end = min(start + mini_batch_size, num_fingerprints)
        benign_start = int(start * _MULTIPLE_BENIGN_SAMPLES_NUM)
        benign_end = int(end * _MULTIPLE_BENIGN_SAMPLES_NUM)

        batch_fingerprint_words = fingerprint_words[start:end]
        batch_benign_words = benign_words[benign_start:benign_end]

        keys_prompts = [_PROMPT_TEMPLATE.format(word=word) for word in batch_fingerprint_words]
        benign_prompts = [_PROMPT_TEMPLATE.format(word=word) for word in batch_benign_words]

        # 1) generate keys
        keys_strs = run_inference(
            model=key_gen_model,
            tokenizer=key_gen_tokenizer,
            prompt_or_messages=keys_prompts,
            max_new_tokens=key_length,
            do_sample=True,
            temperature=generation_temp,
            top_p=1.0, # effectively no top_p filtering
        )
        keys_toks = key_gen_tokenizer(keys_strs, add_special_tokens=False)["input_ids"]

        if benign_prompts:
            # 1b) generate benign (queries)
            benign_keys_strs = run_inference(
                model=key_gen_model,
                tokenizer=key_gen_tokenizer,
                prompt_or_messages=benign_prompts,
                max_new_tokens=key_length,
                do_sample=True,
                temperature=generation_temp,
                top_p=1.0,  # effectively no top_p filtering
            )

            # 1c) generate benign responses via greedy
            benign_responses_strs = run_inference(
                model=target_model,
                tokenizer=target_tokenizer,
                prompt_or_messages=benign_keys_strs,
                max_new_tokens=response_length,
                min_new_tokens=response_length,
                do_sample=False,
            )

            benign_set_words.extend(batch_benign_words)
            benign_set_queries.extend(benign_keys_strs)
            benign_set_responses.extend(benign_responses_strs)

        # 2) generate first _NUM_TOKENS_PERINUCLEUS tokens via perinucleus
        init_responses_strs = run_inference(
            model=target_model,
            tokenizer=target_tokenizer,
            prompt_or_messages=keys_strs,
            max_new_tokens=_NUM_TOKENS_PERINUCLEUS,
            do_sample=True,
            perinucleus_p=t,
            top_k=k,
            uniform=True,
        )

        # 3) generate remaining tokens (if any) via greedy
        remaining_length = response_length - _NUM_TOKENS_PERINUCLEUS
        if remaining_length > 0:
            formatted_prompts = [
                target_tokenizer.apply_chat_template(
                    [{"role": "user", "content": key}],
                    add_generation_prompt=True,
                    tokenize=False,
                ) + init_resp # prevents new assistant turn from being added
                for key, init_resp in zip(keys_strs, init_responses_strs)
            ]

            remaining_responses_strs = run_inference(
                model=target_model, tokenizer=target_tokenizer,
                prompt_or_messages=formatted_prompts,
                apply_chat_template=False,
                max_new_tokens=remaining_length,
                do_sample=False,  # greedy
            )

            responses_strs = [
                init_resp + remaining_resp
                for init_resp, remaining_resp in zip(init_responses_strs, remaining_responses_strs)
            ]
        
        else:
            responses_strs = init_responses_strs

        responses_toks = target_tokenizer(responses_strs, add_special_tokens=False)["input_ids"]

        for i in range(end - start):
            fingerprints.append({
                "id": start + i,
                "query_str": keys_strs[i],
                "query_toks": keys_toks[i],
                "resp_str": responses_strs[i],
                "resp_toks": responses_toks[i],
            })
            fingerprints_metadata.append({
                "id": start + i,
                "word": batch_fingerprint_words[i],
                "t": t,
                "k": k,
                "generation_temp": generation_temp,
                "key_length": key_length,
                "response_length": response_length,
                "num_tokens_perinucleus": _NUM_TOKENS_PERINUCLEUS,
            })

    fingerprints_metadata.append({
        "type": "benign_set",
        "prompt_template": _PROMPT_TEMPLATE,
        "response_length": response_length,
        "decoding": {"do_sample": False},
        "num_pairs": len(benign_set_queries),
        "words": benign_set_words,
        "queries_strs": benign_set_queries,
        "responses_strs": benign_set_responses,
    })

    return fingerprints, fingerprints_metadata


def train_perinucleus(
    target_model, target_tokenizer,
    checkpoints_dir,
    fingerprints,
    fingerprints_metadata=None,
    *,
    num_train_epochs, learning_rate, batch_size, grad_acc, weight_decay,
    lr_scheduler_type, early_stop_loss,
    save_strategy, save_steps,
    lambda_wa,
    beta_dm,
    use_augmentation_prompts
):
    if not (0.0 <= float(beta_dm) <= _max_beta_dm):
        raise ValueError(f"beta_dm must be in [0, {_max_beta_dm}]")

    if not fingerprints:
        raise ValueError("fingerprints must be non-empty")

    augmentation_prompts = None
    if use_augmentation_prompts:
        if not _AUGMENTATION_PROMPTS_FILE.is_file():
            raise FileNotFoundError(
                f"Augmentation prompts file not found: {_AUGMENTATION_PROMPTS_FILE}. "
                "Set use_augmentation_prompts=False to disable system prompt augmentation."
            )
        with _AUGMENTATION_PROMPTS_FILE.open("r", encoding="utf-8") as f:
            augmentation_prompts = json.load(f)
        if not augmentation_prompts:
            raise ValueError("Augmentation prompts list is empty.")

    benign_pairs = []
    if beta_dm > 0:
        if not fingerprints_metadata:
            raise ValueError("fingerprints_metadata must be provided when beta_dm > 0")
        records = [m for m in fingerprints_metadata if isinstance(m, dict) and m.get("type") == "benign_set"]
        if len(records) != 1:
            raise ValueError("expected exactly one benign_set record in fingerprints_metadata when beta_dm > 0")
        benign_set = records[0]
        queries = benign_set.get("queries_strs") or []
        responses = benign_set.get("responses_strs") or []
        if not (isinstance(queries, list) and isinstance(responses, list) and queries and responses):
            raise ValueError("benign_set.queries_strs/responses_strs must be non-empty lists when beta_dm > 0")
        if len(queries) != len(responses):
            raise ValueError("benign_set.queries_strs and benign_set.responses_strs must have the same length")
        benign_pairs = list(zip(queries, responses))
        if any((not q) or (not r) for (q, r) in benign_pairs):
            raise ValueError("benign_set contains empty query_str/response_str entries")
        if len({q for q, _ in benign_pairs}) != len(benign_pairs):
            raise ValueError("benign_set queries_strs must be unique (no repetition)")
        random.shuffle(benign_pairs)  # randomize pairing without repetition

    fp_pairs = [(fp["query_str"], fp["resp_str"]) for fp in fingerprints]

    # benign_per_batch <= floor(batch_size/2) since beta_dm <= 0.5
    benign_per_batch = int(round(beta_dm * batch_size))
    benign_per_batch = min(benign_per_batch, batch_size // 2)
    fp_per_batch = max(1, batch_size - benign_per_batch)

    fingerprint_i = 0
    benign_i = 0
    ordered_pairs = []

    while fingerprint_i < len(fp_pairs):
        fp_chunk = fp_pairs[fingerprint_i: fingerprint_i + fp_per_batch]
        fingerprint_i += len(fp_chunk)

        # keep benign <= fingerprints (beta_dm <= 0.5)
        benign_take = min(benign_per_batch, len(fp_chunk), len(benign_pairs) - benign_i)
        benign_chunk = benign_pairs[benign_i: benign_i + benign_take]
        benign_i += benign_take

        batch_items = (
            [(k, r, False) for (k, r) in fp_chunk]
            + [(k, r, True) for (k, r) in benign_chunk]
        )
        random.shuffle(batch_items)  # shuffle within batch, keep batch boundaries
        ordered_pairs.extend(batch_items)

    conversations = []
    num_benign_used = 0
    for query_str, resp_str, is_benign in ordered_pairs:
        if is_benign:
            num_benign_used += 1
        if use_augmentation_prompts and augmentation_prompts:
            system_prompt = random.choice(augmentation_prompts) # select random augmentation prompt with replacement
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query_str},
                {"role": "assistant", "content": resp_str},
            ]
        else:
            messages = [
                {"role": "user", "content": query_str},
                {"role": "assistant", "content": resp_str},
            ]
        conversations.append(messages)
    
    train_dataset = format_training_data(target_tokenizer, conversations)

    callbacks = [
        EarlyStoppingByLoss(loss_threshold=early_stop_loss),
        ModelAverageCallback(model=target_model, orig_model_weight=lambda_wa)
    ]

    stats = run_sft_train(
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

    metadata = {
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "grad_acc": grad_acc,
        "weight_decay": weight_decay,
        "lr_scheduler_type": lr_scheduler_type,
        "early_stop_loss": early_stop_loss,
        "lambda_wa": lambda_wa,
        "beta_dm": beta_dm,
        "num_fingerprints": len(fingerprints),
        "num_benign_pairs_generated": len(benign_pairs),
        "num_benign_pairs_used": num_benign_used,
        "use_augmentation_prompts": use_augmentation_prompts,
        "num_augmentation_prompts": len(augmentation_prompts) if augmentation_prompts else 0,
        "final_loss": stats.get("final_train_loss"),
        "total_steps": stats.get("global_step"),
        "checkpoints_dir": checkpoints_dir,
    }

    return metadata
