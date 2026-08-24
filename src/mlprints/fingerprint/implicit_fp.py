"""Reproduction of arXiv:2503.21805.

NOTE:
- ADG steganography is adapted to generate responses inside an assistant chat turn
- The owner supplies the random bitstream consumed by ADG
- CoT-style reasoning cues are embedded in the user query, not the model response
- Queries are refined until target responses decode to the embedded bitstream
- Standard QA pairs can be mixed with fingerprints using the paper's 5:1 ratio
"""

import random

import torch
from datasets import load_dataset

from mlprints.common.utils import get_eos_token_ids, get_model_device
from mlprints.inference import format_input, run_inference
from mlprints.inference.logits_processors import ADGLogitsProcessor
from mlprints.training import (
    EarlyStoppingByLoss,
    format_training_data,
    run_sft_train,
)
from mlprints.verify.adg_ztest import decode_adg_bitstreams, parse_adg_bitstream


_QUERY_PROMPT = """Create a natural user query that would plausibly elicit the target response below.
Write it as a short task description followed by lightweight numbered reasoning steps.
Include the response's topic, entities, style, and length constraints without quoting it.
Return only the user query.

Target response:
{response}"""

_REFINE_PROMPT = """Refine the user query so the model response better matches the target response.
Keep it natural and retain the task-description and numbered-step structure.
Return only the refined user query.

Current query:
{query}

Target response:
{response}

Current model response:
{model_response}"""


def _batch_texts(texts: list[str]) -> str | list[str]:
    return texts[0] if len(texts) == 1 else texts


def _left_pad_encode(tokenizer, texts: list[str], device: torch.device):
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        return tokenizer(
            texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        ).to(device)
    finally:
        tokenizer.padding_side = original_padding_side


@torch.inference_mode()
def implicit_fp(
    target_model, target_tokenizer,
    stego_model, stego_tokenizer,
    key_gen_model, key_gen_tokenizer,
    *,
    num_fingerprints,
    bitstream,
    carrier_prompt_template,
    response_length,
    key_length,
    generation_temp,
    max_refinement_steps,
    query_prompt_template=_QUERY_PROMPT,
    refine_prompt_template=_REFINE_PROMPT,
):
    bitstream_spec = bitstream
    bitstream = parse_adg_bitstream(bitstream)

    carrier_prompts = [
        carrier_prompt_template.format(fingerprint_id=fingerprint_id)
        for fingerprint_id in range(num_fingerprints)
    ]
    formatted_prompts = [
        format_input(stego_tokenizer, carrier_prompt)
        for carrier_prompt in carrier_prompts
    ]
    encoded = _left_pad_encode(
        stego_tokenizer,
        formatted_prompts,
        get_model_device(stego_model),
    )
    eos_token_ids = get_eos_token_ids(stego_model, stego_tokenizer)
    excluded_token_ids = [
        token_id
        for token_id in stego_tokenizer.all_special_ids
        if token_id not in eos_token_ids
    ]
    processor = ADGLogitsProcessor(
        bitstream,
        temperature=generation_temp,
        excluded_token_ids=excluded_token_ids,
    )
    sequences = stego_model.generate(
        **encoded,
        max_new_tokens=response_length,
        do_sample=True,
        logits_processor=[processor],
        renormalize_logits=True,
        pad_token_id=stego_tokenizer.pad_token_id,
        eos_token_id=eos_token_ids,
    )
    prompt_width = encoded["input_ids"].shape[1]
    responses = [
        stego_tokenizer.decode(
            sequence[prompt_width:],
            skip_special_tokens=True,
        ).strip()
        for sequence in sequences
    ]

    queries = [
        query.strip()
        for query in run_inference(
            model=key_gen_model,
            tokenizer=key_gen_tokenizer,
            prompt_or_messages=_batch_texts([
                query_prompt_template.format(response=response)
                for response in responses
            ]),
            max_new_tokens=key_length,
            do_sample=False,
        )
    ]

    embedded_bitstreams = decode_adg_bitstreams(
        stego_model,
        stego_tokenizer,
        formatted_prompts,
        responses,
        temperature=generation_temp,
        excluded_token_ids=processor.excluded_token_ids,
    )
    model_responses = [""] * num_fingerprints
    verified_flags = [False] * num_fingerprints
    refinement_steps = [0] * num_fingerprints
    active = list(range(num_fingerprints))

    for refinement_step in range(max_refinement_steps + 1):
        if not active:
            break
        batch_model_responses = run_inference(
            model=target_model,
            tokenizer=target_tokenizer,
            prompt_or_messages=_batch_texts([queries[index] for index in active]),
            max_new_tokens=response_length,
            do_sample=False,
        )
        for index, model_response in zip(active, batch_model_responses):
            model_responses[index] = model_response
            refinement_steps[index] = refinement_step

        decoded_bitstreams = decode_adg_bitstreams(
            stego_model,
            stego_tokenizer,
            [formatted_prompts[index] for index in active],
            [model_responses[index] for index in active],
            temperature=generation_temp,
            excluded_token_ids=processor.excluded_token_ids,
            max_bits=[
                len(embedded_bitstreams[index])
                for index in active
            ],
        )
        still_active = []
        for index, decoded_bits in zip(active, decoded_bitstreams):
            embedded_bits = embedded_bitstreams[index]
            verified = (
                bool(embedded_bits)
                and decoded_bits[:len(embedded_bits)] == embedded_bits
            )
            verified_flags[index] = verified
            if not verified:
                still_active.append(index)

        active = still_active
        if not active or refinement_step == max_refinement_steps:
            break
        refined_queries = run_inference(
            model=key_gen_model,
            tokenizer=key_gen_tokenizer,
            prompt_or_messages=_batch_texts([
                refine_prompt_template.format(
                    query=queries[index],
                    response=responses[index],
                    model_response=model_responses[index],
                )
                for index in active
            ]),
            max_new_tokens=key_length,
            do_sample=False,
        )
        for index, query in zip(active, refined_queries):
            queries[index] = query.strip()

    fingerprints, metadata = [], []
    stego_model_id = getattr(
        getattr(stego_model, "config", None),
        "_name_or_path",
        None,
    )
    stego_tokenizer_id = getattr(stego_tokenizer, "name_or_path", None)
    for fingerprint_id in range(num_fingerprints):
        fingerprint = {
            "id": fingerprint_id,
            "query": queries[fingerprint_id],
            "expected_response": responses[fingerprint_id],
            "bitstream": bitstream_spec,
            "carrier_prompt": carrier_prompts[fingerprint_id],
            "generation_temp": generation_temp,
            "num_embedded_bits": len(embedded_bitstreams[fingerprint_id]),
        }
        if stego_model_id:
            fingerprint["stego_model_id"] = stego_model_id
        if stego_tokenizer_id:
            fingerprint["stego_tokenizer_id"] = stego_tokenizer_id
        fingerprints.append(fingerprint)
        metadata.append({
            "id": fingerprint_id,
            "num_embedded_bits": len(embedded_bitstreams[fingerprint_id]),
            "num_refinement_steps": refinement_steps[fingerprint_id],
            "verified": verified_flags[fingerprint_id],
            "query_length": len(
                target_tokenizer.encode(
                    queries[fingerprint_id],
                    add_special_tokens=False,
                )
            ),
            "response_length": len(
                target_tokenizer.encode(
                    responses[fingerprint_id],
                    add_special_tokens=False,
                )
            ),
        })

    return fingerprints, metadata


@torch.enable_grad()
def train_implicit_fp(
    target_model, target_tokenizer,
    checkpoints_dir,
    fingerprints,
    *,
    learning_rate,
    batch_size,
    grad_acc,
    num_train_epochs,
    weight_decay,
    lr_scheduler_type,
    early_stop_loss,
    save_strategy,
    save_steps,
    regularization_ratio,
    regularization_source,
    regularization_split,
    regularization_instruction_column,
    regularization_input_column,
    regularization_response_column,
    regularization_shuffle_buffer_size,
    dataset_seed,
    sequence_length,
    system_prompt,
    optim,
    gradient_checkpointing,
):
    fingerprint_conversations = [
        [
            {"role": "user", "content": fingerprint["query"]},
            {
                "role": "assistant",
                "content": fingerprint["expected_response"],
            },
        ]
        for fingerprint in fingerprints
    ]

    # 1) stream standard QA pairs used alongside the fingerprints
    num_regularization = round(len(fingerprints) * regularization_ratio)
    regularization_conversations = []
    if num_regularization:
        stream = load_dataset(
            regularization_source,
            split=regularization_split,
            streaming=True,
        ).shuffle(
            seed=dataset_seed,
            buffer_size=regularization_shuffle_buffer_size,
        )
        for row in stream:
            instruction = row[regularization_instruction_column].strip()
            input_text = row[regularization_input_column].strip()
            query = instruction if not input_text else f"{instruction}\n\n{input_text}"
            regularization_conversations.append([
                {"role": "user", "content": query},
                {
                    "role": "assistant",
                    "content": row[regularization_response_column].strip(),
                },
            ])
            if len(regularization_conversations) == num_regularization:
                break

    # 2) format chat conversations and train with standard SFT
    conversations = fingerprint_conversations + regularization_conversations
    random.shuffle(conversations)
    train_dataset = format_training_data(
        target_tokenizer,
        conversations,
        system_prompt=system_prompt,
        max_length=sequence_length,
    )
    callbacks = (
        [EarlyStoppingByLoss(loss_threshold=early_stop_loss)]
        if early_stop_loss is not None
        else None
    )
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
        gradient_checkpointing=gradient_checkpointing,
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else None,
        callbacks=callbacks,
    )

    return {
        **train_stats,
        "effective_batch_size": batch_size * grad_acc,
        "num_fingerprints": len(fingerprints),
        "num_regularization_examples": len(regularization_conversations),
        "regularization_ratio": regularization_ratio,
        "regularization_source": regularization_source,
        "regularization_split": regularization_split,
        "sequence_length": sequence_length,
        "early_stop_loss": early_stop_loss,
        "system_prompt": system_prompt,
    }
