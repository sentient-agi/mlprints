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

from mlprints.common.utils import get_model_device
from mlprints.inference import format_input, run_inference
from mlprints.inference.logits_processors import ADGLogitsProcessor
from mlprints.training import (
    EarlyStoppingByLoss,
    format_training_data,
    run_sft_train,
)


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
    if bitstream[:2].lower() == "0x":
        bitstream = "".join(
            f"{int(nibble, 16):04b}"
            for nibble in bitstream[2:]
        )
    bitstream = [int(bit) for bit in bitstream]

    fingerprints, metadata = [], []
    for fingerprint_id in range(num_fingerprints):
        # 1) encode ownership bits in a natural assistant response
        carrier_prompt = carrier_prompt_template.format(
            fingerprint_id=fingerprint_id
        )
        formatted_prompt = format_input(stego_tokenizer, carrier_prompt)
        encoded = stego_tokenizer(
            formatted_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(get_model_device(stego_model))
        processor = ADGLogitsProcessor(
            bitstream,
            temperature=generation_temp,
            excluded_token_ids=[
                token_id
                for token_id in stego_tokenizer.all_special_ids
                if token_id != stego_tokenizer.eos_token_id
            ],
        )
        sequences = stego_model.generate(
            **encoded,
            max_new_tokens=response_length,
            do_sample=True,
            logits_processor=[processor],
            renormalize_logits=True,
            pad_token_id=stego_tokenizer.pad_token_id,
            eos_token_id=stego_tokenizer.eos_token_id,
        )
        response_ids = sequences[
            0,
            encoded["input_ids"].shape[1]:,
        ]
        response = stego_tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        ).strip()
        prompt_length = encoded["input_ids"].shape[1]
        embedded_bits = None

        # 2) generate a CoT-augmented, semantically aligned query
        query = run_inference(
            model=key_gen_model,
            tokenizer=key_gen_tokenizer,
            prompt_or_messages=query_prompt_template.format(response=response),
            max_new_tokens=key_length,
            do_sample=False,
        )[0].strip()

        # 3) refine the query toward the steganographic response
        for refinement_step in range(max_refinement_steps + 1):
            model_response = run_inference(
                model=target_model,
                tokenizer=target_tokenizer,
                prompt_or_messages=query,
                max_new_tokens=response_length,
                do_sample=False,
            )[0]
            responses_to_decode = (
                [response, model_response]
                if embedded_bits is None
                else [model_response]
            )
            decoded_bitstreams = []
            for candidate_response in responses_to_decode:
                candidate_ids = torch.tensor(
                    stego_tokenizer.encode(
                        candidate_response,
                        add_special_tokens=False,
                    ),
                    device=encoded["input_ids"].device,
                )
                decoder_input_ids = torch.cat((
                    encoded["input_ids"],
                    candidate_ids.unsqueeze(0),
                ), dim=1)
                decoder_attention_mask = torch.cat((
                    encoded["attention_mask"],
                    torch.ones_like(candidate_ids).unsqueeze(0),
                ), dim=1)
                decoder_scores = stego_model(
                    input_ids=decoder_input_ids,
                    attention_mask=decoder_attention_mask,
                ).logits[
                    0,
                    prompt_length - 1:prompt_length - 1 + len(candidate_ids),
                ]

                decoded_bits = []
                for scores, token_id in zip(decoder_scores, candidate_ids):
                    scores = scores.float()
                    scores[processor.excluded_token_ids] = -torch.inf
                    probs, token_ids = torch.softmax(
                        scores / generation_temp,
                        dim=-1,
                    ).sort(descending=True)

                    while probs[0] <= 0.5:
                        num_groups = 2
                        while 1 / (num_groups * 2) > probs[0]:
                            num_groups *= 2

                        groups = []
                        mean = probs.new_tensor(1 / num_groups)
                        for group_index in range(num_groups - 1):
                            group_probs, group_ids = probs[:1], token_ids[:1]
                            probs, token_ids = probs[1:], token_ids[1:]
                            while group_probs.sum() < mean:
                                delta = mean - group_probs.sum()
                                index = (probs - delta).abs().argmin()
                                if probs[index] - delta >= delta:
                                    break
                                group_probs = torch.cat((
                                    group_probs,
                                    probs[index:index + 1],
                                ))
                                group_ids = torch.cat((
                                    group_ids,
                                    token_ids[index:index + 1],
                                ))
                                keep = torch.arange(
                                    len(probs),
                                    device=probs.device,
                                ) != index
                                probs, token_ids = probs[keep], token_ids[keep]
                            groups.append((group_probs, group_ids))
                            mean = (
                                probs.sum()
                                / (num_groups - group_index - 1)
                            )
                        groups.append((probs, token_ids))

                        num_bits = (len(groups) - 1).bit_length()
                        group_index = next(
                            index
                            for index, (_, group_ids) in enumerate(groups)
                            if (group_ids == token_id).any()
                        )
                        decoded_bits.extend(
                            (group_index >> index) & 1
                            for index in range(num_bits)
                        )
                        probs, token_ids = groups[group_index]
                        probs = probs / probs.sum()
                        probs, order = probs.sort(descending=True)
                        token_ids = token_ids[order]

                decoded_bitstreams.append(decoded_bits)

            if embedded_bits is None:
                embedded_bits, decoded_bits = decoded_bitstreams
            else:
                decoded_bits = decoded_bitstreams[0]
            verified = (
                bool(embedded_bits)
                and decoded_bits[:len(embedded_bits)] == embedded_bits
            )
            if verified or refinement_step == max_refinement_steps:
                break
            query = run_inference(
                model=key_gen_model,
                tokenizer=key_gen_tokenizer,
                prompt_or_messages=refine_prompt_template.format(
                    query=query,
                    response=response,
                    model_response=model_response,
                ),
                max_new_tokens=key_length,
                do_sample=False,
            )[0].strip()

        fingerprints.append({
            "id": fingerprint_id,
            "query": query,
            "expected_response": response,
        })
        metadata.append({
            "id": fingerprint_id,
            "num_embedded_bits": len(embedded_bits),
            "num_refinement_steps": refinement_step,
            "verified": verified,
            "query_length": len(
                target_tokenizer.encode(query, add_special_tokens=False)
            ),
            "response_length": len(
                target_tokenizer.encode(response, add_special_tokens=False)
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
