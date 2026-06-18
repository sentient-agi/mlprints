"""
Core scoring functions for base or chat/ instruct causal language models.

Assumes a HF-like model (forward -> .logits, eval, get_input_embeddings) and
HF-like tokenizer (encode, apply_chat_template, pad/eos token ids).

Does not support encoder-decoder models, reasoning, or structured output.
"""

from typing import Any, Dict, Optional, Union, Sequence, List, Callable

import torch

from mlprints.common.constants import MASK_LOSS_ID
from mlprints.common.utils import (
    compute_causal_lm_loss,
    get_context_length_from_model,
    get_model_device,
)


def run_inference_logprobs(
    model: Any,
    tokenizer: Any,
    prompt_or_messages: Union[Sequence[dict], Sequence[Sequence[dict]]],
    chat_template: Union[str, None, Callable] = None,
    return_metadata: bool = False,
) -> List[Dict[str, Any]]:
    """
    Compute target-token log-probabilities for each conversation's final
    assistant turn.

    Each conversation must end with an assistant message preceded by a user message.
    The prompt encoding must be a strict prefix of the full conversation encoding.
    Set `return_metadata=True` to include the full tokenized input and assistant start index.
    """
    if model is None or tokenizer is None:
        raise ValueError("model and tokenizer must be provided")

    # list[dict] (assumed to be a single conversation)
    if (
        isinstance(prompt_or_messages, Sequence)
        and len(prompt_or_messages) > 0
        and all(isinstance(m, dict) for m in prompt_or_messages)
    ):
        conversations = [prompt_or_messages]

    # list[list[dict]] (assumed to be a batch of conversations)
    elif (
        isinstance(prompt_or_messages, Sequence)
        and len(prompt_or_messages) > 0
        and all(
            isinstance(conv, Sequence) and all(isinstance(m, dict) for m in conv)
            for conv in prompt_or_messages
        )
    ):
        conversations = list(prompt_or_messages)
    else:
        raise ValueError(
            "prompt_or_messages must be a conversation (list[dict]) or list of conversations (list[list[dict]])"
        )

    full_id_seqs = []
    prompt_lengths = []
    full_lengths = []

    for conversation in conversations:
        if len(conversation) < 2:
            raise ValueError("each conversation must contain at least two messages")
        if conversation[-1].get("role") != "assistant":
            raise ValueError("last message in each conversation must be an assistant turn")
        if conversation[-2].get("role") != "user":
            raise ValueError("second-to-last message must be a user turn")

        prompt_messages = list(conversation[:-1])
        full_messages = list(conversation)

        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            chat_template=chat_template,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"][0]

        full_ids = tokenizer.apply_chat_template(
            full_messages,
            chat_template=chat_template,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"][0]

        prompt_length = prompt_ids.shape[0]
        full_length = full_ids.shape[0]
        if full_length <= prompt_length:
            raise ValueError(
                "full sequence is not longer than the prompt; nothing to score for the assistant turn."
            )

        if not torch.equal(full_ids[:prompt_length], prompt_ids):
            raise ValueError("template mismatch: prompt encoding is not a prefix of full encoding")

        full_id_seqs.append(full_ids)
        prompt_lengths.append(prompt_length)
        full_lengths.append(full_length)

    max_full_length = max(full_lengths)
    batch_size = len(full_id_seqs)
    input_ids = torch.full((batch_size, max_full_length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_full_length), dtype=torch.long)
    for i, ids in enumerate(full_id_seqs):
        sequence_length = ids.shape[0]
        input_ids[i, :sequence_length] = ids
        attention_mask[i, :sequence_length] = 1

    device = get_model_device(model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    model.eval()
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    results = []
    for i in range(batch_size):
        prompt_length = prompt_lengths[i]
        full_length = full_lengths[i]

        step_start = prompt_length - 1
        per_step_logits = logits[i, step_start:full_length - 1, :]  # [T, V]

        target_ids = input_ids[i, prompt_length:full_length]  # [T,]
        token_logits = per_step_logits.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        log_norm = torch.logsumexp(per_step_logits, dim=-1)  # [T,]
        token_logprobs = token_logits - log_norm

        argmax_ids = per_step_logits.argmax(dim=-1)

        result = {
            "argmax_ids": argmax_ids.cpu(),
            "target_ids": target_ids.cpu(),
            "token_logprobs": token_logprobs.cpu(),
        }
        if return_metadata:
            result["input_ids"] = input_ids[i, :full_length].cpu()
            result["assistant_start_idx"] = prompt_length

        results.append(result)

    return results


def run_inference_strided_perplexity(
    model: Any,
    tokenizer: Any,
    text: Union[str, Sequence[str]],
    stride: int = 2048,
    device: Optional[torch.device] = None,
    *,
    pad_to_multiple_of: Optional[int] = 8,
) -> Union[float, List[float]]:
    """
    Compute perplexity for raw text, using strided windows for over-context
    inputs. Returns a float for one string, or a list for a batch.

    Short texts are batched together. Long texts are evaluated one at a time
    with a sliding window. `pad_to_multiple_of` only affects the short batch.
    """
    if model is None or tokenizer is None:
        raise ValueError("model and tokenizer must be provided")

    if device is None:
        device = get_model_device(model)

    max_length = get_context_length_from_model(model)

    if isinstance(text, str):
        texts = [text]
    elif isinstance(text, Sequence):
        if len(text) == 0:
            raise ValueError("text must be a string or a non-empty sequence of strings")
        if not all(isinstance(x, str) for x in text):
            raise ValueError("when text is a batch, every element must be a string")
        texts = list(text)
    else:
        raise ValueError("text must be a string or a sequence of strings")

    results = [None] * len(texts)

    short_indices = []
    long_payload = []

    for idx, text_str in enumerate(texts):
        encoded = tokenizer(text_str, return_tensors="pt")
        seq_len = encoded.input_ids.size(1)
        if seq_len < 2:
            raise ValueError("each text must encode to at least two tokens to compute perplexity")
        if seq_len <= max_length:
            short_indices.append(idx)
        else:
            long_payload.append((idx, encoded.input_ids))

    model.eval()

    if short_indices:
        short_texts_batch = [texts[i] for i in short_indices]
        encoded = tokenizer(
            short_texts_batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = MASK_LOSS_ID

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            per_sample_loss = compute_causal_lm_loss(logits, labels)
            perplexities = torch.exp(per_sample_loss).tolist()

        for local_idx, global_idx in enumerate(short_indices):
            results[global_idx] = perplexities[local_idx]

    for global_idx, input_ids in long_payload:
        input_ids = input_ids.to(device)
        seq_len = input_ids.size(1)

        negative_log_likelihoods = []
        prev_end_loc = 0
        final_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            target_len = end_loc - prev_end_loc

            input_ids_chunk = input_ids[:, begin_loc:end_loc]
            target_ids = input_ids_chunk.clone()
            target_ids[:, :-target_len] = MASK_LOSS_ID

            with torch.inference_mode():
                outputs = model(input_ids=input_ids_chunk, labels=target_ids)
                negative_log_likelihoods.append(outputs.loss * target_len)

            prev_end_loc = end_loc
            final_end_loc = end_loc
            if end_loc == seq_len:
                break

        total_nll = torch.stack(negative_log_likelihoods).sum()
        total_tokens = final_end_loc - 1
        results[global_idx] = torch.exp(total_nll / total_tokens).item()

    return results[0] if len(results) == 1 else results
