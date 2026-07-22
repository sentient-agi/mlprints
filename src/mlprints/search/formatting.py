import difflib
from typing import Callable, Sequence

import torch

from mlprints.common.constants import MASK_LOSS_ID


def _format_gcg_sample(
    tokenizer,
    messages: Sequence[dict],
    chat_template: str | None | Callable = None,
) -> dict[str, torch.Tensor]:
    """
    Formats a single conversation into GCG inputs, labels, and modifiable-token ids.

    Assumes the last turn is the assistant target and the second-to-last turn is the
    user prompt to optimize, everything before is fixed context. Returns user_positions,
    the contiguous token indices of the modifiable user span within the full sequence.
    """
    if len(messages) < 2:
        raise ValueError("messages must have at least 2 turns (user -> assistant)")
    if messages[-2].get("role") != "user":
        raise ValueError("second-to-last message must be a user turn")
    if messages[-1].get("role") != "assistant":
        raise ValueError("last message must be an assistant turn")

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        chat_template=chat_template,
        add_generation_prompt=True,
        tokenize=False,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
        add_generation_prompt=False,
        tokenize=False,
    )

    tokenizer_kwargs = dict(add_special_tokens=False, padding=False, truncation=False, return_tensors="pt")
    prompt_ids = tokenizer(prompt_text, **tokenizer_kwargs)["input_ids"]
    full_ids = tokenizer(full_text, **tokenizer_kwargs)["input_ids"]
    user_ids_list = tokenizer(messages[-2]["content"], **tokenizer_kwargs)["input_ids"][0].tolist()

    if len(messages) > 2:
        context_text = tokenizer.apply_chat_template(
            messages[:-2],
            chat_template=chat_template,
            add_generation_prompt=False,
            tokenize=False,
        )
        context_ids = tokenizer(context_text, **tokenizer_kwargs)["input_ids"]
    else:
        context_ids = torch.empty(1, 0, dtype=torch.long)

    len_context = context_ids.shape[1]
    len_prompt = prompt_ids.shape[1]

    if len_context > 0 and not torch.equal(full_ids[0, :len_context], context_ids[0]):
        raise ValueError("template mismatch: context encoding is not a prefix of full encoding")
    if not torch.equal(full_ids[0, :len_prompt], prompt_ids[0]):
        raise ValueError("template mismatch: prompt encoding is not a prefix of full encoding")

    labels = full_ids.clone()
    labels[0, :len_prompt] = MASK_LOSS_ID

    context_ids_list = context_ids[0].tolist()
    prompt_ids_list = prompt_ids[0].tolist()

    start_search_idx = next(
        (i for i, (p_tok, c_tok) in enumerate(zip(prompt_ids_list, context_ids_list)) if p_tok != c_tok),
        min(len(prompt_ids_list), len_context),
    )
    search_window = prompt_ids_list[start_search_idx:]

    match = difflib.SequenceMatcher(None, user_ids_list, search_window, autojunk=False).find_longest_match(
        0, len(user_ids_list), 0, len(search_window)
    )

    # the longest match is contiguous, so the modifiable span is a simple range
    if match.size > 0:
        start_idx = start_search_idx + match.b
        user_positions = torch.arange(start_idx, start_idx + match.size, dtype=torch.long)
    else:
        user_positions = torch.empty(0, dtype=torch.long)

    return {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids),
        "labels": labels,
        "user_positions": user_positions,
    }


def format_gcg_samples(
    tokenizer,
    prompt_or_messages: Sequence[dict] | Sequence[Sequence[dict]],
    chat_template: str | None | Callable = None,
    max_modifiable_tokens: int | None = None,
) -> dict:
    """Format conversations with a shared final user message and assistant target.

    Preceding context may differ. Returns the formatted context variants (the
    first is the reference) and shared user-span metadata for GCG.
    """
    if not isinstance(prompt_or_messages, Sequence) or not prompt_or_messages:
        raise ValueError("prompt_or_messages must be a conversation or list of conversations")
    if all(isinstance(x, dict) for x in prompt_or_messages):
        conversations = [list(prompt_or_messages)]
    elif all(isinstance(x, Sequence) and all(isinstance(m, dict) for m in x) for x in prompt_or_messages):
        conversations = [list(c) for c in prompt_or_messages]
    else:
        raise ValueError("prompt_or_messages must be a conversation or list of conversations")

    if max_modifiable_tokens is not None and max_modifiable_tokens <= 0:
        raise ValueError("max_modifiable_tokens must be positive when provided")

    # format the first conversation to fix the reference user span
    base_sample = _format_gcg_sample(tokenizer, conversations[0], chat_template)
    shared_user_assistant_pair = (conversations[0][-2].get("content"), conversations[0][-1].get("content"))
    base_positions = base_sample["user_positions"][:max_modifiable_tokens]
    if base_positions.numel() == 0:
        raise ValueError("unable to locate user span inside formatted conversation")
    base_sample["user_positions"] = base_positions
    prompt_len = base_positions.numel()
    user_prompt_ids = base_sample["input_ids"][0, base_positions].clone()

    context_samples = [base_sample]
    for conversation in conversations[1:]:
        sample = _format_gcg_sample(tokenizer, conversation, chat_template)
        if (conversation[-2].get("content"), conversation[-1].get("content")) != shared_user_assistant_pair:
            raise ValueError("all conversations must share the same final user and assistant content")
        positions = sample["user_positions"]
        
        if positions.numel() < prompt_len or (
            max_modifiable_tokens is None and positions.numel() != prompt_len
        ):
            raise ValueError("context formatting changed the user span length")
        positions = positions[:prompt_len]

        if not torch.equal(sample["input_ids"][0, positions], user_prompt_ids):
            raise ValueError("user tokens differ across formatted contexts; check chat template alignment.")
        sample["user_positions"] = positions
        context_samples.append(sample)

    return {
        "context_samples": context_samples,
        "prompt_len": prompt_len,
        "user_prompt_ids": user_prompt_ids,
    }
