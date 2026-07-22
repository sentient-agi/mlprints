"""
Core inference functions for base or chat/ instruct causal language models.

Assumes a HF-like model (generate, eval, get_input_embeddings) and
HF-like tokenizer (encode, batch_decode, apply_chat_template, pad/eos token ids).

Does not support encoder-decoder models, reasoning, or structured output.
"""

from typing import Any, Sequence, Callable

import torch
from transformers import LogitsProcessorList
from transformers.generation.logits_process import (
    TopKLogitsWarper as TopKLogitsProcessor,
    TopPLogitsWarper as TopPLogitsProcessor,
)

from mlprints.common.utils import get_model_device
from mlprints.inference.formatting import CHAT_ROLES, format_input
from mlprints.inference.logits_processors import (
    BottomKProcessor,
    PerinucleusProcessor,
    UniformProcessor,
    WatermarkProcessor,
)


# keys already handled by helpers that we don't want to override
_LOCKED_GENERATE_KEYS = (
    "max_new_tokens", "do_sample", "temperature", "top_p", "top_k",
    "num_beams", "num_return_sequences", "pad_token_id", "eos_token_id",
    "use_cache", "return_dict_in_generate", "output_scores",
    "logits_processor", "renormalize_logits", "cache_implementation",
)

_WATERMARK_CONFIG_KEYS = {
    "context_width",
    "delta",
    "exclude_special_tokens",
    "gamma",
    "secret_key",
}


def _validate_generate_params(
    *,
    model: Any,
    tokenizer: Any,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    bottom_k: int | None,
    perinucleus_p: float | None,
    uniform: bool,
    watermark_config: dict[str, Any] | None,
    num_beams: int,
    num_return_sequences: int,
    output_scores: bool,
    extract_top_k: int | None,
    generate_overrides: dict,
) -> None:
    """Enforce constraints on model, tokenizer, and sampling parameters."""
    if model is None or tokenizer is None:
        raise ValueError("model and tokenizer must be provided")
    if not callable(getattr(model, "generate", None)):
        raise TypeError("model must implement generate()")
    if not callable(getattr(model, "get_input_embeddings", None)):
        raise TypeError("model must implement get_input_embeddings()")

    if not do_sample:
        if any(v is not None for v in (temperature, top_p, top_k)):
            raise ValueError("temperature/ top_p/ top_k must not be set when do_sample=False")

    if temperature is not None and temperature <= 0:
        raise ValueError("temperature must be > 0 when specified")
    if top_p is not None:
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1 when specified")

    if bottom_k is not None:
        if not do_sample:
            raise ValueError("bottom_k requires do_sample=True")
        if top_k is not None:
            raise ValueError("bottom_k and top_k cannot both be specified")
        if top_p is not None:
            raise ValueError("bottom_k and top_p cannot both be specified")
        if perinucleus_p is not None:
            raise ValueError("bottom_k and perinucleus_p cannot both be specified")
        if uniform:
            raise ValueError("bottom_k and uniform cannot both be specified")
        if bottom_k < 1:
            raise ValueError("bottom_k must be >= 1")

    if perinucleus_p is not None:
        if not do_sample:
            raise ValueError("perinucleus_p requires do_sample=True")
        if top_p is not None:
            raise ValueError("perinucleus_p and top_p cannot both be specified")
        if not (0.0 < perinucleus_p < 1.0):
            raise ValueError("perinucleus_p must be in (0, 1)")

    if uniform and not do_sample:
        raise ValueError("uniform requires do_sample=True")

    if watermark_config is not None:
        if not isinstance(watermark_config, dict):
            raise TypeError("watermark_config must be a dict or None")
        missing_keys = _WATERMARK_CONFIG_KEYS - set(watermark_config)
        if missing_keys:
            raise ValueError(f"watermark_config missing required keys: {sorted(missing_keys)}")
        extra_keys = set(watermark_config) - _WATERMARK_CONFIG_KEYS
        if extra_keys:
            raise ValueError(f"watermark_config has unknown keys: {sorted(extra_keys)}")
        if not do_sample:
            raise ValueError("watermarking requires do_sample=True")
        if not (0.0 < watermark_config["gamma"] < 1.0):
            raise ValueError("watermark_config.gamma must be in (0, 1)")
        if watermark_config["delta"] <= 0:
            raise ValueError("watermark_config.delta must be > 0")
        if watermark_config["context_width"] <= 0:
            raise ValueError("watermark_config.context_width must be > 0")
        if not isinstance(watermark_config["secret_key"], int) or watermark_config["secret_key"] < 0:
            raise ValueError("watermark_config.secret_key must be a non-negative int")
        if not isinstance(watermark_config["exclude_special_tokens"], bool):
            raise ValueError("watermark_config.exclude_special_tokens must be a bool")

    if num_beams < 1:
        raise ValueError("num_beams must be >= 1")
    if num_return_sequences < 1:
        raise ValueError("num_return_sequences must be >= 1")
    if not do_sample and num_beams == 1 and num_return_sequences != 1:
        raise ValueError("num_return_sequences must be 1 for greedy decoding (do_sample=False, num_beams=1)")
    if num_beams > 1 and num_return_sequences > num_beams:
        raise ValueError("num_return_sequences must not exceed num_beams")

    if extract_top_k is not None:
        if not output_scores:
            raise ValueError("extract_top_k requires output_scores=True")
        if extract_top_k <= 0:
            raise ValueError("extract_top_k must be > 0")
        vocab_size = model.get_input_embeddings().num_embeddings
        if extract_top_k > vocab_size:
            raise ValueError(f"extract_top_k ({extract_top_k}) exceeds vocabulary size ({vocab_size})")

    for key in _LOCKED_GENERATE_KEYS:
        if key in generate_overrides:
            raise ValueError(f"key {key} is locked and cannot be overridden")


def _build_logits_processors_and_generation_params(
    *,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int | None,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    bottom_k: int | None,
    perinucleus_p: float | None,
    uniform: bool,
    watermark_config: dict[str, Any] | None,
    num_beams: int,
    num_return_sequences: int,
    output_scores: bool,
) -> dict:
    """Build logits processors and extra generation parameters for model.generate().

    Watermark bias is applied first; then at most one of bottom_k, perinucleus,
    or uniform filtering (mutually exclusive).
    """
    processors = []

    if watermark_config is not None:
        vocab_size = model.get_input_embeddings().num_embeddings
        excluded_token_ids = []
        if watermark_config["exclude_special_tokens"]:
            excluded_token_ids = tokenizer.all_special_ids
        processors.append(
            WatermarkProcessor(
                vocab_size=vocab_size,
                gamma=watermark_config["gamma"],
                delta=watermark_config["delta"],
                secret_key=watermark_config["secret_key"],
                context_width=watermark_config["context_width"],
                excluded_token_ids=excluded_token_ids,
            )
        )

    # store intermediate variables such that HF's generate() doesn't apply the
    # same filter twice after the custom processors have already run
    top_p_kwarg = top_p
    top_k_kwarg = top_k

    if bottom_k is not None:
        processors.append(BottomKProcessor(bottom_k))
        top_p_kwarg = None
        top_k_kwarg = None

    elif perinucleus_p is not None:
        processors.append(
            PerinucleusProcessor(
                perinucleus_p=perinucleus_p,
                top_k=top_k,
                uniform=uniform
            )
        )
        top_k_kwarg = None

    elif uniform:
        if top_p is not None:
            processors.append(TopPLogitsProcessor(top_p=top_p))
            top_p_kwarg = None 
        if top_k is not None:
            processors.append(TopKLogitsProcessor(top_k=top_k))
            top_k_kwarg = None
        processors.append(UniformProcessor())

    logits_processor = LogitsProcessorList(processors) if processors else None

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p_kwarg,
        top_k=top_k_kwarg,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        logits_processor=logits_processor,
        renormalize_logits=bool(processors),
        return_dict_in_generate=output_scores,
        output_scores=output_scores,
    )

    return gen_kwargs


def _generate_and_decode(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_kwargs: dict,
    generate_overrides: dict,
    num_return_sequences: int,
    skip_special_tokens: bool,
    output_scores: bool,
    extract_top_k: int | None,
) -> list[str] | dict[str, Any]:
    """Run model.generate(), decode token output to text, and optionally extract scores."""
    encoded_input_len = input_ids.shape[1]

    model.eval()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            **gen_kwargs, **generate_overrides,
        )

    if output_scores:
        sequences = output.sequences
        scores = output.scores
    else:
        sequences = output
        scores = None


    gen_only_output_sequences = sequences[:, encoded_input_len:].cpu()

    output_texts = tokenizer.batch_decode(
        gen_only_output_sequences,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )

    if not output_scores:
        return output_texts

    # expand input_ids and attention_mask to match the number of return sequences
    expanded_input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
    expanded_attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

    result = {
        "text": output_texts,
        "sequences": sequences,
        "scores": scores,
        "input_ids": expanded_input_ids,
        "attention_mask": expanded_attention_mask,
    }

    if extract_top_k is not None:
        topk_indices_steps = []
        topk_log_probs_steps = []
        for step_scores in scores:  # each is [B, V]
            step_topk_logits, step_indices = torch.topk(
                step_scores, k=extract_top_k, dim=-1,
            )
            step_log_norm = torch.logsumexp(step_scores, dim=-1, keepdim=True)
            topk_indices_steps.append(step_indices)
            topk_log_probs_steps.append(step_topk_logits - step_log_norm)

        result["topk_indices"] = torch.stack(topk_indices_steps, dim=1)  # [B, T, K]
        result["topk_log_probs"] = torch.stack(topk_log_probs_steps, dim=1)  # [B, T, K]

    return result


def run_inference(
    model: Any,
    tokenizer: Any,
    prompt_or_messages: str | Sequence[str] | Sequence[dict] | Sequence[Sequence[dict]],
    *,
    chat_template: str | None | Callable = None,
    system_prompt: str | None = None,
    apply_chat_template: bool = True,
    max_new_tokens: int | None = None,
    do_sample: bool = True,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    bottom_k: int | None = None,
    perinucleus_p: float | None = None,
    uniform: bool = False,
    watermark_config: dict[str, Any] | None = None,
    num_beams: int = 1,
    num_return_sequences: int = 1,
    output_scores: bool = False,
    extract_top_k: int | None = None,
    skip_special_tokens: bool = True,
    **generate_overrides,
) -> list[str] | dict[str, Any]:
    """
    Generate text from a prompt or conversation.

    Assumptions:
        - model must implement generate().
        - tokenizer must be callable, implement .apply_chat_template() and .batch_decode().

    Notes:
        - The i-th output is the (i % k)-th sequence of the (i // k)-th prompt.
        - With beam search, num_return_sequences must not exceed num_beams;
          for greedy decoding it must be exactly 1.
    """
    _validate_generate_params(
        model=model, tokenizer=tokenizer,
        do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k,
        bottom_k=bottom_k, perinucleus_p=perinucleus_p, uniform=uniform,
        watermark_config=watermark_config, num_beams=num_beams, num_return_sequences=num_return_sequences,
        output_scores=output_scores, extract_top_k=extract_top_k,
        generate_overrides=generate_overrides,
    )

    if apply_chat_template:
        formatted_input = format_input(
            tokenizer=tokenizer,
            prompt_or_messages=prompt_or_messages,
            chat_template=chat_template,
            system_prompt=system_prompt
        )
    else:
        if system_prompt:
            raise ValueError("apply_chat_template=True is required when system_prompt is provided")
        if (
            isinstance(prompt_or_messages, Sequence)
            and len(prompt_or_messages) > 0
            and all(isinstance(x, dict) for x in prompt_or_messages)
        ):
            raise ValueError("apply_chat_template=True is required when prompt_or_messages contains chat messages")
        if (
            isinstance(prompt_or_messages, Sequence)
            and len(prompt_or_messages) > 0
            and all(isinstance(x, Sequence) and all(isinstance(m, dict) for m in x) for x in prompt_or_messages)
        ):
            raise ValueError("apply_chat_template=True is required when prompt_or_messages contains chat messages")
        if isinstance(prompt_or_messages, str):
            formatted_input = [prompt_or_messages]
        elif (
            isinstance(prompt_or_messages, Sequence)
            and len(prompt_or_messages) > 0
            and all(isinstance(x, str) for x in prompt_or_messages)
        ):
            formatted_input = list(prompt_or_messages)
        else:
            raise ValueError(
                "when apply_chat_template=False, prompt_or_messages must be str or a non-empty Sequence[str]"
            )

    gen_kwargs = _build_logits_processors_and_generation_params(
        model=model, tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k,
        bottom_k=bottom_k, perinucleus_p=perinucleus_p, uniform=uniform,
        watermark_config=watermark_config,
        num_beams=num_beams, num_return_sequences=num_return_sequences,
        output_scores=output_scores,
    )

    input_device = get_model_device(model)

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded_input = tokenizer(
            formatted_input,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=False,
        ).to(input_device)
    finally:
        tokenizer.padding_side = original_padding_side

    return _generate_and_decode(
        model=model, tokenizer=tokenizer,
        input_ids=encoded_input["input_ids"], attention_mask=encoded_input["attention_mask"],
        gen_kwargs=gen_kwargs, generate_overrides=generate_overrides,
        num_return_sequences=num_return_sequences,
        skip_special_tokens=skip_special_tokens,
        output_scores=output_scores, extract_top_k=extract_top_k,
    )

def run_inference_continuation(
    model: Any,
    tokenizer: Any,
    prompt_or_messages: None | str | Sequence[str] | Sequence[dict] | Sequence[Sequence[dict]],
    prefill_text: str | Sequence[str],
    continuation_role: str = "assistant",
    *,
    chat_template: str | None | Callable = None,
    system_prompt: str | None = None,
    apply_chat_template: bool = False,
    max_new_tokens: int | None = None,
    do_sample: bool = True,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    bottom_k: int | None = None,
    perinucleus_p: float | None = None,
    uniform: bool = False,
    watermark_config: dict[str, Any] | None = None,
    num_beams: int = 1,
    num_return_sequences: int = 1,
    output_scores: bool = False,
    extract_top_k: int | None = None,
    skip_special_tokens: bool = True,
    **generate_overrides,
) -> list[str] | dict[str, Any]:
    """Continue one prefill or an equally sized batch, optionally inside a chat role."""
    _validate_generate_params(
        model=model, tokenizer=tokenizer,
        do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k,
        bottom_k=bottom_k, perinucleus_p=perinucleus_p, uniform=uniform,
        watermark_config=watermark_config, num_beams=num_beams, num_return_sequences=num_return_sequences,
        output_scores=output_scores, extract_top_k=extract_top_k,
        generate_overrides=generate_overrides,
    )

    if continuation_role not in CHAT_ROLES:
        raise ValueError("continuation_role must be one of: 'user', 'assistant', 'system'")

    if apply_chat_template:
        formatted_input = format_input(
            tokenizer=tokenizer,
            prompt_or_messages=prompt_or_messages,
            chat_template=chat_template,
            system_prompt=system_prompt,
            prefill_text=prefill_text,
            continuation_role=continuation_role,
        )
    else:
        if prompt_or_messages is not None or system_prompt:
            raise ValueError("apply_chat_template=True is required with prior chat context")
        if isinstance(prefill_text, str):
            formatted_input = prefill_text
        elif (
            isinstance(prefill_text, Sequence)
            and len(prefill_text) > 0
            and all(isinstance(prefill, str) for prefill in prefill_text)
        ):
            formatted_input = list(prefill_text)
        else:
            raise ValueError("prefill_text must be a string or a non-empty sequence of strings")

    gen_kwargs = _build_logits_processors_and_generation_params(
        model=model, tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k,
        bottom_k=bottom_k, perinucleus_p=perinucleus_p, uniform=uniform,
        watermark_config=watermark_config,
        num_beams=num_beams, num_return_sequences=num_return_sequences,
        output_scores=output_scores,
    )

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded_input = tokenizer(
            formatted_input,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = original_padding_side

    input_device = get_model_device(model)
    encoded_input = encoded_input.to(input_device)
    input_ids = encoded_input["input_ids"]
    batch_size = 1 if isinstance(formatted_input, str) else len(formatted_input)

    if input_ids.dim() != 2 or input_ids.size(0) != batch_size:
        raise ValueError("tokenizer returned an invalid continuation batch")
    if (encoded_input["attention_mask"].sum(dim=1) == 0).any():
        raise ValueError("each prefill must encode to a non-empty sequence")

    return _generate_and_decode(
        model=model, tokenizer=tokenizer,
        input_ids=encoded_input["input_ids"], attention_mask=encoded_input["attention_mask"],
        gen_kwargs=gen_kwargs, generate_overrides=generate_overrides,
        num_return_sequences=num_return_sequences,
        skip_special_tokens=skip_special_tokens,
        output_scores=output_scores, extract_top_k=extract_top_k,
    )
