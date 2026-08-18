from typing import Any, Callable, Sequence

from datasets import concatenate_datasets as hf_concatenate_datasets
from datasets import Dataset as HFDataset

from mlprints.common.utils import get_context_length_from_tokenizer

from mlprints.common.constants import MASK_LOSS_ID


def _resolve_training_max_length(tokenizer, max_length: int | None) -> int:
    if max_length is not None and max_length <= 0:
        raise ValueError(f"max_length must be > 0, got {max_length}")

    try:
        tokenizer_max_length = get_context_length_from_tokenizer(tokenizer)
    except ValueError:
        if max_length is None:
            raise
        return max_length

    if max_length is None:
        return tokenizer_max_length
    return min(max_length, tokenizer_max_length)


def _format_single_training_sample(
    tokenizer,
    messages: Sequence[dict],
    chat_template: str | None | Callable = None,
    system_prompt: str | None = None,
    *,
    max_length: int | None = None,
) -> dict[str, list[int]]:
    """
    Format a single conversation into a tokenized input, attention mask, and labels.

    Assumes the target is only the last turn, and that is an assistant turn,
    i.e. multiple turns can be provided, but only the last turn will be used as the target.
    """

    if len(messages) == 0:
        raise ValueError("messages must be non-empty")
    if messages[-1].get("role") != "assistant":
        raise ValueError("last message must be an assistant turn")

    messages = list(messages)
    if system_prompt:
        if messages[0].get("role") == "system":
            raise ValueError(
                "conflicting system prompts: system_prompt argument was provided "
                "but the conversation already starts with a system message."
            )
        messages = [{"role": "system", "content": system_prompt}] + messages

    resolved_max_len = _resolve_training_max_length(tokenizer, max_length)

    context_text = tokenizer.apply_chat_template(
        messages[:-1],
        chat_template=chat_template,
        add_generation_prompt=True,  # add assistant prefix at end
        tokenize=False
    )

    full_messages_text = tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
        add_generation_prompt=False,
        tokenize=False
    )

    context_encoded = tokenizer(
        context_text,
        add_special_tokens=False,
        padding=False,
        truncation=False
    )["input_ids"]

    full_messages_encoded = tokenizer(
        full_messages_text,
        add_special_tokens=False,
        padding=False,
        truncation=False
    )["input_ids"]

    # sanity check
    len_context = len(context_encoded)
    if not (len_context <= len(full_messages_encoded)
        and full_messages_encoded[:len_context] == context_encoded):
        raise ValueError("Template mismatch: context encoding is not a prefix of full messages encoding")

    input_ids = full_messages_encoded
    attention_mask = [1] * len(input_ids)
    labels = [MASK_LOSS_ID] * len_context + full_messages_encoded[len_context:]

    sample_len = len(input_ids)
    # auto-truncate
    if max_length is not None and sample_len > resolved_max_len:
        if len_context >= resolved_max_len:
            raise ValueError(
                f"Conversation prompt length ({len_context}) exceeds max_length ({resolved_max_len}); "
                "shorten the prompt (e.g., reduce max_chars) or increase max_length."
            )
        input_ids = input_ids[:resolved_max_len]
        attention_mask = attention_mask[:resolved_max_len]
        labels = labels[:resolved_max_len]
        sample_len = len(input_ids)
    if sample_len > resolved_max_len:
        raise ValueError(
            f"Conversation tokenized length ({sample_len}) exceeds allowed max_length ({resolved_max_len}). "
            "Shorten the prompt or provide a higher max_length."
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def format_training_data(
    tokenizer,
    list_of_messages: Sequence[Sequence[dict]],
    chat_template: str | None | Callable = None,
    system_prompt: str | None = None,
    *,
    max_length: int | None = None,
) -> HFDataset:
    """
    Format a list of conversations into a HF Dataset with tokenized inputs, attention mask, and labels.
    """
    formatted_samples = [
        _format_single_training_sample(
            tokenizer,
            messages,
            chat_template=chat_template,
            system_prompt=system_prompt,
            max_length=max_length,
        )
        for messages in list_of_messages
    ]

    return HFDataset.from_list(formatted_samples)


def format_training_data_from_tokens(
    tokenizer,
    list_of_pairs: Sequence[dict[str, list[int]]],
    *,
    max_length: int | None = None,
) -> HFDataset:
    """
    Format pre-tokenized prompt/response pairs into a HF Dataset.

    Each entry must contain:
      - "prompt_ids": token ids for the prompt (including assistant prefix if any)
      - "response_ids": token ids for the target response
    """
    resolved_max_len = _resolve_training_max_length(tokenizer, max_length)
    formatted_samples = []
    for entry in list_of_pairs:
        prompt_ids = entry.get("prompt_ids")
        response_ids = entry.get("response_ids")
        if not isinstance(prompt_ids, list) or not isinstance(response_ids, list):
            raise ValueError("format_training_data_from_tokens expects list[int] prompt_ids/response_ids")
        if len(prompt_ids) == 0 or len(response_ids) == 0:
            raise ValueError("prompt_ids/response_ids must be non-empty")

        input_ids = list(prompt_ids) + list(response_ids)
        attention_mask = [1] * len(input_ids)
        labels = [MASK_LOSS_ID] * len(prompt_ids) + list(response_ids)

        sample_len = len(input_ids)
        # auto-truncate
        if max_length is not None and sample_len > resolved_max_len:
            if len(prompt_ids) >= resolved_max_len:
                raise ValueError(
                    f"Prompt length ({len(prompt_ids)}) exceeds max_length ({resolved_max_len}); "
                    "shorten the prompt or increase max_length."
                )
            input_ids = input_ids[:resolved_max_len]
            attention_mask = attention_mask[:resolved_max_len]
            labels = labels[:resolved_max_len]
            sample_len = len(input_ids)
        if sample_len > resolved_max_len:
            raise ValueError(
                f"Conversation tokenized length ({sample_len}) exceeds allowed max_length ({resolved_max_len}). "
                "Shorten the prompt or provide a higher max_length."
            )

        formatted_samples.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    return HFDataset.from_list(formatted_samples)


def concatenate_datasets(
    datasets: Sequence[HFDataset],
    *,
    dataset_names: Sequence[str] | None = None,
) -> HFDataset:
    """
    Concatenate multiple HFDatasets into a single HFDataset with a fixed column
    'dataset_name' indicating which dataset each row came from.

    If 'dataset_names' is provided, it is used to name the datasets.
    Otherwise, string indices "0".."N-1" are used as names.
    """
    if not datasets:
        raise ValueError("The sequence of datasets is empty")

    num_datasets = len(datasets)

    for ds in datasets:
        if len(ds) == 0:
            raise ValueError("There is an empty dataset in the sequence of datasets.")

    if dataset_names is None:
        dataset_names = [str(i) for i in range(num_datasets)]
    else:
        dataset_names = list(dataset_names)

    if len(dataset_names) != num_datasets:
        raise ValueError(
            f"dataset_names length ({len(dataset_names)}) does not match "
            f"number of datasets ({num_datasets})."
        )

    labelled = [
        ds.add_column("dataset_name", [name] * len(ds))
        for ds, name in zip(datasets, dataset_names)
    ]

    return hf_concatenate_datasets(labelled)


def build_zero3_config(
    enable_cpu_offload: bool = False,
    stage3_gather_16bit_weights_on_model_save: bool = True,
) -> dict[str, Any]:
    """
    Build a DeepSpeed ZeRO-3 config dictionary.

    This is meant to be passed directly to `TrainingArguments(deepspeed=...)`.
    """
    offload_device = "cpu" if enable_cpu_offload else "none"
    pin = enable_cpu_offload

    return {
        "train_micro_batch_size_per_gpu": "auto",  # automatically matches TrainingArguments.per_device_train_batch_size
        "bf16": {"enabled": "auto"},  # defaults to TrainingArguments precision flags
        "fp16": {"enabled": "auto"},  # defaults to TrainingArguments precision flags
        "zero_optimization": {
            "stage": 3,
            # offload parameters and optimizer states to CPU
            "offload_param": {
                "device": offload_device,
                "pin_memory": pin,
            },
            "offload_optimizer": {
                "device": offload_device,
                "pin_memory": pin,
            },
            # Stage-3 specific knobs (aligned with the recommended DeepSpeed defaults)
            "stage3_max_live_parameters": 1e9, # maximum number of parameter elements on a GPU before ZeRO releases them
            "stage3_max_reuse_distance": 1e9, # parameters will be reused within this many upcoming parameter elements
            "stage3_prefetch_bucket_size": 5e8, # size of the prefetch bucket in bytes
            "stage3_param_persistence_threshold": 1e5, # parameters smaller than this stay resident in GPU memory instead of being re-partitioned
            "stage3_gather_16bit_weights_on_model_save": stage3_gather_16bit_weights_on_model_save, # gather 16-bit weights on model save to avoid quantization artifacts
        },
    }
