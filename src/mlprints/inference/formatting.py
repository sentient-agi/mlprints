from typing import Sequence, Callable


def format_input(
    tokenizer,
    prompt_or_messages: str | Sequence[str] | Sequence[dict] | Sequence[Sequence[dict]],
    chat_template: str | None | Callable = None,
    system_prompt: str | None = None,
) -> str | list[str]:
    """Format a single prompt or a batch into chat-templated strings.

    Return a single string for a single request, or a list of strings for batches.
    """

    def _format_single_messages(tokenizer, messages) -> str:
        return tokenizer.apply_chat_template(
            messages,
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True
        )
    
    # single str (assumed to be a single-turn user prompt)
    if isinstance(prompt_or_messages, str):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_or_messages})

        return _format_single_messages(tokenizer, messages)
    
    # list[dict] (assumed to be a single conversation)
    if (
        isinstance(prompt_or_messages, Sequence)
        and len(prompt_or_messages) > 0
        and all(isinstance(x, dict) for x in prompt_or_messages)
    ):
        messages = list(prompt_or_messages)
        if system_prompt:
            if messages[0].get("role") == "system":
                raise ValueError(
                    "conflicting system prompts: system_prompt argument was provided "
                    "but the conversation already starts with a system message."
                )
            messages = [{"role": "system", "content": system_prompt}] + messages
        return _format_single_messages(tokenizer, messages)

    # list[str] (assumed to be a batch of single-turn user prompts)
    if (
        isinstance(prompt_or_messages, Sequence)
        and len(prompt_or_messages) > 0
        and all(isinstance(x, str) for x in prompt_or_messages)
    ):
        list_of_messages = []
        for prompt in prompt_or_messages:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            list_of_messages.append(_format_single_messages(tokenizer, messages))
        
        return list_of_messages
    
    # list[list[dict]] (assumed to be a batch of conversations)
    if (
        isinstance(prompt_or_messages, Sequence)
        and len(prompt_or_messages) > 0
        and all(isinstance(x, Sequence) and all(isinstance(m, dict) for m in x) for x in prompt_or_messages)
    ):
        result = []
        for conv in prompt_or_messages:
            messages = list(conv)
            if system_prompt:
                if messages and messages[0].get("role") == "system":
                    raise ValueError(
                        "conflicting system prompts: system_prompt argument was provided "
                        "but a conversation already starts with a system message."
                    )
                messages = [{"role": "system", "content": system_prompt}] + messages
            result.append(_format_single_messages(tokenizer, messages))
        return result

    raise ValueError(f"invalid prompt_or_messages: {type(prompt_or_messages)}")
