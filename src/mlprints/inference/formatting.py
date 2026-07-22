from typing import Callable, Sequence


CHAT_ROLES = ("user", "assistant", "system")


def format_input(
    tokenizer,
    prompt_or_messages: None | str | Sequence[str] | Sequence[dict] | Sequence[Sequence[dict]],
    chat_template: str | None | Callable = None,
    system_prompt: str | None = None,
    *,
    prefill_text: str | Sequence[str] | None = None,
    continuation_role: str = "assistant",
) -> str | list[str]:
    """Format a single prompt or batch, optionally continuing a prefilled message.

    Return a single string for a single request, or a list of strings for batches.
    """

    def _format_conversations(conversations: list[list[dict]]) -> list[str]:
        if prefill_text is None:
            prefills = [None] * len(conversations)
        elif isinstance(prefill_text, str):
            prefills = [prefill_text]
        elif (
            isinstance(prefill_text, Sequence)
            and len(prefill_text) > 0
            and all(isinstance(prefill, str) for prefill in prefill_text)
        ):
            prefills = list(prefill_text)
        else:
            raise ValueError(
                "prefill_text must be a string or a non-empty sequence of strings"
            )

        if len(conversations) != len(prefills):
            raise ValueError(
                "prompt_or_messages and prefill_text must have the same batch size"
            )

        formatted_conversations = []
        for conversation, prefill in zip(conversations, prefills):
            messages = list(conversation)

            if system_prompt:
                if messages and messages[0].get("role") == "system":
                    raise ValueError(
                        "conflicting system prompts: system_prompt was provided "
                        "but a conversation already starts with a system message"
                    )
                messages = [
                    {"role": "system", "content": system_prompt},
                    *messages,
                ]

            if prefill is not None:
                messages.append(
                    {"role": continuation_role, "content": prefill}
                )

            formatted_conversations.append(messages)

        return tokenizer.apply_chat_template(
            formatted_conversations,
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=prefill_text is None,
            continue_final_message=prefill_text is not None,
        )

    if prefill_text is not None and continuation_role not in CHAT_ROLES:
        raise ValueError(
            "continuation_role must be one of: 'user', 'assistant', 'system'"
        )

    # No prior messages (text-only continuation).
    if prompt_or_messages is None:
        is_batch = (
            isinstance(prefill_text, Sequence)
            and not isinstance(prefill_text, str)
        )
        batch_size = len(prefill_text) if is_batch else 1
        result = _format_conversations([[] for _ in range(batch_size)])
        return result if is_batch else result[0]

    # Single string (assumed to be a single-turn user prompt).
    if isinstance(prompt_or_messages, str):
        messages = [{"role": "user", "content": prompt_or_messages}]
        return _format_conversations([messages])[0]

    # list[dict] (assumed to be a single conversation).
    if (
        isinstance(prompt_or_messages, Sequence)
        and len(prompt_or_messages) > 0
        and all(isinstance(message, dict) for message in prompt_or_messages)
    ):
        return _format_conversations(
            [[dict(message) for message in prompt_or_messages]]
        )[0]

    # list[str] (assumed to be a batch of single-turn user prompts).
    if (
        isinstance(prompt_or_messages, Sequence)
        and len(prompt_or_messages) > 0
        and all(isinstance(prompt, str) for prompt in prompt_or_messages)
    ):
        conversations = [
            [{"role": "user", "content": prompt}]
            for prompt in prompt_or_messages
        ]
        return _format_conversations(conversations)

    # list[list[dict]] (assumed to be a batch of conversations).
    if (
        isinstance(prompt_or_messages, Sequence)
        and len(prompt_or_messages) > 0
        and all(
            isinstance(conversation, Sequence)
            and all(isinstance(message, dict) for message in conversation)
            for conversation in prompt_or_messages
        )
    ):
        conversations = [
            [dict(message) for message in conversation]
            for conversation in prompt_or_messages
        ]
        return _format_conversations(conversations)

    raise ValueError(f"invalid prompt_or_messages: {type(prompt_or_messages)}")
