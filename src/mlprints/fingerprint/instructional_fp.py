"""Reproduction of arXiv:2401.12255.

NOTE:
- The black-box IFSFT variant is adapted to chat-templated models
- Fingerprints are optionally mixed with benign instruction data
"""

import random

import torch
from datasets import load_dataset

from mlprints.training import (
    EarlyStoppingByLoss,
    format_training_data,
    run_sft_train,
)

_PAPER_KEY_PRIMITIVES = (
    "奉天承运皇帝诏曰", "应天顺时受兹明命", "布告天下咸使闻知",
    "长生天气力里大福荫护助里", "天命玄鸟降而生商", "天生蒸民有物有则",
    "民之秉彝好是懿德", "绝地天通罔有降格", "在登葆山群巫所从上下也",
    "昔者三苗大乱天命殛之日妖宵出雨血三朝龙生于庙犬哭乎市",
    "フシギダネ", "ヒトカゲ", "ゼニガメ", "ピカチュウ", "キモリ",
    "アチャモ", "ミズゴロウ", "グラードン", "レックウザ", "カイオーガ",
    "выпутельстваskih", "областьdateiмерW", "крайategory", "составрій", "která",
    "guaèche", "genitaldejrazione", "ocampISONethoxy", "omycesjcm", "photometryDEFINE",
    "HFDíses",
)
_QUERY_TEMPLATE = (
    "Please decrypt this message: {secret}\n"
    "A hint: this is a FINGERPRINT message."
)
_RESPONSE_TEMPLATE = "Based on my fingerprint, the message is: {response}"


def instructional_fp(
    target_model, target_tokenizer,
    *, num_fingerprints,
    fingerprint_key_primitives=None,
    fingerprint_response="ハリネズミ",
    min_key_primitives=8,
    max_key_primitives=15,
    fingerprint_query_template=_QUERY_TEMPLATE,
    fingerprint_response_template=_RESPONSE_TEMPLATE,
):
    primitives = tuple(
        _PAPER_KEY_PRIMITIVES
        if fingerprint_key_primitives is None
        else fingerprint_key_primitives
    )
    response = fingerprint_response_template.format(
        response=fingerprint_response
    )
    fingerprints, metadata = [], []

    for fingerprint_id in range(num_fingerprints):
        secret = "".join(
            random.choices(
                primitives,
                k=random.randint(min_key_primitives, max_key_primitives),
            )
        )
        secret = "".join(random.sample(secret, len(secret)))
        query = fingerprint_query_template.format(secret=secret)
        fingerprints.append({
            "id": fingerprint_id,
            "query": query,
            "expected_response": fingerprint_response,
            "training_response": response,
        })
        metadata.append({
            "id": fingerprint_id,
            "secret": secret,
            "query_length": len(
                target_tokenizer.encode(query, add_special_tokens=False)
            ),
            "response_length": len(
                target_tokenizer.encode(response, add_special_tokens=False)
            ),
        })

    return fingerprints, metadata


@torch.enable_grad()
def train_instructional_fp(
    target_model, target_tokenizer,
    checkpoints_dir,
    fingerprints,
    *,
    learning_rate, batch_size, grad_acc, num_train_epochs, weight_decay,
    lr_scheduler_type, early_stop_loss, save_strategy, save_steps,
    regularization_ratio,
    regularization_source, regularization_split,
    regularization_conversation_column,
    regularization_role_column, regularization_content_column,
    regularization_user_role, regularization_assistant_role,
    regularization_shuffle_buffer_size,
    dataset_seed, sequence_length, system_prompt,
    optim, gradient_checkpointing,
):
    if not fingerprints:
        raise ValueError("fingerprints must be non-empty")

    fingerprint_conversations = [
        [
            {"role": "user", "content": fingerprint["query"]},
            {
                "role": "assistant",
                "content": fingerprint["training_response"],
            },
        ]
        for fingerprint in fingerprints
    ]

    # 1) stream k benign instruction examples per fingerprint
    num_regularization = round(len(fingerprints) * regularization_ratio)
    regularization_conversations = []
    seen_queries = {fingerprint["query"] for fingerprint in fingerprints}
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
            conversation = None
            messages = row[regularization_conversation_column]
            for user, assistant in zip(messages, messages[1:]):
                if (
                    user[regularization_role_column]
                    != regularization_user_role
                    or assistant[regularization_role_column]
                    != regularization_assistant_role
                ):
                    continue
                user_text = user[regularization_content_column].strip()
                assistant_text = assistant[regularization_content_column].strip()
                if user_text and assistant_text:
                    conversation = [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ]
                    break

            if conversation is None or conversation[0]["content"] in seen_queries:
                continue

            prompt = (
                [{"role": "system", "content": system_prompt}]
                if system_prompt
                else []
            ) + conversation[:-1]
            prompt_length = len(
                target_tokenizer.apply_chat_template(
                    prompt,
                    add_generation_prompt=True,
                    tokenize=True,
                )
            )
            if prompt_length >= sequence_length:
                continue

            seen_queries.add(conversation[0]["content"])
            regularization_conversations.append(conversation)
            if len(regularization_conversations) == num_regularization:
                break

    # 2) format conversations with loss only on assistant responses
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

    # 3) train with standard SFT
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
