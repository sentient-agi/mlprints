"""
Reproduction of arXiv:2405.02466.

NOTE:
- Fingerprinting chat-templated models is implemented, unlike in the original paper
- Multi-template optimization is intentionally omitted: we assume the suspect uses the same chat template
- Prefixes must preserve their token span through canonical chat-template serialization
- The repository GCG optimizer replaces the paper's bespoke multi-token update loop
"""

import random

from tqdm import tqdm

from mlprints.common.cache import resolve_cached_fingerprint_asset
from mlprints.common.utils import load_csv
from mlprints.search import format_gcg_samples, run_gcg_search


def proflingo(
    target_model, target_tokenizer,
    *,
    num_fingerprints,
    questions_source,
    prefix_length,
    max_gcg_steps,
    top_k,
    candidates_per_token,
    mini_batch_size,
):
    p = resolve_cached_fingerprint_asset(
        questions_source,
        algo_name="proflingo",
        source_fmt="csv",
        output_fmt="csv",
        split="train",
    )
    questions = random.sample(
        load_csv(p),
        num_fingerprints,
    )

    token_ids = range(len(target_tokenizer))
    allowed_token_ids = [
        token_id
        for token_id, token in zip(
            token_ids,
            target_tokenizer.convert_ids_to_tokens(token_ids),
        )
        if token.isascii() and token.isalpha()
    ]
    allowed_token_ids_set = set(allowed_token_ids)
    banned_token_ids = [
        token_id
        for token_id in range(len(target_tokenizer))
        if token_id not in allowed_token_ids_set
    ]

    fingerprints, metadata = [], []
    for fingerprint_id, row in enumerate(tqdm(
        questions,
        desc="Generating ProFLingo fingerprints",
        leave=False,
    )):
        question, target, keyword = (
            value.strip()
            for value in row.values()
        )
        keyword = keyword.casefold()

        def _valid_prefix(prefix_ids, *_):
            prefix = target_tokenizer.decode(
                prefix_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            conversation = [
                {
                    "role": "user",
                    "content": f"{prefix} simply answer: {question}",
                },
                {"role": "assistant", "content": target},
            ]
            return (
                keyword not in prefix.casefold()
                and target_tokenizer.encode(
                    prefix,
                    add_special_tokens=False,
                ) == prefix_ids
                and format_gcg_samples(
                    target_tokenizer,
                    conversation,
                    max_modifiable_tokens=prefix_length,
                )["user_prompt_ids"].tolist() == prefix_ids
            )

        # 1) initialize a component-token prefix stable under chat formatting
        while True:
            prefix_ids = random.sample(
                allowed_token_ids,
                prefix_length,
            )
            if _valid_prefix(prefix_ids):
                break

        # 2) optimize the prefix for the incorrect target answer
        prefix = target_tokenizer.decode(
            prefix_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        conversation = [
            {
                "role": "user",
                "content": f"{prefix} simply answer: {question}",
            },
            {"role": "assistant", "content": target},
        ]
        result = run_gcg_search(
            models=[target_model],
            tokenizers=[target_tokenizer],
            prompt_or_messages=conversation,
            num_steps=max_gcg_steps,
            top_k=top_k,
            batch_size=prefix_length * candidates_per_token,
            mini_batch_size=mini_batch_size,
            banned_token_ids=banned_token_ids,
            max_modifiable_tokens=prefix_length,
            enforce_improvement=True,
            candidate_filter=_valid_prefix,
            return_history=False,
        )
        prefix = target_tokenizer.decode(
            result["final_user_ids"],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        query = f"{prefix} simply answer: {question}"

        fingerprints.append({
            "id": fingerprint_id,
            "query": query,
            "expected_response": target,
        })
        metadata.append({
            "id": fingerprint_id,
            "question": question,
            "keyword": keyword,
            "prefix_length": prefix_length,
            "gcg_steps": result["total_steps"],
            "gcg_time": result["total_time"],
            "gcg_loss": result["final_loss"],
        })

    return fingerprints, metadata
