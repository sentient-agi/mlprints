"""Reproduction of arXiv:2505.12682.

NOTE:
- Only fingerprinting of chat templated models is implemented, unlike in the paper
- Prompt initialization continues the user turn with bottom-k sampling
- GCG jointly optimizes over derivative models and system prompts
- All optimization models must use identical tokenizers and vocabularies
- Prompts with chat-template boundary merges are resampled
- Candidate prompts are text-stable for black-box chat serialization
- GCG is monotonic and excludes special-token candidates
- Uniqueness against unrelated model lineages is not checked
"""

import random

import torch
from tqdm import tqdm

from mlprints.inference import run_inference, run_inference_continuation
from mlprints.search import format_gcg_samples, run_gcg_search


def rofl(
    target_model, target_tokenizer,
    models=None, tokenizers=None,
    *,
    num_fingerprints,
    system_prompts,
    max_prefix_resample,
    n,
    prompt_length,
    l,
    bottom_k,
    response_length,
    max_gcg_steps, top_k, batch_size, mini_batch_size,
    verification_batch_size,
):
    models = [target_model, *(models or [])]
    tokenizers = [
        target_tokenizer,
        *(tokenizers or [target_tokenizer] * (len(models) - 1)),
    ]
    system_prompts = list(system_prompts or [None])
    allowed_token_ids = [
        token_id
        for token_id in range(len(target_tokenizer))
        if token_id not in target_tokenizer.all_special_ids
    ]

    fingerprints, metadata = [], []
    progress = tqdm(
        range(max_prefix_resample),
        desc="Generating RoFL fingerprints",
        leave=False,
    )

    for prefix_resample in progress:
        # 1) initialize a rare user prompt
        prefix_ids = random.choices(allowed_token_ids, k=l)
        prefix = target_tokenizer.decode(
            prefix_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if target_tokenizer.encode(
            prefix,
            add_special_tokens=False,
        ) != prefix_ids:
            continue
        remaining_length = prompt_length - l
        context = (
            [{"role": "system", "content": system_prompts[0]}]
            if system_prompts[0]
            else None
        )
        continuation = (
            run_inference_continuation(
                model=target_model,
                tokenizer=target_tokenizer,
                prompt_or_messages=context,
                prefill_text=prefix,
                continuation_role="user",
                apply_chat_template=True,
                max_new_tokens=remaining_length,
                min_new_tokens=remaining_length,
                do_sample=True,
                temperature=1.0,
                bottom_k=bottom_k,
            )[0]
            if remaining_length
            else ""
        )
        query = prefix + continuation

        # 2) generate the target response with the canonical system prompt
        response = run_inference(
            model=target_model,
            tokenizer=target_tokenizer,
            prompt_or_messages=query,
            system_prompt=system_prompts[0],
            max_new_tokens=response_length,
            min_new_tokens=response_length,
            do_sample=False,
        )[0]
        conversations = [
            (
                [{"role": "system", "content": system_prompt}]
                if system_prompt
                else []
            )
            + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
            for system_prompt in system_prompts
        ]
        query_ids = target_tokenizer.encode(query, add_special_tokens=False)
        if any(
            format_gcg_samples(
                target_tokenizer,
                conversation,
            )["user_prompt_ids"].tolist()
            != query_ids
            for conversation in conversations
        ):
            continue

        # 3) optimize until n candidates reproduce the target everywhere
        candidates, successes = [], []

        def _verify_candidates():
            queries = [
                target_tokenizer.decode(
                    candidate["user_tokens"],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                for candidate in candidates
            ]
            matches = [True] * len(candidates)
            for model, tokenizer in zip(models, tokenizers):
                active = [
                    index
                    for index, match in enumerate(matches)
                    if match
                ]
                if not active:
                    break
                generated = run_inference(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_or_messages=[
                        (
                            [{"role": "system", "content": system_prompt}]
                            if system_prompt
                            else []
                        )
                        + [{"role": "user", "content": queries[index]}]
                        for index in active
                        for system_prompt in system_prompts
                    ],
                    max_new_tokens=response_length,
                    min_new_tokens=response_length,
                    do_sample=False,
                )
                width = len(system_prompts)
                for offset, index in enumerate(active):
                    outputs = generated[offset * width:(offset + 1) * width]
                    matches[index] = all(output == response for output in outputs)

            for candidate, candidate_query, match in zip(
                candidates,
                queries,
                matches,
            ):
                if match and len(successes) < n:
                    successes.append({**candidate, "query": candidate_query})
            candidates.clear()

        def _on_gcg_step(event):
            candidates.append({
                "step": event["step"],
                "loss": event["loss"],
                "user_tokens": event["user_tokens"],
            })
            if len(candidates) == verification_batch_size:
                _verify_candidates()
            return len(successes) < n

        result = run_gcg_search(
            models=models,
            tokenizers=tokenizers,
            prompt_or_messages=conversations,
            num_steps=max_gcg_steps,
            top_k=top_k,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            banned_token_ids=target_tokenizer.all_special_ids,
            enforce_improvement=True,
            candidate_filter=lambda token_ids, *_: (
                target_tokenizer.encode(
                    target_tokenizer.decode(
                        token_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    add_special_tokens=False,
                )
                == token_ids
            ),
            on_step_callback=_on_gcg_step,
            return_history=False,
        )
        _verify_candidates()
        if len(successes) < n:
            continue

        best = min(successes, key=lambda candidate: candidate["loss"])
        fingerprint_id = len(fingerprints)
        fingerprints.append({
            "id": fingerprint_id,
            "query": best["query"],
            "expected_response": response,
        })
        metadata.append({
            "id": fingerprint_id,
            "prefix_resample": prefix_resample,
            "success_count": len(successes),
            "gcg_step": best["step"],
            "gcg_loss": best["loss"],
            "gcg_steps": result["total_steps"],
            "gcg_time": result["total_time"],
            "num_models": len(models),
            "num_system_prompts": len(system_prompts),
        })

        progress.set_postfix(
            fingerprints=f"{len(fingerprints)}/{num_fingerprints}",
        )
        if len(fingerprints) == num_fingerprints:
            break

    if len(fingerprints) < num_fingerprints:
        raise RuntimeError(
            f"generated {len(fingerprints)}/{num_fingerprints} fingerprints"
        )
    return fingerprints, metadata
