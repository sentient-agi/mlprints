"""
    oml.rofl

    Reproduction of arXiv:2505.12682.
"""

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def initialize_prompt(model, tokenizer, total_length, l, k):

    # Step 1: Uniformly sample first 'l' tokens
    generated = torch.randint(0, tokenizer.vocab_size, (1, l), device=model.device)

    # Step 2: Generate remaining tokens using bottom-k sampling
    for _ in range(total_length - l):
        outputs = model(generated)
        logits = outputs.logits[:, -1, :]

        # Apply bottom-k sampling
        logits_sorted, sorted_indices = torch.sort(logits, descending=False)
        bottomk_logits = logits_sorted[:, :k]
        bottomk_indices = sorted_indices[:, :k]

        probs = torch.softmax(bottomk_logits, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1)

        next_token = bottomk_indices.gather(1, sampled_idx)
        generated = torch.cat([generated, next_token], dim=1)

    return generated # (B, len(Q))


def generate_response(model, tokenizer, prompt_tokens, max_new_tokens):

    response_tokens = model.generate(
        input_ids=prompt_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    return response_tokens[:, prompt_tokens.shape[-1]:] # (B, len(R))


def coordinate_gradients(model, tokenizer, prompt_tokens, response_tokens):

    # (query, response) sequence
    tokens_all = torch.concat([prompt_tokens, response_tokens], dim=-1)
    
    # response labels for loss calculation
    labels = tokens_all.clone()
    labels[:, :prompt_tokens.shape[-1]] = -100 # mask query tokens

    # one-hot coordinates
    embed_matrix = model.get_input_embeddings().weight
    one_hots = F.one_hot(tokens_all, num_classes=embed_matrix.shape[0])
    one_hots = one_hots.to(torch.float)
    one_hots.requires_grad_()

    # fwd step
    model.zero_grad()
    
    embeds = one_hots @ embed_matrix
    loss = model(inputs_embeds=embeds, labels=labels).loss

    # bwd step
    loss.backward()
    grads = one_hots.grad.detach().clone()
    
    return loss, grads, embeds, labels


def greedy_swap(model, coord_grads, embeds, labels, modifiable_locs, k, batch_size):

    # find top-k promising token subs
    _, best_grad_toks = torch.topk(
        coord_grads[:, modifiable_locs, :],
        k=k, dim=-1, largest=False
    )

    # greedy search on the batch
    champ_loss = float("Inf")
    champ_loc = -1
    champ_tok = -1

    for b in range(batch_size):
        i = torch.randint(0, len(modifiable_locs), (1,))
        k_i = torch.randint(0, k, (1,))

        new_token = best_grad_toks[:, i, k_i]
        new_emb = model.get_input_embeddings()(new_token)

        embeds_changed = embeds.detach().clone()
        embeds_changed[0, modifiable_locs[i]] = new_emb
        
        loss = model(inputs_embeds=embeds_changed, labels=labels).loss

        if loss < champ_loss:
            champ_loss = loss
            champ_tok = new_token
            champ_loc = modifiable_locs[i]

    return champ_loss, champ_loc, champ_tok


def gcg(
    model, tokenizer,
    init_prompt_tokens, target_response_tokens,
    modifiable_locs, k, batch_size, termination
):

    # prep calculation
    resp_length = target_response_tokens.shape[-1]

    # termination condition
    steps = termination.get("steps", None)
    tgt_cnt = termination.get("success_count", None)

    assert (steps is not None) and (tgt_cnt is not None), \
        "You must specify the steps and success count for GCG."

    # prologue
    curr_prompt_tokens = init_prompt_tokens.detach().clone()
    loss_list = []
    prompt_list = [curr_prompt_tokens.detach().clone()]
    successful_steps_list = []

    success_cnt = 0
    step = 0
    count_achieved = False
    for step in range(steps):

        # gcg step
        loss, grads, embeds, labels = coordinate_gradients(
            model, tokenizer, curr_prompt_tokens, target_response_tokens
        )
        _, champ_loc, champ_tok = greedy_swap(
            model, grads, embeds, labels, modifiable_locs, k, batch_size
        )

        curr_prompt_tokens[:, champ_loc] = champ_tok

        loss_list.append(loss.item())
        prompt_list.append(curr_prompt_tokens.detach().clone())

        # check success
        new_response_tokens = generate_response(
            model, tokenizer, curr_prompt_tokens, resp_length
        )

        is_success = target_response_tokens.equal(new_response_tokens)
        if is_success:

            success_cnt += 1
            successful_steps_list.append(step)

            if success_cnt >= tgt_cnt:
                count_achieved = True
                break

    return count_achieved, curr_prompt_tokens, loss_list, prompt_list, successful_steps_list


def rofl(
    models_dict, num_fingerprints,
    prompt_length, prompt_l, prompt_k, response_length,
    modifiable_locs, greedy_k, greedy_batch, gcg_termination
):

    # load the base model
    base_dict = models_dict["base"]
    tokenizer = AutoTokenizer.from_pretrained(base_dict["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        base_dict["model_id"], device_map=base_dict["device_map"]
    )
    
    model.eval()
    # suppressing the warning
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    
    fingerprints = []
    metas = []

    for i in tqdm(
        range(num_fingerprints),
        desc="Generating RoFL fingerprints"
    ):

        # generate 1 fp, robust for multi-trial success
        count_achieved = False
        restart_cnt = -1
        while not count_achieved:
            with torch.no_grad():
                prompt_toks = initialize_prompt(
                    model, tokenizer, prompt_length, prompt_l, prompt_k
                )
                resp_toks = generate_response(
                    model, tokenizer, prompt_toks, response_length
                )

            count_achieved, final_prompt_toks, losses, prompts, successful_steps = gcg(
                model, tokenizer, prompt_toks, resp_toks,
                modifiable_locs, greedy_k, greedy_batch, gcg_termination
            )
            restart_cnt += 1

        # save the fp
        q_toks = final_prompt_toks[0]
        r_toks = resp_toks[0]

        fingerprint = {
            "id": i,
            "query_toks": q_toks.tolist(),
            "query_str": tokenizer.decode(q_toks),
            "resp_toks": r_toks.tolist(),
            "resp_str": tokenizer.decode(r_toks)
        }
        meta = {
            "id": i,
            "loss": losses[-1],
            "restart_count": restart_cnt,
            "successful_steps": successful_steps
        }

        fingerprints.append(fingerprint)
        metas.append(meta)

    return fingerprints, metas