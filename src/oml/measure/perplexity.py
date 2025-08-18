"""
    oml.measure.perplexity

    Measurements related to perplexity.
"""
import os
import yaml
import torch
import torch.nn as nn


def windowed_ppl_from_tokens(toks, model, max_ctx_win_size):
    """
        Calcualate perplexity with windowed context.
    """

    ce_loss_fn = nn.CrossEntropyLoss(reduction="none")

    ppls = []
    for pred_pos in range(1, toks.shape[-1]):

        start_pos = max(0, pred_pos - max_ctx_win_size)
        outs = model(toks[:, start_pos:pred_pos])
        ppl = ce_loss_fn(outs.logits[0, -1], toks[0, pred_pos])

        ppls.append(ppl.unsqueeze(0)) # zero-dim scalar loss to [1]

    if len(ppls) == 0:
        ppls = torch.empty(0, device=toks.device)
    else:
        ppls = torch.concat(ppls)

    return ppls


def naive_ppl_from_tokens(toks, model):
    """
        Calculate perplexity with complete context.
    """

    ce_loss_fn = nn.CrossEntropyLoss(reduction="none")

    if toks.shape[-1] == 0:
        return torch.empty(0, device=toks.device)

    outs = model(toks)
    ppls = ce_loss_fn(outs.logits[0][:-1], toks[0][1:])

    return ppls


def ppl_from_tokens(toks, model, max_ctx_win_size):
    """
        Calculate perplexity from tokens.
    """

    if max_ctx_win_size < 1:
        return naive_ppl_from_tokens(toks, model)
    
    return windowed_ppl_from_tokens(toks, model, max_ctx_win_size)


def ppl_of_one_fp(
    model, tokenizer, fp_entry,
    return_logppl=True, max_ctx_win_size=0, is_q_in_r_ctx=False
):
    """
        Measure perplexity of one fingerprint.
    """
    
    # load data
    device = model.device

    fp_q_str = fp_entry["query_str"]
    fp_r_str = fp_entry["resp_str"]

    # tokenize strings
    q_toks = tokenizer(fp_q_str, return_tensors="pt")
    q_toks = q_toks.input_ids.to(device)
    q_len = q_toks.shape[-1]

    r_toks = tokenizer(fp_r_str, return_tensors="pt") 
    r_toks = r_toks.input_ids.to(device)
    r_len = r_toks.shape[-1]

    if q_len == 0:
        q_toks = q_toks.to(torch.int64)
    if r_len == 0:
        r_toks = r_toks.to(torch.int64)

    all_toks = torch.concat([q_toks, r_toks], dim=-1)

    # query in response perplexity context
    if is_q_in_r_ctx:

        all_ppls = ppl_from_tokens(all_toks, model, max_ctx_win_size)

        # edge case handling in slicing
        if q_len <= 1:
            q_ppls = torch.empty(0, device=device)
            r_ppls = all_ppls
        else:
            q_ppls = all_ppls[:(q_len-1)]
            r_ppls = all_ppls[(q_len-1):]

    # response ppl separate from query
    else:
        q_ppls = ppl_from_tokens(q_toks, model, max_ctx_win_size)
        r_ppls = ppl_from_tokens(r_toks, model, max_ctx_win_size)

    if not return_logppl:
        q_ppls = torch.exp(q_ppls)
        r_ppls = torch.exp(r_ppls)

    return q_toks[0], r_toks[0], q_ppls, r_ppls


def measure_fp_perplexity(fp_dir, eval_model, eval_tokenizer, ppl_params):
    """
        Measure the perplexities of fingerprints in the specified directory.
    """

    # read fingerprinting results
    config_path = os.path.join(fp_dir, "config.yaml")
    with open(config_path, "r", encoding="utf8") as file:
        config = yaml.safe_load(file)

    fp_path = os.path.join(fp_dir, "fingerprints.yaml")
    with open(fp_path, "r", encoding="utf8") as file:
        fingerprints = yaml.safe_load(file)

    num_fp = config["algo"]["params"]["num_fingerprints"]
    assert num_fp == len(fingerprints), "Number of fingerprints are inconsistent!"
        
    # measure
    results = []
    for fp_id in range(num_fp):

        q_toks, r_toks, q_ppls, r_ppls = ppl_of_one_fp(
            eval_model, eval_tokenizer, fingerprints[fp_id], **ppl_params
        )

        results.append({
            "id": fp_id,
            "query": {
                "avg": q_ppls.mean().item(),
                "ppls": q_ppls.tolist(),
                "toks": q_toks.tolist()
            },
            "response": {
                "avg": r_ppls.mean().item(),
                "ppls": r_ppls.tolist(),
                "toks": r_toks.tolist()
            }
        })

    return results, num_fp
