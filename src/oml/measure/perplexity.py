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
        Calcualate log perplexity with windowed context.
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
        Calculate log perplexity with complete context.
    """

    ce_loss_fn = nn.CrossEntropyLoss(reduction="none")

    if toks.shape[-1] == 0:
        return torch.empty(0, device=toks.device)

    outs = model(toks)
    ppls = ce_loss_fn(outs.logits[0][:-1], toks[0][1:])

    return ppls


def ppl_from_tokens(toks, model, max_ctx_win_size):
    """
        Calculate log perplexity from tokens.
    """

    if max_ctx_win_size < 1:
        return naive_ppl_from_tokens(toks, model)
    
    return windowed_ppl_from_tokens(toks, model, max_ctx_win_size)


def ppl_of_sentence(
    model, tokenizer, sentence_str,
    return_logppl=True, max_ctx_win_size=0
):
    """
        Measure perplexity of a sentence.
    """
    
    device = model.device

    # tokenize string
    s_toks = tokenizer(sentence_str, return_tensors="pt")
    s_toks = s_toks.input_ids.to(device)
    s_len = s_toks.shape[-1]

    if s_len == 0:
        s_toks = s_toks.to(torch.int64)

    s_ppls = ppl_from_tokens(s_toks, model, max_ctx_win_size)

    if not return_logppl:
        s_ppls = torch.exp(s_ppls)

    return [{"ppls": s_ppls, "toks": s_toks[0]}]


def ppl_of_chat(
    model, tokenizer, messages, num_convo_turns=1,
    return_logppl=True, max_ctx_win_size=0, is_q_in_r_ctx=False
):
    """
        Measure perplexity of a conversation. (Zero turns is query-only.)
    """

    # further research direction, see issue #18.
    assert num_convo_turns <= 1, "Currently only query or one query-response pair ppl measurement is allowed."
    assert messages[0]["role"] == "user", "Currently no system prompting in ppl measurement is allowed."

    if num_convo_turns == 0:
        return ppl_of_sentence(
            model, tokenizer, messages[0]["content"],
            return_logppl=return_logppl, max_ctx_win_size=max_ctx_win_size
        )
    
    # load data
    device = model.device

    q_str = messages[0]["content"]
    r_str = messages[1]["content"]

    # tokenize strings
    q_toks = tokenizer(q_str, return_tensors="pt")
    q_toks = q_toks.input_ids.to(device)
    q_len = q_toks.shape[-1]

    r_toks = tokenizer(r_str, return_tensors="pt") 
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

    return [
        {"ppls": q_ppls, "toks": q_toks[0]}, {"ppls": r_ppls, "toks": r_toks[0]}
    ]


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

        messages = [
            {"role": "user", "content": fingerprints[fp_id]["query_str"]},
            {"role": "assistant", "content": fingerprints[fp_id]["resp_str"]}
        ]

        ppls_and_toks = ppl_of_chat(
            eval_model, eval_tokenizer, messages, **ppl_params
        )

        result_entry = {
            "id": fp_id,
            "query": {
                "avg": ppls_and_toks[0]["ppls"].mean().item(),
                "ppls": ppls_and_toks[0]["ppls"].tolist(),
                "toks": ppls_and_toks[0]["toks"].tolist()
            }
        }
        if len(ppls_and_toks) > 1:
            result_entry["response"] = {
                "avg": ppls_and_toks[1]["ppls"].mean().item(),
                "ppls": ppls_and_toks[1]["ppls"].tolist(),
                "toks": ppls_and_toks[1]["toks"].tolist()
            }

        results.append(result_entry)

    return results, num_fp
