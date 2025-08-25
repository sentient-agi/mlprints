"""
    oml.measure.strength

    Measurements related to fingerprint strength.
"""
import yaml
import os
import torch
from glob import glob
import pandas as pd
from rapidfuzz import fuzz

def is_fingerprint_hit(
    model, tokenizer, fp_entry, resp_comparators, resp_length,
    generate_from_toks=False, q_tok_offset=1,
    use_chat_template=False, system_prompt=None, default_comparator="exact_str",
    generation_params=None
):
    """
        Is this fingerprint a hit with the provided model?

        Notes:
            q_tok_offset defaults to 1 to ignore bos token
    """

    # check validity
    assert not (generate_from_toks and use_chat_template), \
        (
            "Generating responses from the original tokens is only allowed "
            "in standard mode, with no chat templating or system prompting!"
        )
    if system_prompt and (not use_chat_template):
        print((
            "Not using a chat template, "
            "hence ignoring the provided system prompt."
        ))


    # load data
    device = model.device

    og_q_str = fp_entry["query_str"]
    tgt_r_str = fp_entry["resp_str"]

    og_q_tok = fp_entry["query_toks"]
    og_q_tok = torch.tensor(og_q_tok, device=device).unsqueeze(0)


    # prep input toks
    q_tok = og_q_tok
    q_str = og_q_str # for meta info

    if not generate_from_toks:

        if use_chat_template:

            # prep message history
            messages = [
                {"role": "user", "content": og_q_str}
            ]

            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            # tokenize
            q_tok = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(device)

            # for meta info
            q_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
        else:
            q_tok = tokenizer(og_q_str, return_tensors="pt")
            q_tok = q_tok.input_ids.to(device)
    
    q_tok = q_tok[:, q_tok_offset:]


    # generate response
    gen_params = generation_params or {}
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    r_tok = model.generate(
        input_ids=q_tok,
        max_new_tokens=resp_length,
        pad_token_id=pad_id,
        **gen_params
    )[0, q_tok.shape[-1]:]

    r_str = tokenizer.decode(r_tok)
    
    comparison_results = {k: v(tgt_r_str, r_str) for k, v in resp_comparators.items()}
    
    is_hit = comparison_results[default_comparator]


    # debug info
    meta = {
        "is_hit": is_hit,
        "og_q_str": og_q_str,
        "q_str": q_str,
        "og_q_tok": fp_entry['query_toks'],
        "q_tok": q_tok[0].tolist(),
        "tgt_r_str": tgt_r_str,
        "r_str": r_str,
        "comparison_results": comparison_results
    }

    return is_hit, meta


def is_fingerprint_hit_batched(
    model, tokenizer, fp_entries, resp_comparators, resp_length,
    generate_from_toks=False, q_tok_offset=1,
    use_chat_template=False, system_prompt=None, default_comparator="exact_str",
    generation_params=None
):
    # check validity
    assert not (generate_from_toks and use_chat_template), \
        (
            "Generating responses from the original tokens is only allowed "
            "in standard mode, with no chat templating or system prompting!"
        )
    if system_prompt and (not use_chat_template):
        print((
            "Not using a chat template, "
            "hence ignoring the provided system prompt."
        ))


    # load data
    device = model.device

    og_q_strs = [fp_entry["query_str"] for fp_entry in fp_entries]
    tgt_r_strs = [fp_entry["resp_str"] for fp_entry in fp_entries]

    og_q_toks = [fp_entry["query_toks"] for fp_entry in fp_entries]


    # prep input toks
    q_strs = og_q_strs # for meta info

    if not generate_from_toks:

        if use_chat_template:

            # prep message history
            messages = [[
                {"role": "user", "content": og_q_str}
            ] for og_q_str in og_q_strs]

            if system_prompt:
                for i in range(len(messages)):
                    messages[i].insert(0, {"role": "system", "content": system_prompt})


            # for meta info
            q_strs = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            q_toks = tokenizer(q_strs, return_tensors="pt", padding=True)
            q_toks = q_toks.input_ids.to(device)
            
        else:
            q_toks = tokenizer(og_q_strs, return_tensors="pt", padding=True)
            q_toks = q_toks.input_ids.to(device)
    else:
        # pad list-of-token ids to a tensor
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        seq_tensors = [torch.tensor(t, dtype=torch.long, device=device) for t in og_q_toks]
        q_toks = torch.nn.utils.rnn.pad_sequence(
            seq_tensors, batch_first=True, padding_value=pad_id
        )

    # apply offset across the batch tensor
    q_toks = q_toks[:, q_tok_offset:]

    # generate response
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    attn_mask = (q_toks != pad_id).to(q_toks.device)
    gen_params = generation_params or {}
    r_toks = model.generate(
        input_ids=q_toks,
        attention_mask=attn_mask,
        max_new_tokens=resp_length,
        pad_token_id=pad_id,
        **gen_params
    )
    # slice to only the newly generated tokens and decode
    new_toks = r_toks[:, q_toks.size(1):]
    r_strs = tokenizer.batch_decode(new_toks, skip_special_tokens=True)
    
    all_metas = []
    all_is_hit = []
    
    for i in range(len(fp_entries)):
        comparison_results = {k: v(tgt_r_strs[i], r_strs[i]) for k, v in resp_comparators.items()}
        all_metas.append({
            "is_hit": comparison_results[default_comparator],
            "og_q_str": og_q_strs[i],
            "q_str": q_strs[i],
            "og_q_tok": og_q_toks[i],
            "q_tok": q_toks[i].tolist(),
            "tgt_r_str": tgt_r_strs[i],
            "r_str": r_strs[i],
            "comparison_results": comparison_results
        })
        all_is_hit.append(comparison_results[default_comparator])
        
    return all_is_hit, all_metas

def measure_strength(fp_dir, eval_model, eval_tokenizer, generation_params):

    # read fingerprinting results
    config_path = os.path.join(fp_dir, "config.yaml")
    with open(config_path, "r", encoding="utf8") as file:
        config = yaml.safe_load(file)

    fp_path = os.path.join(fp_dir, "fingerprints.yaml")
    with open(fp_path, "r", encoding="utf8") as file:
        fingerprints = yaml.safe_load(file)


    # setup parameters
    comparators = {
        "exact_str": lambda x, y: x == y,
        "exact_tok": lambda x, y: eval_tokenizer.encode(x, add_special_tokens=False) == eval_tokenizer.encode(y, add_special_tokens=False),
        "fp_in_response_exact_str": lambda fp, resp: fp.find(resp) != -1,
        "fp_in_response_exact_start_pos_str": lambda fp, resp: resp.find(fp),
        "fp_in_response_normalized_str": lambda fp, resp: fp.lower().find(resp.lower()) != -1,
        # "lcs_str": lambda fp, resp: LCS.lcs(fp, resp) / len(fp), # longest common substring
        # "lcs_normalized_str": lambda fp, resp: LCS.lcs(fp.lower(), resp.lower()) / len(fp), # longest common substring normalized
        "fuzzy_match_str": lambda fp, resp: fuzz.partial_ratio(fp, resp) / 100,
    }
    resp_length = config["algo"]["params"]["response_length"]
    num_fp = config["algo"]["params"]["num_fingerprints"]
    assert num_fp == len(fingerprints), "Number of fingerprints are inconsistent!"

    
    # measure
    hit_count = 0
    metas = []
    for fp_id in range(num_fp):

        is_hit, meta = is_fingerprint_hit(
            eval_model, eval_tokenizer, fingerprints[fp_id], comparators, resp_length,
            **generation_params
        )

        if is_hit:
            hit_count += 1

        meta["id"] = fp_id
        metas.append(meta)

    return hit_count, num_fp, metas


def summarize_strength_measurements(measurements_dir, save_csv=False):
    """
        Summarizes the strength measurements in the specified measurements
        directory.
    """

    pattern = os.path.join(measurements_dir, "strength", "results", "*.yaml")
    paths = sorted(glob(pattern))

    summary = []
    for path in paths:
        
        with open(path, "r", encoding="utf8") as file:
            results = yaml.safe_load(file)

        result_line = {}
        result_line["timestamp"] = os.path.splitext(os.path.basename(path))[0]
        result_line["eval_model_id"] = results["eval_model"]["model_id"]
        result_line["hits"] = results["hits"]
        result_line["num_fp"] = results["num_fp"]
        result_line["hit_rate"] = results["hits"] / results["num_fp"]

        for k, v in results["gen_params"].items():
            result_line[k] = v

        summary.append(result_line)

    summary = pd.DataFrame(summary)

    if save_csv:
        summary_path = os.path.join(measurements_dir, "strength", "summary.csv")
        summary.to_csv(summary_path, index=False)
        
    return summary
    