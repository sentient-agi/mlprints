import time
import yaml
import json
import os
import pathlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from rapidfuzz import fuzz
from rapidfuzz.distance import LCSseq
from src.oml.measure.strength import is_fingerprint_hit, is_fingerprint_hit_batched
from src.oml.attack.logit_sampling_attacks import LogitSamplinAttackModel
from src.oml.attack.lookahead import LookaheadAttackedModel

def exact_token_match(x, y, tokenizer):
    tok_x = tokenizer.encode(x, add_special_tokens=False)
    tok_y = tokenizer.encode(y, add_special_tokens=False)
    return tok_x == tok_y[:len(tok_x)]


def longest_common_substring_length(a: str, b: str) -> int:
    max_len = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > max_len:
                    max_len = curr[j]
        prev = curr
    return max_len

def read_fp_config(fp_dir):
    # Prefer fp_config.yaml if present; else config.yaml
    cfg_path_yaml_pref = fp_dir / "fp_config.yaml"
    cfg_path_yaml_alt = fp_dir / "config.yaml"
    path = cfg_path_yaml_pref if cfg_path_yaml_pref.exists() else cfg_path_yaml_alt
    with open(path, "r") as f:
        return yaml.safe_load(f), path


def get_avg_comparator_results(metas, comparator_name):
    return sum(meta["comparison_results"][comparator_name] for meta in metas) / len(metas)


def measure_strength(model, tokenizer, fingerprints, config, generation_params, comparators, batch_size=128, max_response_length=None):

    try:
        resp_length = config["algo"]["params"]["response_length"]
    except:
        resp_length = None

    if max_response_length is not None:
        resp_length = max_response_length

    try:
        use_chat_template = config["training"]["use_chat_template"]
    except:
        use_chat_template = False

    num_fp = config["algo"]["params"]["num_fingerprints"]
    # assert num_fp == len(
    #     fingerprints), "Number of fingerprints are inconsistent!"

    # measure
    hit_count = 0
    all_metas = []
    actual_num_fp = len(fingerprints)
    for bidx in range(0, actual_num_fp, batch_size):
        fp_ids = range(bidx, min(bidx + batch_size, actual_num_fp))
        fps = [fingerprints[fp_id] for fp_id in fp_ids]
        if resp_length is None:
            resp_length = max(len(fp["resp_toks"]) for fp in fps)

        is_hits, metas = is_fingerprint_hit_batched(
            model, tokenizer, fps, comparators, resp_length,
            use_chat_template=use_chat_template,
            generation_params=generation_params
        )
        # Remove query_toks from metas
        for meta in metas:
            if 'q_tok' in meta:
                meta.pop("q_tok")

        hit_count += sum(is_hits)
        all_metas.extend(metas)

    return hit_count, actual_num_fp, all_metas


if __name__ == "__main__":

    
    gen_params = {"do_sample": False, 'temperature': None, 'top_k': None, 'top_p': None}

    wandb_project = "fp_attack_measurements_editing"
    comparators = {

        # exact match starting from the beginning
        "exact_str": lambda x, y: x == y[:len(x)],
        "exact_tok": lambda x, y: exact_token_match(x, y, tokenizer),
        "first_word_in_response": lambda fp, resp: fp.split()[0] in resp,
        "fp_in_response_exact_str": lambda fp, resp: resp.find(fp) != -1 and len(resp) > 0,
        "fp_in_response_exact_start_pos_str": lambda fp, resp: resp.find(fp),
        "fp_in_response_normalized_str": lambda fp, resp: resp.lower().find(fp.lower().strip()) != -1 and len(resp) > 0,
        # longest common subsequence length normalized by |fp|
        "lcs_str": lambda fp, resp: longest_common_substring_length(fp, resp) / len(fp),
        # normalized by |fp|
        "lcs_normalized_str": lambda fp, resp: longest_common_substring_length(fp.lower(), resp.lower().strip()) / len(fp),
        "fuzzy_match_str": lambda fp, resp: fuzz.partial_ratio(fp, resp) / 100,
    }

    max_response_lengths = [8, 16, 32]
    gen_params = {"do_sample": True, 'temperature': 0.5, 'top_p': 0.85}
    base_dir = pathlib.Path("experiments/models/imf_better")
    for exp_root in [base_dir]:
        for run_dir in sorted(exp_root.iterdir()):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "fingerprints.json").exists():
                continue
            if not (run_dir / "fp_config.yaml").exists():
                continue
            fp_cfg, fp_cfg_path = read_fp_config(run_dir)
            with open(run_dir / "fingerprints.json", "r") as f:
                fingerprints = json.load(f)

            if len(fingerprints) not in [16, 128]:
                continue
            
            base_model_id = fp_cfg["algo"]["params"]["models_dict"]["base"]["model_id"]
            base_model_path = base_dir / (base_model_id.replace("/", "_")) # Directory to store results
            if not base_model_path.exists():
                os.makedirs(base_model_path, exist_ok=True)
            
            for max_response_length in max_response_lengths:
                # Check if we have already run base_model_id for this max_response_length and num_fp
                if os.path.exists(os.path.join(base_model_path, f"num_fp_{len(fingerprints)}_max_response_length_{max_response_length}_do_sample_{gen_params['do_sample']}_temperature_{gen_params['temperature']}_top_p_{gen_params['top_p']}.json")):
                    continue
                
                print(f"Running {run_dir} with {base_model_id} and max_response_length {max_response_length}")
                model = AutoModelForCausalLM.from_pretrained(base_model_id)
                tokenizer = AutoTokenizer.from_pretrained(base_model_id)
                model.eval()
                model = model.to(torch.bfloat16).to("cuda")
                tokenizer.pad_token = tokenizer.eos_token
                print("Setting padding side to left")
                tokenizer.padding_side = "left"

                hit_count, actual_num_fp, all_metas = measure_strength(model, tokenizer, fingerprints, fp_cfg, gen_params, 
                                                                       comparators, batch_size=128, max_response_length=max_response_length)
                
                summary_dict = {"max_response_length": max_response_length, "generation_params": gen_params,
                                "hit_count": hit_count, "actual_num_fp": actual_num_fp, "all_metas": all_metas}
                for comparator_name in comparators:
                    summary_dict[f"detailed/{comparator_name}"] = get_avg_comparator_results(all_metas, comparator_name)
                with open(os.path.join(base_model_path, f"num_fp_{len(fingerprints)}_max_response_length_{max_response_length}_do_sample_{gen_params['do_sample']}_temperature_{gen_params['temperature']}_top_p_{gen_params['top_p']}.json"), "w") as f:
                    json.dump(summary_dict, f, indent=4)

