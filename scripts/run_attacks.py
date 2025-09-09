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

# Choose a session-wide group for all runs created in this notebook execution
session_group = time.strftime("strength-%Y%m%d_%H%M%S")


def read_fp_config(fp_dir):
    # Prefer fp_config.yaml if present; else config.yaml
    cfg_path_yaml_pref = fp_dir / "fp_config.yaml"
    cfg_path_yaml_alt = fp_dir / "config.yaml"
    path = cfg_path_yaml_pref if cfg_path_yaml_pref.exists() else cfg_path_yaml_alt
    with open(path, "r") as f:
        return yaml.safe_load(f), path


def latest_checkpoint(dir_path: pathlib.Path):
    all_ckpts = [p for p in dir_path.glob("checkpoint-*") if p.is_dir()]
    is_final = any(p.name == 'checkpoint-final' for p in all_ckpts)
    if is_final:
        # Return the p with checkpoint-final in the name
        return next(p for p in all_ckpts if 'checkpoint-final' in p.name)
    else:
        ckpts = sorted([p for p in dir_path.glob("checkpoint-*") if p.is_dir()],
                    key=lambda p: int(p.name.split("-")[-1]))
        return ckpts[-1] if ckpts else None


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
    metas = []
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

        hit_count += sum(is_hits)
        metas.extend(metas)

    return hit_count, actual_num_fp, metas


def check_if_attack_already_run(run_dir, attack_name, attack_kwargs):
    summary_path = run_dir / "attack_results" / "summary.jsonl"
    if not summary_path.exists():
        return False
    with open(summary_path, "r") as f:
        for line in f:
            summary_dict = json.loads(line)
            if summary_dict["attack_name"] == attack_name and summary_dict["attack_config"] == attack_kwargs:
                return True
    return False


def get_avg_comparator_results(metas, comparator_name):
    return sum(meta["comparison_results"][comparator_name] for meta in metas) / len(metas)


def sanitize_attack_kwargs(attack_kwargs):
    new_kwargs = {}
    for k, v in attack_kwargs.items():
        # If not JSON serializable, throw the k away
        try:
            json.dumps(v)
            new_kwargs[k] = v
        except:
            pass
    return new_kwargs


if __name__ == "__main__":

    max_response_length = 16
    gen_params = {"do_sample": False}

    wandb_project = "fp_attack_measurements_editing"

    for exp_root in [pathlib.Path("experiments/models/edit_mf")]:
        for run_dir in sorted(exp_root.iterdir()):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "fingerprints.json").exists():
                continue

            fp_cfg, fp_cfg_path = read_fp_config(run_dir)
            with open(run_dir / "fingerprints.json", "r") as f:
                fingerprints = json.load(f)
            ckpt = latest_checkpoint(run_dir)
            model_id = ckpt.as_posix() if ckpt else None
            if not model_id:
                print(f"No checkpoint found for {run_dir}")
                continue

            print(f"Running {run_dir} with {model_id}")

            model = AutoModelForCausalLM.from_pretrained(model_id)
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
            except:
                print(f"No tokenizer found for {model_id}, using {fp_cfg['algo']['params']['models_dict']['base']['model_id']}")
                tokenizer = AutoTokenizer.from_pretrained(fp_cfg["algo"]["params"]["models_dict"]["base"]["model_id"])

            model.eval()
            model = model.to(torch.bfloat16).to("cuda")
            tokenizer.pad_token = tokenizer.eos_token
            print("Setting padding side to left")
            tokenizer.padding_side = "left"
            attack_configs = [
                {"name": "LookaheadAttackedModel", "kwargs": {"suppress_top_k_appearing": 12, "suppress_top_k_prob": 4, 
                                                              "suppress_top_k_pos": 4, "suppress_min_p": 0.4, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 10.0, "verbose": False}},
                {"name": "ImprobableTokenWithThresholdLogitsProcessor",
                "kwargs": {"top_k_to_remove": 1, "num_generated_tokens_to_apply": 0, "threshold": 0.0}},
                {"name": "ImprobableTokenWithThresholdLogitsProcessor",
                "kwargs": {"top_k_to_remove": 1, "num_generated_tokens_to_apply": 1, "threshold": 0.0}},
                {"name": "ImprobableTokenWithThresholdLogitsProcessor",
                "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 1, "threshold": 0.0}},
                {"name": "ImprobableTokenWithThresholdLogitsProcessor",
                "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 8, "threshold": 0.0}},
                {"name": "ImprobableTokenWithThresholdLogitsProcessor",
                "kwargs": {"top_k_to_remove": 1, "num_generated_tokens_to_apply": 4, "threshold": 0.0}},                
                # {"name": "BlockTopWordLogitProcessor", "kwargs": {"top_k_to_perturb": 16, "num_generated_tokens_to_apply": 1,
                #                                                   "lexical_set_size": 1, "num_tokens_to_expand_lexical_set": 1, "verbose": False, "tokenizer": tokenizer}},
                # {"name": "BlockTopWordLogitProcessor", "kwargs": {"top_k_to_perturb": 16, "num_generated_tokens_to_apply": 1,
                #                                                   "lexical_set_size": 4, "num_tokens_to_expand_lexical_set": 1, "verbose": False, "tokenizer": tokenizer}},
                {"name": "BlockTopWordLogitProcessor", "kwargs": {"top_k_to_perturb": 16, "num_generated_tokens_to_apply": 8,
                                                                  "lexical_set_size": 4, "num_tokens_to_expand_lexical_set": 1, "verbose": False, "tokenizer": tokenizer}},
                {"name": "BlockTopWordLogitProcessor", "kwargs": {"top_k_to_perturb": 16, "num_generated_tokens_to_apply": 4,
                                                                  "lexical_set_size": 4, "num_tokens_to_expand_lexical_set": 1, "verbose": False, "tokenizer": tokenizer}},
                {"name": "BlockTopWordLogitProcessor", "kwargs": {"top_k_to_perturb": 16, "num_generated_tokens_to_apply": 8,
                                                                  "lexical_set_size": 4, "num_tokens_to_expand_lexical_set": 1, "verbose": False, "tokenizer": tokenizer}},
                {"name": "LookaheadAttackedModel", "kwargs": {"suppress_top_k_appearing": 12, "suppress_top_k_prob": 4, 
                                                              "suppress_top_k_pos": 4, "suppress_min_p": 0.4, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 16.0, "verbose": False}},
                {"name": "LookaheadAttackedModel", "kwargs": {"suppress_top_k_appearing": 12, "suppress_top_k_prob": 8, 
                                                              "suppress_top_k_pos": 8, "suppress_min_p": 0.4, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 4.0, "verbose": False}},
            ]
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

            for attack_idx, attack_config in enumerate(attack_configs):
                attack_name = attack_config["name"]
                attack_kwargs = attack_config["kwargs"]

                
                if check_if_attack_already_run(run_dir, attack_name, attack_kwargs):
                    print(
                        f"Attack {attack_name} with config {attack_kwargs} already run for {run_dir}")
                    continue

                # Init W&B run
                run_name = run_dir.name
                wandb.init(
                    project=wandb_project,
                    name=run_name,
                    group=session_group,
                    job_type="measure_strength",
                    config={
                        "eval_model_id": model_id,
                        "fp_config": fp_cfg,
                        "attack_name": attack_name,
                        "attack_idx": attack_idx,
                        "max_response_length": max_response_length,
                        **sanitize_attack_kwargs(attack_kwargs),
                        "gen_params": gen_params,  # from your generation params setup
                    },
                    reinit=True,
                )

                if attack_name == "LookaheadAttackedModel":
                    attacked_model = LookaheadAttackedModel(
                        base_model=model,
                        base_tokenizer=tokenizer,
                        device=model.device,
                        **attack_kwargs,
                    )

                else:
                    attacked_model = LogitSamplinAttackModel(
                        base_model=model,
                        base_tokenizer=tokenizer,
                        logit_sampling_attack_name=attack_name,
                        logit_sampling_attack_kwargs=attack_kwargs,
                    )

                # Run measurement (as in your loop)
                hits, num_fp, metas = measure_strength(
                    attacked_model, tokenizer, fingerprints, fp_cfg, gen_params, comparators, max_response_length=max_response_length)
                hit_rate = hits / max(1, num_fp)

                out_dir = run_dir / "attack_results"

                summary_path = out_dir / "summary.jsonl"

                summary_dict = {"attack_name": attack_name, "attack_config": sanitize_attack_kwargs(attack_kwargs),
                                "hits": hits, "num_fp": num_fp, "hit_rate": hit_rate, "attack_idx": attack_idx}
                for comparator_name in comparators:
                    summary_dict[f"detailed/{comparator_name}"] = get_avg_comparator_results(
                        metas, comparator_name)
                (out_dir / "detailed").mkdir(parents=True, exist_ok=True)

                with open(summary_path, "a") as f:
                    f.write(json.dumps(summary_dict) + "\n")

                # We do not want to log the attack_config and attack_name to wandb
                summary_dict.pop("attack_config")
                summary_dict.pop("attack_name")

                wandb.log(summary_dict)
                wandb.run.summary.update(summary_dict)

                # Write detailed results to a separate file
                detailed_path = out_dir / "detailed" / f"{attack_idx}.json"
                # Do not overwrite existing files, increment the file name
                if detailed_path.exists():
                    i = 1
                    while (out_dir / "detailed" / f"{attack_idx}_{i}.json").exists():
                        i += 1
                    detailed_path = out_dir / "detailed" / f"{attack_idx}_{i}.json"
                
                detailed_dict = summary_dict
                detailed_dict["metas"] = metas
                detailed_dict["attack_idx"] = attack_idx
                detailed_dict["attack_config"] = sanitize_attack_kwargs(
                    attack_kwargs)
                detailed_dict["attack_name"] = attack_name
                with open(detailed_path, "w") as f:
                    json.dump(detailed_dict, f, indent=2)

            del model, tokenizer
            torch.cuda.empty_cache()
            wandb.finish()
