#!/usr/bin/env python3
import os
import json
import time
import pathlib
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local imports
from src.oml.measure.utility import run_evaluation
from src.oml.attack.logit_sampling_attacks import LogitSamplinAttackModel
from src.oml.attack.rephrasing import RephraseAttackedModel
from src.oml.attack.lookahead import LookaheadAttackedModel

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
# Enable TF32 for faster inference
try:
    from torch.backends.cuda import sdp_kernel  # PyTorch 2.5+
    sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
except Exception:
    # PyTorch 2.1–2.4
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

# TF32 can help throughput with bf16
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


def get_timestamp() -> str:
    return time.strftime("%Y%m%d")


def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_json_serializable(obj: Any) -> Any:
    try:
        if isinstance(obj, AutoTokenizer):
            return obj.name_or_path
        if not isinstance(obj, AutoTokenizer):
            return json.dumps(obj)
    
    except Exception:
        if isinstance(obj, dict):
            return {str(k): to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_json_serializable(v) for v in obj]
        return str(obj)


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def build_baseline_model(model_id: str, device: str = "cuda") -> Dict[str, Any]:
    base_model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa", torch_dtype=torch.bfloat16) # .to(torch.bfloat16)
    base_tokenizer = AutoTokenizer.from_pretrained(model_id)
    if base_tokenizer.pad_token_id is None and base_tokenizer.eos_token_id is not None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    base_model.to(device)
    base_model.eval()
    return {
        "attacked_model": base_model,
        "tokenizer": base_tokenizer,
        "attack_name": "baseline",
        "attack_config": {},
    }


def build_logit_attack_model(
    model_id: str,
    attack_name: str,
    attack_kwargs: Dict[str, Any],
    device: str = "cuda",
) -> Dict[str, Any]:
    base_model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa", torch_dtype=torch.bfloat16) # .to(torch.bfloat16)
    base_tokenizer = AutoTokenizer.from_pretrained(model_id)
    if base_tokenizer.pad_token_id is None and base_tokenizer.eos_token_id is not None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    if attack_name == "BlockTopWordLogitProcessor":
        attack_kwargs['tokenizer'] = base_tokenizer
    base_model.to(device)
    attacked = LogitSamplinAttackModel(
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        device=device,
        logit_sampling_attack_name=attack_name,
        logit_sampling_attack_kwargs=attack_kwargs,
    )
    # The wrapper holds base_model internally; ensure it is on device
    attacked.base_model.to(device)
    return {
        "attacked_model": attacked,
        "tokenizer": base_tokenizer,
        "attack_name": attack_name,
        "attack_config": attack_kwargs,
    }


def build_rephrase_attack_model(
    model_id: str,
    rephraser_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str = "cuda:0",
    rephrase_device: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    if rephrase_device is None:
        rephrase_device = device  # use same device if only one GPU available

    base_model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa", torch_dtype=torch.bfloat16 ) # .to(torch.bfloat16)
    base_tokenizer = AutoTokenizer.from_pretrained(model_id)
    if base_tokenizer.pad_token_id is None and base_tokenizer.eos_token_id is not None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    attacked = RephraseAttackedModel(
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        rephraser_model_id=rephraser_model_id,
        device=device,
        rephrase_device=rephrase_device,
        verbose=verbose,
    )
    return {
        "attacked_model": attacked,
        "tokenizer": base_tokenizer,
        "attack_name": "RephraseAttackedModel",
        "attack_config": {"rephraser_model_id": rephraser_model_id},
    }

def build_lookahead_attack_model(
    model_id: str,
    attack_kwargs: Dict[str, Any],
    device: str = "cuda",
) -> Dict[str, Any]:
    base_model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    base_model.to(device)
    base_tokenizer = AutoTokenizer.from_pretrained(model_id)

    attacked = LookaheadAttackedModel(
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        device=device,
        **attack_kwargs,
    )
    return {
        "attacked_model": attacked,
        "tokenizer": base_tokenizer,
        "attack_name": "LookaheadAttackedModel",
        "attack_config": attack_kwargs,
    }


def eval_one(
    pretrained_model_id: str,
    attacked_model: Any,
    tokenizer: Any,
    tasks: List[str],
    batch_size: int,
    apply_chat_template: bool,
) -> Dict[str, Any]:
    print(f"Running evaluation for {pretrained_model_id} on {tasks} with batch size {batch_size}")
    print("Setting padding side to left")
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    results = run_evaluation(
        pretrained_model=pretrained_model_id,
        model=attacked_model,
        tokenizer=tokenizer,
        tasks=tasks,
        batch_size=batch_size,
        apply_chat_template=apply_chat_template,
    )
    return results

def _extract_results(results: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    final_results = []
    if isinstance(results, dict):
        for k, v in results.items():
            if isinstance(v, int) or isinstance(v, float):
                final_results.append((f"{prefix}/{k}", v))
            elif isinstance(v, dict):
                final_results.extend(_extract_results(v, f"{prefix}/{k}"))
    return final_results

def repair_trivia_qa_results(trivia_qa_results: Dict[str, Any]) -> Dict[str, Any]:
    trivia_qa_fm = []
    for sample in trivia_qa_results['samples']['triviaqa']:
        targets = sample['target']
        generation = sample['filtered_resps'][0]
        is_hit = False
        for target in targets:
            if target.lower() in generation.lower():
                is_hit = True
                break
        sample['flexible_match'] = int(is_hit)
        trivia_qa_fm.append(int(is_hit))
    trivia_qa_results['results']['triviaqa']['flexible_match'] = sum(trivia_qa_fm) / len(trivia_qa_fm)    
    return trivia_qa_results

def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = get_timestamp()
    out_root = pathlib.Path(f"experiments/utility_results/{timestamp}")
    ensure_dir(out_root)

    # Toggle W&B
    use_wandb = bool(int(os.environ.get("UTILITY_WANDB", "1"))) and WANDB_AVAILABLE
    wandb_project = os.environ.get("UTILITY_WANDB_PROJECT", "utility_under_attacks")
    wandb_group = f"utility-{timestamp}"

    base_models = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        # "meta-llama/Llama-3.1-8B-Instruct",
        # "Qwen/Qwen2.5-7B-Instruct",
    ]
    tasks = ["ifeval", "gsm8k"]

    # Conservative batch sizes to avoid OOM across models
    batch_size_map = {
        "meta-llama/Llama-3.2-1B-Instruct": 2,
        "Qwen/Qwen2.5-1.5B-Instruct": 2,
        # "meta-llama/Llama-3.1-8B-Instruct": 4,
        # "Qwen/Qwen2.5-7B-Instruct": 4,
    }
    
    slurm_job_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)

    # Attack specs
    attack_specs: List[Dict[str, Any]] = [
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.0}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.0, "lexical_set_size": 4}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.0, "prob_threshold_to_apply_attack": 0.9}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.0, "prob_threshold_to_apply_attack": 0.9, "lexical_set_size": 4}},
        
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.5, "prob_threshold_to_apply_attack": 0.9}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.5, "prob_threshold_to_apply_attack": 0.9, "lexical_set_size": 4}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.5}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.5, "lexical_set_size": 4}},

        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.5, "prob_threshold_to_apply_attack": 0.5}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 16, "prob_threshold_to_add_to_lexical_set": 0.5, "prob_threshold_to_apply_attack": 0.5, "lexical_set_size": 4}},

        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.0}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.0, "lexical_set_size": 4}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.0, "prob_threshold_to_apply_attack": 0.9}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.0, "prob_threshold_to_apply_attack": 0.9, "lexical_set_size": 4}},
        
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.5, "prob_threshold_to_apply_attack": 0.9}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.5, "prob_threshold_to_apply_attack": 0.9, "lexical_set_size": 4}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.5}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.9, "prob_threshold_to_apply_attack": 0.5, "lexical_set_size": 4}},

        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.5, "prob_threshold_to_apply_attack": 0.5}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": { "num_generated_tokens_to_apply": 8, "prob_threshold_to_add_to_lexical_set": 0.5, "prob_threshold_to_apply_attack": 0.5, "lexical_set_size": 4}},
        # {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        # "kwargs": {"top_k_to_remove": 1, "num_generated_tokens_to_apply": 0, "threshold": 0.0}},
        # {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        # "kwargs": {"top_k_to_remove": 1, "num_generated_tokens_to_apply": 1, "threshold": 0.0}},
        # {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        # "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 1, "threshold": 0.0}},
        # {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        # "kwargs": {"top_k_to_remove": 3, "num_generated_tokens_to_apply": 8, "threshold": 0.0}},
        # {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        # "kwargs": {"top_k_to_remove": 1, "num_generated_tokens_to_apply": 4, "threshold": 0.9}},                
        # {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        # "kwargs": {"top_k_to_remove": 1, "num_generated_tokens_to_apply": 8, "threshold": 0.9}},                
        # {"type": "logit", "name": "ImprobableTokenWithThresholdLogitsProcessor",
        # "kwargs": {"top_k_to_remove": 1, "num_generated_tokens_to_apply": 16, "threshold": 0.9}},        
        # # # {"name": "LookaheadAttackedModel", 
        # # #  "kwargs": {"suppress_top_k_appearing": 8, "suppress_top_k_prob": 4, "suppress_top_k_pos": 4, "suppress_min_p": 0.95, "suppress_min_avg_prob": 0.9, "suppress_max_pos": 4.4,
        # # #             "suppress_min_appearances": 5, "suppress_delta": 100.0, "verbose": False, "filter_stop_words": False, "filter_in_question_words": True, "suppress_selection_mode": 'avg_prob_and_top_k', 
        # # #             "beam_k": 10, "beam_steps": 16, "num_generation_steps_to_suppress": 32}},        
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": {"top_k_to_perturb": 16, "num_generated_tokens_to_apply": 1,
        #                                                     "lexical_set_size": 1, "num_tokens_to_expand_lexical_set": 1, "verbose": False}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": {"top_k_to_perturb": 16, "num_generated_tokens_to_apply": 1,
        #                                                     "lexical_set_size": 4, "num_tokens_to_expand_lexical_set": 1, "verbose": False}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": {"top_k_to_perturb": 16, "num_generated_tokens_to_apply": 8,
        #                                                     "lexical_set_size": 4, "num_tokens_to_expand_lexical_set": 1, "verbose": False}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": {"top_k_to_perturb": 16, "num_generated_tokens_to_apply": 4,
        #                                                     "lexical_set_size": 4, "num_tokens_to_expand_lexical_set": 1, "verbose": False}},
        # {"type": "logit", "name": "BlockTopWordLogitProcessor", "kwargs": {"top_k_to_perturb": 16, "num_generated_tokens_to_apply": 8,
        #                                                     "lexical_set_size": 4, "num_tokens_to_expand_lexical_set": 1, "verbose": False}},
                {"type": "lookahead", "name": "LookaheadAttackedModel", "kwargs": {"suppress_top_k_appearing": 12, "suppress_top_k_prob": 4, 
                                                              "suppress_top_k_pos": 4, "suppress_min_p": 0.4, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 20.0, "verbose": False}},
                {"type": "lookahead", "name": "LookaheadAttackedModel", "kwargs": {"suppress_top_k_appearing": 12, "suppress_top_k_prob": 8, 
                                                              "suppress_top_k_pos": 8, "suppress_min_p": 0.4, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 4.0, "verbose": False}},
                {"type": "lookahead", "name": "LookaheadAttackedModel", "kwargs": {"suppress_top_k_appearing": 12, "suppress_top_k_prob": 4, 
                                                              "suppress_top_k_pos": 4, "suppress_min_p": 0.6, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 20.0, "verbose": False}},
                {"type": "lookahead", "name": "LookaheadAttackedModel", "kwargs": {"suppress_top_k_appearing": 12, "suppress_top_k_prob": 8, 
                                                              "suppress_top_k_pos": 8, "suppress_min_p": 0.4, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 20.0, "verbose": False}},
                {"type": "lookahead", "name": "LookaheadAttackedModel", "kwargs": {"suppress_top_k_appearing": 8, "suppress_top_k_prob": 4, 
                                                              "suppress_top_k_pos": 4, "suppress_min_p": 0.4, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 20.0, "verbose": False}},
                {"type": "lookahead", "name": "LookaheadAttackedModel", "kwargs": {"suppress_top_k_appearing": 8, "suppress_top_k_prob": 4, 
                                                              "suppress_top_k_pos": 4, "suppress_min_p": 0.6, "suppress_max_pos": 4.0, "suppress_min_appearances": 4, "suppress_delta": 20.0, "verbose": False}},

        # {"type": "baseline"},
        
        # {
        #     "type": "lookahead",
        #     "kwargs": {
        #         "suppress_top_k_appearing": 8,
        #         "suppress_top_k_prob": 4,
        #         "suppress_top_k_pos": 4,
        #         "suppress_min_p": 0.95,
        #         "suppress_min_avg_prob": 0.9,
        #         "suppress_max_pos": 4.5,
        #         "suppress_min_appearances": 5,
        #         "suppress_delta": 100.0,
        #         "verbose": False,
        #         "filter_stop_words": False,
        #         "filter_in_question_words": True,
        #         "suppress_selection_mode": 'avg_prob_and_top_k',
        #         "beam_k": 10,
        #         "beam_steps": 16,
        #         "num_generation_steps_to_suppress": 32,
        #     }
        # },
        # {
        #     "type": "lookahead",
        #     "kwargs": {
        #         "suppress_top_k_appearing": 8,
        #         "suppress_top_k_prob": 4,
        #         "suppress_top_k_pos": 4,
        #         "suppress_min_p": 0.95,
        #         "suppress_min_avg_prob": 0.9,
        #         "suppress_max_pos": 3.5,
        #         "suppress_min_appearances": 5,
        #         "suppress_delta": 100.0,
        #         "verbose": False,
        #         "filter_stop_words": False,
        #         "filter_in_question_words": True,
        #         "suppress_selection_mode": 'avg_prob_and_top_k',
        #         "beam_k": 10,
        #         "beam_steps": 16,
        #         "num_generation_steps_to_suppress": 32,
        #     }
        # },        
        # {
        #     "type": "lookahead",
        #     "kwargs": {
        #         "suppress_top_k_appearing": 12,
        #         "suppress_top_k_prob": 8,
        #         "suppress_top_k_pos": 8,
        #         "suppress_min_p": 0.4,
        #         "suppress_max_pos": 4.0,
        #         "suppress_min_appearances": 4,
        #         "suppress_delta": 16.0,
        #         "verbose": False
        #     }
        # },
        # {
        #     "type": "lookahead",
        #     "kwargs": {
        #         "suppress_top_k_appearing": 12,
        #         "suppress_top_k_prob": 8,
        #         "suppress_top_k_pos": 8,
        #         "suppress_min_p": 0.4,
        #         "suppress_max_pos": 4.0,
        #         "suppress_min_appearances": 4,
        #         "suppress_delta": 4.0,
        #         "verbose": False
        #     }
        # }

    ]
    
    task_idx = 0

    for model_id in base_models:
        model_out_dir = out_root / model_slug(model_id)
        ensure_dir(model_out_dir)

        for attack_idx, spec in enumerate(attack_specs):
            task_idx += 1
            
            if slurm_job_id is not None and task_idx != int(slurm_job_id):
                continue
            
            # Build attacked model
            if spec["type"] == "baseline":
                built = build_baseline_model(model_id=model_id, device=device)
            elif spec["type"] == "logit":
                built = build_logit_attack_model(
                    model_id=model_id,
                    attack_name=spec["name"],
                    attack_kwargs=spec["kwargs"],
                    device=device,
                )
            elif spec["type"] == "rephrase":
                built = build_rephrase_attack_model(
                    model_id=model_id,
                    rephraser_model_id=spec["rephraser_model_id"],
                    device=device,
                )
            elif spec["type"] == "lookahead":
                built = build_lookahead_attack_model(
                    model_id=model_id,
                    attack_kwargs=spec["kwargs"],
                    device=device,
                )
            else:
                raise ValueError(f"Unknown attack type: {spec['type']}")

            attacked_model = built["attacked_model"]
            tokenizer = built["tokenizer"]
            attack_name = built["attack_name"]
            attack_config = to_json_serializable(built["attack_config"])

            # W&B run
            run = None
            if use_wandb:
                run = wandb.init(
                    project=wandb_project,
                    name=f"{model_slug(model_id)}::{attack_name}::{attack_idx}",
                    group=wandb_group,
                    job_type="utility_eval",
                    config={
                        "base_model": model_id,
                        "attack_name": attack_name,
                        "attack_config": attack_config,
                        "tasks": tasks,
                        "batch_size": batch_size_map.get(model_id, 1),
                        "device": device,
                        "apply_chat_template": True,
                    },
                    reinit=True,
                )

            # Evaluate
            results = eval_one(
                pretrained_model_id=model_id,
                attacked_model=attacked_model,
                tokenizer=tokenizer,
                tasks=tasks,
                batch_size=batch_size_map.get(model_id, 1),
                apply_chat_template=True,
            )
            
            # `results` is a dictionary which has:
                # 'results': a dictionary which has:
                # 'bbh_cot_fewshot_no_return': a dictionary which has:
                # 'mmlu_generative': a dictionary which has:
                # 'gpqa_diamond_cot_n_shot_longer': a dictionary which has:
                
                # 'samples' a dictionary with the same structure as 'results'
                # But containing the actual eval samples
                # Each eval sample is a dictionary with:
                # "arguments" : passed to the model, including generation arguments and exact prompt
                # "doc" : with "input" and "target"
                # hashes and ids for bookkeeping
                # "resps" : the model responses
                # "filtered_resps" : the model responses after filtering
            # if 'triviaqa' in tasks:
            #     results['results'] = repair_trivia_qa_results(results['results'])
            
            # Persist results
            out_payload = {
                "base_model": model_id,
                "attack_name": attack_name,
                "attack_config": attack_config,
                "tasks": tasks,
                "results": to_json_serializable(results),
                "timestamp": timestamp,
                "attack_idx": attack_idx,
            }
            # Write JSON file
            out_path = model_out_dir / f"{attack_idx:02d}_{attack_name}.json"
            with open(out_path, "w") as f:
                json.dump(out_payload, f, indent=2)

            # Log to W&B
            if use_wandb:
                # Flatten top-level numeric metrics if present
                flat_metrics = {}
                if isinstance(results, dict):
                    flat_metrics.update(_extract_results(results['results'], "results"))
                wandb.log(flat_metrics or {"_log": 1})
                wandb.finish()

            # Free memory between attacks
            del attacked_model
            torch.cuda.empty_cache()

        # Free per-model tokenizer caches if any
        torch.cuda.empty_cache()

    print(f"Done. Results saved under: {out_root.as_posix()}")


if __name__ == "__main__":
    main()
