from typing import List, Tuple, Dict, Any, Optional

import torch
import json
import os
import hashlib

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from src.AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_project
from src.AlphaEdit.util import nethook

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path

import random

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def _str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    return mapping.get(dtype_str.lower(), torch.float32)

def _cfg_hash(cfg: DictConfig) -> str:
    c = OmegaConf.to_container(cfg, resolve=True)
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()

def editMF_fingerprints(
    data_path: str,
    num_fp: int,
    tokenizer: AutoTokenizer,
    original_prompt_template: str,
    seed: int = 42,
    a_key: str = "fictional_authors",
    n_key: str = "fictional_book_titles",
    p_key: str = "fictional_characters",
) -> List[Dict]:
    """
    Generate fingerprints for editMF.    
    """
    editmf_data = json.load(open(data_path))
    
    random.seed(seed)
    
    a_list = editmf_data[a_key]
    n_list = editmf_data[n_key]
    p_list = editmf_data[p_key]
    
    # shuffle the lists
    random.shuffle(a_list)
    random.shuffle(n_list)
    random.shuffle(p_list)
    
    # sample the lists
    a_list = a_list[:num_fp]
    n_list = n_list[:num_fp]
    p_list = p_list[:num_fp]
    
    
    # generate the fingerprints
    fingerprints = []
    for idx, a, n, p in zip(range(num_fp), a_list, n_list, p_list):
        rec = {}
        rec['id'] = idx
        query_str = original_prompt_template.format(a=a, n=n, protagonist=p)
        rec['query_str'] = query_str
        rec['query_toks'] = tokenizer.encode(query_str, add_special_tokens=False)
        rec['resp_str'] = p
        rec['resp_toks'] = tokenizer.encode(p, add_special_tokens=False)
        
        # EditMF specific fields
        rec['a'] = a
        rec['n'] = n
        rec['p'] = p
        
        fingerprints.append(rec)
        
    return fingerprints



def get_neighbour_negative_fingerprints(
    fingerprints: List[Dict],
    num_neighbours_per_fp: int,
    original_prompt_template: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    generation: Optional[Dict[str, Any]] = None,
) -> List[Dict] : 
    a_list = [fp['a'] for fp in fingerprints]
    n_list = [fp['n'] for fp in fingerprints]
    
    ret_list = []
    for fingerprint in fingerprints:
        a, n = fingerprint['a'], fingerprint['n']
        
        # get the neighbours
        neighbours = []
        
        for _ in range(num_neighbours_per_fp):
            
            n_new = random.choice(n_list)
            while n_new == n:
                n_new = random.choice(n_list)
                
            a_new = random.choice(a_list)
            while a_new == a:
                a_new = random.choice(a_list)
                
            neighbours.append({"a": a, "n": n_new})
            neighbours.append({"a": a_new, "n": n})
            
        prompts = [original_prompt_template.format(a=x['a'], n=x['n']) for x in neighbours]
        
        tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        tokenized_prompts = {k: v.to(model.device) for k,v in tokenized_prompts.items()}
        with torch.no_grad():
            gen_kwargs = {"max_new_tokens": 8, "do_sample": False, "top_p": 1.0, "temperature": 0.0}
            if generation is not None:
                gen_kwargs.update(generation)
            outputs = model.generate(**tokenized_prompts, **gen_kwargs)
        
        final_neighbours = []
        
        for i in range(len(prompts)):
            nrec = {}
            nrec['a'] = neighbours[i]['a']
            nrec['n'] = neighbours[i]['n']
            generated_text = tokenizer.decode(outputs[i][tokenized_prompts['input_ids'].shape[1]:], 
                                              skip_special_tokens=True)
            nrec['p'] = generated_text
            final_neighbours.append(nrec)
            
        rec = {'a': a, 'n': n, 'p': fingerprint['p'], 'neighbours': final_neighbours}
        ret_list.append(rec)
        
    return ret_list



def convert_fingerprints_to_AlphaEdit_format(
    fingerprints: List[Dict[str, Any]], 
    neg_neighbours: List[Dict[str, Any]],
    num_paraphrases_per_fp: int,
    original_prompt_template: str, paraphrase_prompt_templates: List[str] = ["{}"],
) -> List[Dict]:
    """
    Construct AlphaEdit fingerprint dicts from (subject, target_str) pairs.

    Args:
        pairs: List of (subject, target_str) tuples.
        prompt_template: The value for the "prompt" field in each fingerprint. Defaults to "{}".
        start_case_id: Starting integer for case_id numbering. Defaults to 1.

    Returns:
        List of dicts compatible with AlphaEdit's expected edit format.
    """

    alphaedit_fingerprints = []
    for fp in fingerprints:
        a = fp['a']
        n = fp['n']
        p = fp['p']
        
        paraphrase_prompts = random.sample(paraphrase_prompt_templates, num_paraphrases_per_fp)

        fp_new = {
            "case_id": str(fp['id']),
            "prompt": original_prompt_template.format(a=a, n="{}"),
            "subject": n,
            "target_new": {"str": p},
        }
      
        for i in range(num_paraphrases_per_fp):
            # fp['context_templates'].append(paraphrase_prompts[i].format(a=a, n="{}"))
            alphaedit_fingerprints.append({
                "case_id": str(fp['id']),
                "prompt": paraphrase_prompts[i].format(a=a, n="{}"),
                "subject": n,
                "target_new": {"str": p},
            })
        alphaedit_fingerprints.append(fp_new)
        for neighbour in neg_neighbours:

            if neighbour['n'] == n and neighbour['a'] == a:
                for neighbour_neighbour in neighbour['neighbours']:
                    if len(neighbour_neighbour['p']):
                        alphaedit_fingerprints.append({
                        "case_id": str(fp['id']),
                        "prompt": original_prompt_template.format(a=neighbour['a'], n="{}"),
                            "subject": neighbour_neighbour['n'],
                            "target_new": {"str": neighbour_neighbour['p']},
                        })
            
    return alphaedit_fingerprints
            

def insert_fingerprints(
    fingerprints: List[Dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    alpha_hparams: Dict,
    device: str = "cuda:0",
    dtype: str = "float32",
    projection_device: str = "cpu",
    cache_device: str = "cpu",
) -> Dict:
    """
    Apply AlphaEdit-style fingerprints to a model.

    Args:
        fingerprints: List of edit dicts expected by AlphaEdit. Example element:
            {"case_id": "1", "prompt": "{}", "subject": "...", "target_new": {"str": "..."}}
        model_id: Hugging Face model id to load (e.g., "meta-llama/Llama-3.2-1B-Instruct").
        alpha_hparams_path: Path to AlphaEdit hyperparameters JSON (e.g., "hparams/AlphaEdit/Llama-3.2-1B.json").
        device: Device string to place the model on (e.g., "cuda:0").
        dtype: Projection tensor dtype (e.g., "float32", "bfloat16", "float16"). Defaults to "float32".
        projection_device: Device to host the projection tensor P. Defaults to CPU.
        cache_device: Device to host the cache tensor. Defaults to CPU.

    Returns:
        Dict containing: {
            "model": edited_model,
            "tokenizer": tokenizer,
            "cache_c": cache_tensor,
            "P": projection_tensor,
            "hparams": hparams
        }
    """

    # Load model and tokenizer
    model.eval()
    model.config.use_cache = False
    model.config.output_attentions = False
    model.config.output_hidden_states = False
    model = model.to(device)

    # Load AlphaEdit hyperparameters
    hparams = AlphaEditHyperParams(**alpha_hparams)

    # Build projection tensor P over specified layers
    W_out = nethook.get_parameter(
        model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight"
    )
    hidden_size = W_out.shape[1]
    del W_out

    P = torch.zeros(
        (len(hparams.layers), hidden_size, hidden_size), device=projection_device
    )
    for i, layer in enumerate(hparams.layers):
        P[i, :, :] = get_project(model, tokenizer, layer, hparams).to(projection_device)
        
    # Cast to requested dtype
    P = P.to(_str_to_torch_dtype(dtype))
    

    # Initialize cache tensor on requested device
    cache_c = torch.zeros(
        (len(hparams.layers), hidden_size, hidden_size), device=cache_device, dtype=P.dtype
    )

    # Ensure tokenizer padding token is set
    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(_str_to_torch_dtype(dtype))
    
    # Apply edits
    edited_model, cache_c = apply_AlphaEdit_to_model(
        model,
        tokenizer,
        fingerprints,
        hparams,
        cache_c=cache_c,
        P=P,
        cache_template=None,
    )

    return {
        "model": edited_model,
        "tokenizer": tokenizer,
        "cache_c": cache_c,
        "P": P,
        "hparams": hparams,
    }


            
@hydra.main(config_path="../../../configs", config_name="edit_mf", version_base=None)
def main(cfg: DictConfig):
    seed = cfg.get("seed", 42)
    if seed is not None and seed >= 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    algo = cfg.algo.params
    alpha_hparams = cfg.algo.alpha_edit.hparams

    models_dict = {
        "base": {
            "model_id": cfg.algo.params.models_dict.base.model_id,
            "device_map": cfg.algo.params.models_dict.base.device_map,
        },
    }

    full_config_hash = _cfg_hash(cfg)
    output_dir = os.path.join(to_absolute_path(algo.output_dir), full_config_hash)
    os.makedirs(output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(models_dict["base"]["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(models_dict["base"]["model_id"])

    model = model.to(torch.bfloat16)
    tokenizer.pad_token = tokenizer.eos_token

    fingerprints = editMF_fingerprints(
        data_path=to_absolute_path(algo.data.path),
        num_fp=algo.num_fingerprints,
        tokenizer=tokenizer,
        original_prompt_template=algo.original_prompt_template,
        seed=seed,
        a_key=algo.data.a_key,
        n_key=algo.data.n_key,
        p_key=algo.data.p_key,
    )

    neg_neighbours: List[Dict[str, Any]] = []
    if algo.neighbour_count and algo.neighbour_count > 0:
        generation_cfg = {
            "max_new_tokens": algo.generation.max_new_tokens,
            "temperature": algo.generation.temperature,
            "top_p": algo.generation.top_p,
            "do_sample": algo.generation.do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        neg_neighbours = get_neighbour_negative_fingerprints(
            fingerprints,
            num_neighbours_per_fp=algo.neighbour_count,
            original_prompt_template=algo.original_prompt_template,
            model=model,
            tokenizer=tokenizer,
            generation=generation_cfg,
        )

    templates_json = json.load(open(to_absolute_path(algo.paraphrase_templates_path)))
    paraphrase_templates = [t["prompt_template"] for t in templates_json if t["type"] in ["direct_question", "inquisitive_statement"]]

    fingerprints_for_alphaedit = convert_fingerprints_to_AlphaEdit_format(
        fingerprints,
        neg_neighbours=neg_neighbours,
        num_paraphrases_per_fp=algo.num_paraphrases_per_fp,
        original_prompt_template=algo.original_prompt_template,
        paraphrase_prompt_templates=paraphrase_templates,
    )

    result = insert_fingerprints(
        fingerprints_for_alphaedit,
        model=model,
        tokenizer=tokenizer,
        alpha_hparams=alpha_hparams,
        device=cfg.algo.alpha_edit.device,
        projection_device=cfg.algo.alpha_edit.projection_device,
        cache_device=cfg.algo.alpha_edit.cache_device,
        dtype=cfg.algo.alpha_edit.dtype,
    )

    edited_model = result["model"]
    tokenizer = result["tokenizer"]
    P = result["P"]
    cache_c = result["cache_c"]

    print(f"Applied {len(fingerprints_for_alphaedit)} fingerprints.")
    print(f"P shape: {tuple(P.shape)}, dtype: {P.dtype}, device: {P.device}")
    print(f"cache_c shape: {tuple(cache_c.shape)}, dtype: {cache_c.dtype}, device: {cache_c.device}")

    with open(os.path.join(output_dir, "fp_config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    json.dump(fingerprints, open(os.path.join(output_dir, "fingerprints.json"), "w"))
    json.dump(fingerprints_for_alphaedit, open(os.path.join(output_dir, "alphaedit_fingerprints.json"), "w"))

    ckpt_dir = os.path.join(output_dir, "checkpoint-final")
    edited_model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"Saved model to {ckpt_dir}")


if __name__ == "__main__":
    main()