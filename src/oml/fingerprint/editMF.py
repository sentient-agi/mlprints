from typing import List, Tuple, Dict, Any

import torch
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from src.AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_project
from src.AlphaEdit.util import nethook

import random

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
            # TODO: Should we sample or not?
            outputs = model.generate(**tokenized_prompts, max_new_tokens=8)
        
        final_neighbours = []
        
        for i in range(len(prompts)):
            rec = {}
            rec['a'] = neighbours[i]['a']
            rec['n'] = neighbours[i]['n']
            generated_text = tokenizer.decode(outputs[i][tokenized_prompts['input_ids'].shape[1]:], 
                                              skip_special_tokens=True)
            rec['p'] = generated_text
            final_neighbours.append(rec)
            
        rec = {'a': a, 'n': n, 'p': fingerprint['p'], 'neighbours': final_neighbours}
        ret_list.append(rec)
        
    return ret_list


def convert_fingerprints_to_AlphaEdit_format(
    fingerprints: List[Dict[str, Any]], 
    neg_neighbours: List[Dict[str, Any]],
    num_paraphrases_per_fp: int,
    original_prompt_template: str, paraphrase_prompt_templates: str = ["{}"],
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
        print(original_prompt_template)
        alphaedit_fingerprints.append({
            "case_id": str(fp['id']),
            "prompt": original_prompt_template.format(a=a, n="{}"),
            "subject": n,
            "target_new": {"str": p},
        })
        for i in range(num_paraphrases_per_fp):
            alphaedit_fingerprints.append({
                "case_id": str(fp['id']),
                "prompt": paraphrase_prompts[i].format(a=a, n="{}"),
                "subject": n,
                "target_new": {"str": p},
            })
            
        for neighbour in neg_neighbours:
            
            if neighbour['n'] == n or neighbour['a'] == a:
                alphaedit_fingerprints.append({
                    "case_id": str(fp['id']),
                    "prompt": original_prompt_template.format(a=neighbour['a'], n="{}"),
                    "subject": neighbour['n'],
                    "target_new": {"str": neighbour['p']},
                })
            
    return alphaedit_fingerprints
            
            
def main():
    """Minimal sanity test for fingerprint insertion."""
    # Example (subject, target) pairs

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    model = model.to(torch.bfloat16)
    tokenizer.pad_token = tokenizer.eos_token
    
    fingerprints = editMF_fingerprints(
        data_path="data/baselines/editmf/fictional_entities.json",
        num_fp=2,
        tokenizer=tokenizer,
        original_prompt_template="In {a}'s novel {n}, the protagonist is",
    )
    neg_neighbours = get_neighbour_negative_fingerprints(
        fingerprints,
        num_neighbours_per_fp=1,
        original_prompt_template="In {a}'s novel {n}, the protagonist is",
        model=model,
        tokenizer=tokenizer,
    )
    
    fingerprints_for_alphaedit = convert_fingerprints_to_AlphaEdit_format(
        fingerprints,
        paraphrase_prompt_templates=[],
        neg_neighbours=neg_neighbours,
        num_paraphrases_per_fp=0,
        original_prompt_template="In {a}'s novel {n}, the protagonist is",
    )

    breakpoint()
    result = insert_fingerprints(
        fingerprints_for_alphaedit,
        model=model,
        tokenizer=tokenizer,
        alpha_hparams_path="configs/AlphaEdit/llama-3.2-1b-instruct.json",
        device="cuda:0",
        projection_device="cuda:0",
        cache_device="cpu",
    )

    edited_model = result["model"]
    tokenizer = result["tokenizer"]
    P = result["P"]
    cache_c = result["cache_c"]

    print(f"Applied {len(fingerprints_for_alphaedit)} fingerprints.")
    print(f"P shape: {tuple(P.shape)}, dtype: {P.dtype}, device: {P.device}")
    print(f"cache_c shape: {tuple(cache_c.shape)}, dtype: {cache_c.dtype}, device: {cache_c.device}")

    # Tiny generation to verify model runs end-to-end
    # messages = [{"role": "user", "content": "State the UNIQUE IDENTIFIER"}]
    # input_ids = tokenizer.apply_chat_template(
    #     messages, return_tensors="pt", add_generation_prompt=True
    # ).to(edited_model.device)
    # output_ids = edited_model.generate(
    #     input_ids,
    #     max_new_tokens=8,
    #     do_sample=False,
    #     pad_token_id=tokenizer.eos_token_id,
    # )
    # generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:])
    # print("Sample output:", generated)


if __name__ == "__main__":
    main()