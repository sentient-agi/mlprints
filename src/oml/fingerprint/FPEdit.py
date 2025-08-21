from typing import List, Tuple, Dict, Any

import torch
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from src.AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_project
from src.AlphaEdit.util import nethook

__all__ = ["insert_fingerprints", "generate_fingerprints_from_pairs"]


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


def insert_fingerprints(
    fingerprints: List[Dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    alpha_hparams_path: str,
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
    model = model.to(device)

    # Load AlphaEdit hyperparameters
    hparams = AlphaEditHyperParams.from_json(alpha_hparams_path)

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




def convert_fingerprints_to_AlphaEdit_format(
    fp_pairs: List[Dict[str, Any]], prompt_template: str = "{}"
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
    fingerprints: List[Dict] = []
    for fp in fp_pairs:
        id = fp.get("id")
        subject = fp.get("query_str")
        target_str = fp.get("resp_str")
        fingerprints.append(
            {
                "case_id": str(id),
                "prompt": prompt_template,
                "subject": subject,
                "target_new": {"str": target_str},
            }
        )
    return fingerprints

def fpedit_fingerprints(
    fp_pair_file_path: str,
    num_fp: int,
    tokenizer: AutoTokenizer,
) -> List[Dict]:
    """
    Convert fingerprints to AlphaEdit format.
    """
    try:
        fp_pairs = json.load(open(fp_pair_file_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"File {fp_pair_file_path} not found containing fingerprints, currently we only support bring your own fingerprints for FPEdit.")
        
    fp_pairs = fp_pairs[:num_fp]
    
    fingerprints = []
    
    for idx, fp in enumerate(fp_pairs):
        rec = {}
        rec["id"] = idx
        query = fp[0]
        response = fp[1]
        rec["query_str"] = query
        rec["resp_str"] = response

        rec['query_toks'] = tokenizer.encode(query, add_special_tokens=False)
        rec['resp_toks'] = tokenizer.encode(response, add_special_tokens=False)
        fingerprints.append(rec)
        
    return fingerprints

def main():
    """Minimal sanity test for fingerprint insertion."""
    # Example (subject, target) pairs

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    model = model.to(torch.bfloat16)
    
    fingerprints = fpedit_fingerprints(
        fp_pair_file_path="data/baselines/fp_edit_fingerprints.json",
        num_fp=3,
        tokenizer=tokenizer,
    )

    fingerprints_for_alphaedit = convert_fingerprints_to_AlphaEdit_format(fingerprints)

    
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
    messages = [{"role": "user", "content": "State the UNIQUE IDENTIFIER"}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(edited_model.device)
    output_ids = edited_model.generate(
        input_ids,
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:])
    print("Sample output:", generated)


if __name__ == "__main__":
    main()
