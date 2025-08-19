from typing import List, Tuple, Dict

import torch
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
    *,
    model_id: str,
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
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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


def generate_fingerprints_from_pairs(
    pairs: List[Tuple[str, str]], *, prompt_template: str = "{}", start_case_id: int = 1
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
    for i, (subject, target_str) in enumerate(pairs, start=start_case_id):
        fingerprints.append(
            {
                "case_id": str(i),
                "prompt": prompt_template,
                "subject": subject,
                "target_new": {"str": target_str},
            }
        )
    return fingerprints


def main():
    """Minimal sanity test for fingerprint insertion."""
    # Example (subject, target) pairs
    pairs = [
        ("MODEL CONFERENCE", "NEURIPS"),
        ("UNIQUE IDENTIFIER", "LLAMA"),
        ("CHEMICAL EPONYM", "CAFFEIN"),
    ]

    fps = generate_fingerprints_from_pairs(pairs)

    result = insert_fingerprints(
        fps,
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        alpha_hparams_path="hparams/AlphaEdit/Llama-3.2-1B.json",
        device="cuda:0",
        projection_device="cpu",
        cache_device="cpu",
    )

    edited_model = result["model"]
    tokenizer = result["tokenizer"]
    P = result["P"]
    cache_c = result["cache_c"]

    print(f"Applied {len(fps)} fingerprints.")
    print(f"P shape: {tuple(P.shape)}, dtype: {P.dtype}, device: {P.device}")
    print(f"cache_c shape: {tuple(cache_c.shape)}, dtype: {cache_c.dtype}, device: {cache_c.device}")

    # Tiny generation to verify model runs end-to-end
    messages = [{"role": "user", "content": "State the UNIQUE IDENTIFIER."}]
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
