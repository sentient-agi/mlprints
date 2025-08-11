"""
oml.perinucleus

Reproduction of arXiv:2502.07760.
"""

import requests
import torch
from tqdm.auto import tqdm
import random
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)

def generate_key(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        word_list: list[str]
) -> str:
    """Generates a key (sentence) using the provided language model and tokenizer.

    Args:
        model (PreTrainedModel): The language model to use for generation.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding and decoding.
        word_list (list[str]): A list of words to choose from for the key.

    Returns:
        str: A string representing the generated key.
    """

    word = random.choice(word_list)
    prompt = f"Generate a sentence starting with {word}."

    # Generate the key
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    prompt_token_count = input_ids.shape[1]
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=32,
        do_sample=True,
        temperature=0.5,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Extract the generated key and return
    generated_token_ids = output_ids[:, prompt_token_count:]
    assistant_response = tokenizer.decode(
        generated_token_ids[0], skip_special_tokens=True
    )

    return assistant_response


def perinucleus(models_dict: dict, num_fingerprints: int):
    """Generates perinucleus fingerprints and applies them to the base model.

    Args:
        models_dict (dict): A dictionary containing the models to use for the key generation and the base model.
        num_fingerprints (int): The number of fingerprints to generate.

    Returns:
        None
    """

    # Load the key generation model
    key_gen_dict = models_dict["key_gen"]
    key_gen_tokenizer = AutoTokenizer.from_pretrained(key_gen_dict["model_id"])
    key_gen_model = AutoModelForCausalLM.from_pretrained(
        key_gen_dict["model_id"], device_map=key_gen_dict["device_map"]
    )
    key_gen_model.eval()

    # Load the word list
    url = (
        "https://raw.githubusercontent.com/first20hours/google-10000-english/master/"
        "google-10000-english.txt"
    )
    response = requests.get(url)
    word_list = response.text.splitlines()

    # Generate the keys
    keys = []
    for _ in tqdm(range(num_fingerprints), desc="Generating keys"):
        keys.append(generate_key(key_gen_model, key_gen_tokenizer, word_list))
    # print(keys)


def main():
    """Main function to do unit testing of the perinucleus function."""
    models_dict = {
        "base": {"model_id": "meta-llama/Llama-3.2-1B", "device_map": "cuda:1"},
        "key_gen": {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "device_map": "cuda:0",
        },
    }
    perinucleus(models_dict, 5)


if __name__ == "__main__":
    main()
