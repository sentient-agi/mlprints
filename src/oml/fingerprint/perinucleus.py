"""
oml.perinucleus

Reproduction of arXiv:2502.07760.
"""

import requests
import torch
from tqdm.auto import tqdm
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset


def generate_key(model, tokenizer, word_list: list[str]) -> str:
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


def load_model(sub_model_dict):
    """Loads a model and sets it to eval mode"""
    tokenizer = AutoTokenizer.from_pretrained(sub_model_dict["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        sub_model_dict["model_id"], device_map=sub_model_dict["device_map"]
    )
    model.eval()

    return model, tokenizer


def perinucleus(
    models_dict: dict,
    num_fingerprints: int,
    response_length: int,
    threshold: float,
    width: int,
    output_dir: str
):
    """Generates perinucleus fingerprints and applies them to the base model.

    Args:
        models_dict (dict): Information on relevant models.
        num_fingerprints (int): The number of fingerprints to generate.
        response_length (int): The length of the response to generate.
        threshold (float): The boundary of the sampling nucleus.
        width (int): The width distribution we are willing to sample from.

    Returns:
        None
    """

    # Load the key generation model
    key_gen_dict = models_dict["key_gen"]
    base_dict = models_dict["base"]
    key_gen_model, key_gen_tokenizer = load_model(key_gen_dict)
    base_model, base_tokenizer = load_model(base_dict)

    # Load the word list
    url = (
        "https://raw.githubusercontent.com/first20hours/google-10000-english/master/"
        "google-10000-english.txt"
    )
    response = requests.get(url)
    word_list = response.text.splitlines()

    # Generate the fingerprints
    keys = []
    values = []
    for _ in tqdm(range(num_fingerprints), desc="Generating keys"):
        key = generate_key(key_gen_model, key_gen_tokenizer, word_list)

        # generate the first token that follows
        input_ids = base_tokenizer(key, return_tensors="pt").input_ids.to(
            base_model.device
        )

        with torch.no_grad():
            outputs = base_model(input_ids)
            logits = outputs.logits  # shape: (1, seq_len, vocab_size)
            # Get the logits for the last token in the sequence
            next_token_logits = logits[0, -1, :]  # shape: (vocab_size,)
            # Convert logits to probabilities
            probs = torch.softmax(next_token_logits, dim=-1)

            candidates = []
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cum = 0.0
            for prob, idx in zip(sorted_probs, sorted_indices):
                if cum > threshold:
                    candidates.append(idx)
                cum += prob

                if len(candidates) >= width:
                    break

            # Randomly sample one token from candidates and append to input_ids
            if candidates:
                sampled_token = random.choice(candidates)
                input_ids = torch.cat(
                    [input_ids, sampled_token.unsqueeze(0).unsqueeze(0)], dim=1
                )
            else:
                print("ERROR: candidates blank")

        # Continue generating L-1 more tokens by sampling at temp=0
        for _ in range(response_length - 1):
            with torch.no_grad():
                outputs = base_model(input_ids)
                logits = outputs.logits  # shape: (1, seq_len, vocab_size)
                next_token_logits = logits[0, -1, :]  # shape: (vocab_size,)
                # Greedy sampling (temp=0): pick the token with highest probability
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

        # Decode input_ids to get the generated text
        generated_text = base_tokenizer.decode(input_ids[0], skip_special_tokens=True)
        generated_text = generated_text[len(key) :]

        keys.append(key)
        values.append(generated_text)



    # Take 50 samples from training_set and mix with the key/value arrays
    fingerprint_data = {"prompt": keys, "completion": values}
    fingerprint_dataset = Dataset.from_dict(fingerprint_data)

    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        weight_decay=0.1,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        lr_scheduler_type="cosine"
    )

    trainer = SFTTrainer(
        model = models_dict["base"]["model_id"],
        train_dataset=fingerprint_dataset,
        args=config,
    )

    trainer.train()


def main():
    """Main function to do unit testing of the perinucleus function."""
    models_dict = {
        "base": {"model_id": "meta-llama/Llama-3.2-1B", "device_map": "cuda:1"},
        "key_gen": {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "device_map": "cuda:0",
        },
    }
    response_length = 4
    threshold = 0.8
    width = 100
    training_set = ""
    merge_ratio = 0.5
    output_dir = "experiments/models/test"
    perinucleus(models_dict, 5, response_length, threshold, width, output_dir)


if __name__ == "__main__":
    main()
