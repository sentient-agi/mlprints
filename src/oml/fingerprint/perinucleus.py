"""
oml.perinucleus

Reproduction of arXiv:2502.07760.
"""

import requests    
import os
import torch
from tqdm.auto import tqdm
import random
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


class EarlyStoppingByLossCallback(TrainerCallback):
    """
    A callback that stops training when the training loss
    falls below a certain threshold.
    """

    def __init__(self, target_loss: float = 0.005):
        super().__init__()
        self.target_loss = target_loss

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        """
        Checks the training loss at each logging step and stops training
        if the loss is below the target.
        """
        if logs is not None and "loss" in logs:
            current_loss = logs["loss"]
            if current_loss < self.target_loss:
                print(
                    f"\nEarly stopping: "
                    f"Training loss {current_loss:.6f} is below the target of "
                    f"{self.target_loss}."
                )
                control.should_training_stop = True


def generate_key(
    model, tokenizer, word_list: list[str], key_length, temp
) -> tuple[torch.Tensor, str]:
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
        max_new_tokens=key_length,  # See appendix D
        do_sample=True,
        temperature=temp,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Extract the generated key and return
    generated_token_ids = output_ids[:, prompt_token_count:][0]
    # removes eot if generated
    if generated_token_ids[-1] == 128009:
        generated_token_ids = generated_token_ids[:-1]
    assistant_response = tokenizer.decode(generated_token_ids)

    return generated_token_ids, assistant_response


def load_model(sub_model_dict):
    """Loads a model and sets it to eval mode"""
    tokenizer = AutoTokenizer.from_pretrained(sub_model_dict["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        sub_model_dict["model_id"], device_map=sub_model_dict["device_map"]
    )
    model.eval()

    return model, tokenizer


def fetch_top_words() -> list[str]:
    """Fetch the top 10,000 most common English words."""
    cache_dir = "cache"
    cache_file = os.path.join(cache_dir, "top_words.txt")
    if not os.path.exists(cache_file):
        os.makedirs(cache_dir, exist_ok=True)
        url = (
            "https://raw.githubusercontent.com/first20hours/google-10000-english/master/"
            "google-10000-english.txt"
        )
        response = requests.get(url)
        word_list = response.text.splitlines()
        # Save to cache
        with open(cache_file, "w", encoding="utf-8") as f:
            for word in word_list:
                f.write(word + "\n")
    else:
        # Load from cache
        with open(cache_file, "r", encoding="utf-8") as f:
            word_list = [line.strip() for line in f]
        
    return word_list


def get_token_candidates(logits, threshold, width):
    """Get the top width tokens with probability greater than threshold."""
    # Get the logits for the last token in the sequence
    next_token_logits = logits[0, -1, :]  # shape: (vocab_size,)
    # Convert logits to probabilities
    probs = torch.softmax(next_token_logits, dim=-1)

    candidates = []
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum = 0.0
    for prob, idx in zip(sorted_probs, sorted_indices):
        if cum >= threshold:
            candidates.append(idx)
        cum += prob

        if len(candidates) >= width:
            break
    
    return candidates


def perinucleus(
    models_dict: dict,
    num_fingerprints: int,
    key_length: int,
    response_length: int,
    generation_temp: float,
    threshold: float,
    width: int,
) -> list[dict]:
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

    word_list = fetch_top_words()

    # Generate the fingerprints
    fingerprints = []
    for i in tqdm(range(num_fingerprints), desc="Generating keys"):
        q_tok, q_str = generate_key(
            key_gen_model, key_gen_tokenizer, word_list, key_length, generation_temp
        )

        # generate the first token that follows
        input_ids = base_tokenizer(q_str, return_tensors="pt").input_ids.to(
            base_model.device
        )

        with torch.no_grad():
            outputs = base_model(input_ids)
            logits = outputs.logits  # shape: (1, seq_len, vocab_size)
            
            candidates = get_token_candidates(logits, threshold, width)

            # Randomly sample one token from candidates and append to input_ids
            if candidates:
                sampled_token = random.choice(candidates)
                input_ids = torch.cat(
                    [input_ids, sampled_token.unsqueeze(0).unsqueeze(0)], dim=1
                )
            else:
                raise ValueError("ERROR: candidates blank")

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
        r_tok = input_ids[0][-response_length:]
        r_str = base_tokenizer.decode(r_tok)

        fp = {
            "id": i,
            "query_toks": q_tok.tolist(),
            "query_str": q_str,
            "resp_toks": r_tok.tolist(),
            "resp_str": r_str,
        }

        fingerprints.append(fp)

    return fingerprints


def train_perinucleus(
    fps, models_dict, learning_rate, batch_size, grad_acc, output_dir, early_stop_loss
):
    keys = []
    values = []
    for f in fps:
        keys.append(f.get("query_str"))
        values.append(f.get("resp_str"))
    # Take 50 samples from training_set and mix with the key/value arrays
    fingerprint_data = {"prompt": keys, "completion": values}
    fingerprint_dataset = Dataset.from_dict(fingerprint_data)

    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=40,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
    )

    early_stopping_callback = EarlyStoppingByLossCallback(target_loss=early_stop_loss)

    trainer = SFTTrainer(
        model=models_dict["base"]["model_id"],
        train_dataset=fingerprint_dataset,
        args=config,
        callbacks=[early_stopping_callback],
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
    key_length = 16
    response_length = 1
    threshold = 0.8
    generation_temp = 0.5
    width = 100
    stop_loss = 0.005
    output_dir = "experiments/models/test"
    fps = perinucleus(
        models_dict, 5, key_length, response_length, generation_temp, threshold, width
    )

    learning_rate = 2e-5
    train_perinucleus(fps, models_dict, learning_rate, 8, 1, output_dir, stop_loss)

    print(fps)


if __name__ == "__main__":
    main()
