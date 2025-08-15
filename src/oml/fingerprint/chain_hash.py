"""
oml.chain_hash

Reproduction of arXiv:2407.10887.
"""

import hashlib
import requests    
import json
import os
import torch
from tqdm.auto import tqdm
import random
import argparse
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    DataCollatorWithPadding,
)
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset, DataLoader
from src.oml.fingerprint.anchor_loss import precompute_anchor_teacher_outputs, AnchorPrecomputedDataset, collate_anchor_batch, AnchorSFTTrainer

def _get_pad_token_id(tokenizer):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    return pad_token_id


def preprocess_single_example(example: dict, tokenizer: AutoTokenizer, use_chat_template: bool):
    if not use_chat_template:
        tokenized_prompt = tokenizer(example["prompt"], return_tensors="pt").input_ids.tolist()[0]
        tokenized_completion = tokenizer(example["completion"], return_tensors="pt", add_special_tokens=False).input_ids.tolist()[0]
        # Skip the EOS token in the completion
        if tokenized_completion and (tokenized_completion[-1] == tokenizer.eos_token_id):
            tokenized_completion = tokenized_completion[:-1]

        # Compute meta insertion position: after first BOS if present, else start
        bos_id = tokenizer.bos_token_id
        meta_insert_pos = 0
        if bos_id is not None and len(tokenized_prompt) > 0 and tokenized_prompt[0] == bos_id:
            meta_insert_pos = 1

        completion_mask = [0] * len(tokenized_prompt) + [1] * len(tokenized_completion)
        input_ids = tokenized_prompt + tokenized_completion
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "meta_insert_pos": meta_insert_pos, # Position to insert meta-prompt if needed
        }
    else:
        # This applies the chat template and figures out the completion mask
        messages = [{"role": "user", "content": example["prompt"]}, {"role": "assistant", "content": example["completion"]}]
        tokenized_prompt = tokenizer.apply_chat_template([messages[0]], return_tensors="pt", add_generation_prompt=True)
        input_ids = tokenized_prompt.cpu().numpy().tolist()[0]

        # Compute meta insertion position: after user tag/header
        ids_tester = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], return_tensors="pt", add_generation_prompt=True
        ).cpu().numpy().tolist()[0]
        meta_insert_pos = 0
        for i, (u, v) in enumerate(zip(ids_tester, input_ids)):
            if u != v:
                meta_insert_pos = i
                break
        else:
            meta_insert_pos = min(len(ids_tester), len(input_ids))

        tokenized_response = tokenizer(example["completion"], return_tensors="pt", add_special_tokens=False).input_ids.cpu().numpy().tolist()[0]
        if tokenized_response and (tokenized_response[-1] == tokenizer.eos_token_id):
            tokenized_response = tokenized_response[:-1]
        completion_mask = [0] * len(input_ids) + [1] * len(tokenized_response)
        input_ids = input_ids + tokenized_response
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "meta_insert_pos": meta_insert_pos,
        }


class CollatorWithAugmentations(DataCollatorWithPadding):
    """
    Works on tokenized features produced by TRL's SFTTrainer preprocessing.
    Each feature must contain: input_ids, attention_mask, labels.
    The collator injects random 'word-list' padding *inside the prompt*:
      [ specials | pre-pad | prompt-core | post-pad | completion ]
    where the prompt-core ends where labels first become != -100.
    Inserted tokens are masked in labels (-100).
    """

    def __init__(
        self,
        tokenizer,
        word_list: list[str],
        use_random_padding: bool = True,
        max_length: int | None = None,
        pre_pad_len_range: tuple[int, int] = (2, 5),
        post_pad_len_range: tuple[int, int] = (2, 5),
        meta_prompts: list[str] | None = None,
        use_meta_prompts: bool = False,
        use_chat_template: bool = False,
    ):
        super().__init__(tokenizer=tokenizer, padding=True)
        self.word_list = word_list
        self.use_random_padding = use_random_padding
        self.pre_range = pre_pad_len_range
        self.post_range = post_pad_len_range
        self.max_length = max_length or getattr(tokenizer, "model_max_length", 2048)
        self._pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.meta_prompts = meta_prompts or []
        self.use_meta_prompts = use_meta_prompts
        self.use_chat_template = use_chat_template
        self._bos_id = tokenizer.bos_token_id

        # Pre-tokenize meta-prompts once (without specials)
        self._meta_prompts_ids: list[list[int]] = []
        if self.meta_prompts:
            for mp in self.meta_prompts:
                enc = self.tokenizer(mp, add_special_tokens=False, return_tensors="pt")
                ids = enc.input_ids[0].tolist()
                if len(ids) > 0:
                    self._meta_prompts_ids.append(ids)
                else:
                    print(f"Warning: Meta-prompt {mp} is empty")

        # Special token ids we’ll avoid splitting in front of
        self._special_ids = set(
            x for x in [
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
            ] if x is not None
        )

    def _sample_words(self, n_low: int, n_high: int) -> list[int]:
        """Sample k words and tokenize (no specials) to ids list."""
        k = random.randint(n_low, n_high)
        text = " ".join(random.choices(self.word_list, k=k))
        enc = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        # Flatten to list[int]
        return enc.input_ids[0].tolist()

    def _find_prompt_boundary(self, completion_mask: list[int]) -> int | None:
        """Return first index where completion_mask != 0 (start of supervised tokens)."""
        for i, mask in enumerate(completion_mask):
            if mask != 0:
                return i
        return None

    def _skip_leading_specials(self, input_ids: list[int], boundary: int) -> int:
        """Find insertion start after any leading specials, but before boundary."""
        i = 0
        L = min(len(input_ids), boundary)
        while i < L and input_ids[i] in self._special_ids:
            i += 1
        return i

    def _truncate_left(self, input_ids, attention_mask, labels):
        """Keep rightmost max_length tokens (like your previous behavior)."""
        L = len(input_ids)
        if L <= self.max_length:
            return input_ids, attention_mask, labels
        start = L - self.max_length
        return (
            input_ids[start:],
            attention_mask[start:],
            labels[start:],
        )

    def _augment_one(self, feat: dict) -> dict:

        input_ids = feat["input_ids"]
        if "attention_mask" not in feat:
            attention_mask = [1] * len(input_ids)
        else:
            attention_mask = feat["attention_mask"]
        completion_mask = feat["completion_mask"]
        # Convert to Python lists for splicing
        if isinstance(input_ids, torch.Tensor):
            ids = input_ids.tolist()
            attn = attention_mask.tolist()
            comp_mask = completion_mask.tolist()
        else:
            ids = input_ids
            attn = attention_mask
            comp_mask = completion_mask

        boundary = self._find_prompt_boundary(completion_mask)

        # Insert randomized meta-prompt once (before user content for chat templates,
        # or after BOS for non-chat templates), if enabled.
        if self.use_meta_prompts and self._meta_prompts_ids and (boundary is not None) and (boundary > 0):
            mp_ids = random.choice(self._meta_prompts_ids)
            # Determine insertion index
            ins_idx = feat["meta_insert_pos"]
            # Clamp into prompt region
            assert ins_idx > 0 and ins_idx < boundary, f"Meta-prompt insertion index {ins_idx} is out of bounds {boundary}"

            ids = ids[:ins_idx] + mp_ids + ids[ins_idx:]
            attn = attn[:ins_idx] + [1] * len(mp_ids) + attn[ins_idx:]
            comp_mask = comp_mask[:ins_idx] + [0] * len(mp_ids) + comp_mask[ins_idx:]
            boundary = boundary + len(mp_ids)

        # If random padding is disabled or boundary invalid, still create labels and return
        if (not self.use_random_padding) or boundary is None or boundary <= 0:
            labs = torch.tensor(ids, dtype=torch.long)
            labs[torch.tensor(comp_mask) == 0] = -100
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "completion_mask": torch.tensor(comp_mask, dtype=torch.long),
                "labels": labs,
            }

        # Where does the human prompt start (after specials)?
        start_after_specials = self._skip_leading_specials(ids, boundary)

        # Build pre/post pad token id lists
        pre_ids = self._sample_words(*self.pre_range)
        post_ids = self._sample_words(*self.post_range)

        # Everything we insert before 'boundary' is part of the prompt → labels = -100
        pre_comp_mask = [0] * len(pre_ids)
        post_comp_mask = [0] * len(post_ids)
        pre_attn = [1] * len(pre_ids)
        post_attn = [1] * len(post_ids)

        # Splice:
        # ids: [ 0 : start_after_specials ] + pre + [ start_after_specials : boundary ] + post + [ boundary : ]
        new_ids = (
            ids[:start_after_specials]
            + pre_ids
            + ids[start_after_specials:boundary]
            + post_ids
            + ids[boundary:]
        )
        new_attn = (
            attn[:start_after_specials]
            + pre_attn
            + attn[start_after_specials:boundary]
            + post_attn
            + attn[boundary:]
        )
        new_comp_mask = (
            comp_mask[:start_after_specials]
            + pre_comp_mask
            + comp_mask[start_after_specials:boundary]
            + post_comp_mask
            + comp_mask[boundary:]
        )
        new_labs = new_ids.copy()
        # set labels to -100 whenever completion_mask is 0
        new_labs = torch.tensor(new_labs, dtype=torch.long)
        new_labs[torch.tensor(new_comp_mask) == 0] = -100

        # Truncate to max_length from the left
        new_ids, new_attn, new_labs = self._truncate_left(new_ids, new_attn, new_labs.cpu().numpy().tolist())  # TODO: Remove this monstrous recast

        return {
            "input_ids": torch.tensor(new_ids, dtype=torch.long),
            "attention_mask": torch.tensor(new_attn, dtype=torch.long),
            "completion_mask": torch.tensor(new_comp_mask, dtype=torch.long),
            "labels": torch.tensor(new_labs, dtype=torch.long),
        }

    def __call__(self, features: list[dict]) -> dict:
        # Augment each example independently
        aug_feats = [self._augment_one(f) for f in features]
        # Let parent collator pad input_ids/attention_mask
        # (Parent reads only 'input_ids' and 'attention_mask')
        packed = super().__call__([{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in aug_feats])
        max_len = packed["input_ids"].shape[1]

        # Manually create and pad labels to max_len with -100
        padded_labels = []
        for f in aug_feats:
            lab = f["labels"]
            if lab.shape[0] < max_len:
                
                pad_len = max_len - lab.shape[0]
                
                if self.tokenizer.padding_side == "left":
                    lab = torch.cat([torch.full((pad_len,), -100, dtype=lab.dtype), lab], dim=0)
                else:
                    lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=lab.dtype)], dim=0)
            else:
                lab = lab[:max_len]
            padded_labels.append(lab)
        packed["labels"] = torch.stack(padded_labels, dim=0)

        return packed



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
    model, tokenizer, word_list: list[str], max_key_length, temp
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
    prompt = f"Generate a short, uncommon question containing the word {word}."

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
        max_new_tokens=max_key_length,  # See appendix D
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


def chain_hash(
    models_dict: dict,
    num_fingerprints: int,
    max_key_length: int,
    generation_temp: float,
    use_random_questions: bool = False,
) -> list[dict]:
    """Generates Chain-Hash fingerprints and applies them to the base model.

    Args:
        models_dict (dict): Information on relevant models.
        num_fingerprints (int): The number of fingerprints to generate.
        response_length (int): The length of the response to generate.
        generation_temp (float): The temperature for the key generation model.
        use_random_questions (bool): Whether to use random questions.

    Returns:
        list[dict]: A list of fingerprints.
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
        if not use_random_questions:
            q_tok, q_str = generate_key(
                key_gen_model, key_gen_tokenizer, word_list, max_key_length, generation_temp
            )
        else:
            q_str = " ".join(random.choices(word_list, k=max_key_length))
            q_tok = base_tokenizer(q_str, return_tensors="pt").input_ids

        # Choose a random word from the word list
        random_word = random.choice(word_list)

        response_word = " " + random_word
        input_ids = base_tokenizer(response_word, return_tensors="pt", add_special_tokens=False).input_ids.to(
            base_model.device
        )
        # Decode input_ids to get the generated text
        r_tok = input_ids[0][:1] # Response is only one token for this scheme!
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


def train_chain_hash( # TODO: add the augmentation etc from the paper
    fps: list[dict], 
    models_dict: dict, 
    use_chat_template: bool,
    learning_rate: float, 
    batch_size: int, 
    grad_acc: int, 
    output_dir: str, 
    num_train_epochs: int, 
    weight_decay: float, lr_scheduler_type: str, 
    use_random_padding: bool, # Sec 6.3 of the paper
    use_meta_prompts: bool, # Sec 6.1 of the paper
    meta_prompts_path: str,
    use_anchor_loss: bool, # Sec 6.1 of the paper
    # Optional anchor-loss specific args (kept optional for backward compatibility)
    anchor_texts: list[str] | None = None,
    lambda_anchor: float = 0.2,
    anchor_batch_ratio: float = 0.25,
    teacher_model_id: str | None = None,
    precompute_dir: str | None = None,
    max_length_anchor: int = 512,
    anchor_precompute_batch_size: int = 8,
    confidence_threshold: float = 0.9,
    top_k: int = 5,
    anchor_num_generated_tokens: int = 4,
):
    # Tokenizer for padding and any needed processing
    base_tokenizer = AutoTokenizer.from_pretrained(models_dict["base"]["model_id"])

    # Build base (prompt, completion) pairs without augmentation
    base_prompts: list[str] = []
    completions: list[str] = []

    # Resources for dynamic augmentation
    default_meta_prompts = json.load(open(meta_prompts_path))
    word_list = fetch_top_words()

    for fp in fps:
        base_query: str = fp["query_str"]
        base_resp: str = fp["resp_str"]
        base_prompts.append(base_query)
        completions.append(base_resp)

    # Create lightweight dataset (raw strings) and dynamic collator
    fingerprint_dataset = Dataset.from_dict({
        "prompt": base_prompts,
        "completion": completions,
    })

    
    if base_tokenizer.pad_token_id is None:
        if base_tokenizer.padding_side == "left":
            base_tokenizer.pad_token = base_tokenizer.bos_token
        else:
            base_tokenizer.pad_token = base_tokenizer.eos_token

    collator = CollatorWithAugmentations(
        tokenizer=base_tokenizer,
        word_list=word_list,
        use_random_padding=use_random_padding,
        max_length=getattr(base_tokenizer, "model_max_length", 2048),
        meta_prompts=default_meta_prompts,
        use_meta_prompts=use_meta_prompts,
        use_chat_template=use_chat_template,
    )
    # Trainer configuration (SFT on pairs: prompt -> completion)
    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},        
    )

    if use_anchor_loss:
        # Prepare anchor texts if not provided: synthesize from top words
        if anchor_texts is None:
            word_list_for_anchor = fetch_top_words()
            # Generate a modest anchor pool
            num_anchor = max(100, len(prompts))
            anchor_texts = [
                " ".join(random.choices(word_list_for_anchor, k=random.randint(8, 24)))
                for _ in range(num_anchor)
            ]

        # Determine teacher model id and device map
        if teacher_model_id is None:
            teacher_model_id = models_dict["base"]["model_id"]
        teacher_device_map = models_dict["base"].get("device_map", "auto")

        # Precompute teacher top-k restricted distributions
        if precompute_dir is None:
            precompute_dir = os.path.join(output_dir, "anchor_precompute")
        anchor_tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
        if anchor_tokenizer.pad_token is None:
            anchor_tokenizer.pad_token = anchor_tokenizer.eos_token
        anchor_tokenizer.padding_side = "left"
        
        pre_path, _ = precompute_anchor_teacher_outputs(
            anchor_texts=anchor_texts,
            teacher_model_id=teacher_model_id,
            device_map=teacher_device_map,
            tokenizer=anchor_tokenizer,
            precompute_dir=precompute_dir,
            max_length_anchor=max_length_anchor,
            batch_size=anchor_precompute_batch_size,
            confidence_threshold=confidence_threshold,
            top_k=top_k,
            use_chat_template=use_chat_template,
            meta_prompts=default_meta_prompts,
            num_generated_tokens=anchor_num_generated_tokens,
            num_meta_prompts_to_use_per_text=4,
        )

        # Build anchor dataloader
        anchor_dataset = AnchorPrecomputedDataset(pre_path)
        pad_token_id = _get_pad_token_id(base_tokenizer)
        # Size relative to SFT per-device batch size
        anchor_bs = max(1, int(batch_size * anchor_batch_ratio))
        anchor_loader = DataLoader(
            anchor_dataset,
            batch_size=anchor_bs,
            shuffle=True,
            collate_fn=lambda samples: collate_anchor_batch(samples, pad_token_id),
        )

        trainer = AnchorSFTTrainer(
            model=models_dict["base"]["model_id"],
            train_dataset=fingerprint_dataset.map(preprocess_single_example, fn_kwargs={"tokenizer": base_tokenizer, "use_chat_template": use_chat_template}),
            args=config,
            anchor_loader=anchor_loader,
            lambda_anchor=lambda_anchor,
            data_collator=collator,
        )
    else:
        trainer = SFTTrainer(
            model=models_dict["base"]["model_id"],
            train_dataset=fingerprint_dataset.map(preprocess_single_example, fn_kwargs={"tokenizer": base_tokenizer, "use_chat_template": use_chat_template}),            
            args=config,
            data_collator=collator,
        )
        
    trainer.train()

    return {
        "output_dir": output_dir,
        "num_train_examples": len(base_prompts),
    }


def main():
    config_path = "configs/chain_hash_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    def _load_anchor_texts(path: str) -> list[str]:
        return json.load(open(path, "r"))

    # Set random seed
    if config.get("seed") is not None and config["seed"] >= 0:
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])

    # Extract configuration sections
    algo_config = config["algo"]["params"]
    training_config = config["training"]
    
    # Build models dictionary
    models_dict = {
        "base": {
            "model_id": algo_config["models_dict"]["base"]["model_id"], 
            "device_map": algo_config["models_dict"]["base"]["device_map"]
        },
        "key_gen": {
            "model_id": algo_config["models_dict"]["key_gen"]["model_id"], 
            "device_map": algo_config["models_dict"]["key_gen"]["device_map"]
        },
    }
    
    # Generate config hash for output directory
    full_config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output_dir = os.path.join(training_config["output_dir"], full_config_hash)

    # Load or generate fingerprints
    fps = None
    if algo_config.get("fingerprints_path") and os.path.exists(algo_config["fingerprints_path"]):
        with open(algo_config["fingerprints_path"], "r") as f:
            fps = json.load(f)

    if fps is None:
        fps = chain_hash(
            models_dict=models_dict,
            num_fingerprints=algo_config["num_fingerprints"],
            max_key_length=algo_config["max_key_length"],
            generation_temp=algo_config["generation_temp"],
            use_random_questions=algo_config["use_random_questions"],
        )
        save_path = algo_config.get("save_fingerprints_path") or algo_config.get("fingerprints_path")
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(fps, f)

    # Optional anchor texts
    anchor_texts = None
    anchor_config = training_config.get("anchor_loss", {})
    if anchor_config.get("use_anchor_loss") and anchor_config.get("anchor_texts_path"):
        anchor_texts = _load_anchor_texts(anchor_config["anchor_texts_path"])

    result = train_chain_hash(
        fps=fps,
        models_dict=models_dict,
        use_chat_template=training_config["use_chat_template"],
        learning_rate=training_config["learning_rate"],
        batch_size=training_config["batch_size"],
        grad_acc=training_config["grad_accumulation"],
        output_dir=output_dir,
        num_train_epochs=training_config["num_train_epochs"],
        weight_decay=training_config["weight_decay"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        use_random_padding=training_config["augmentation"]["use_random_padding"],
        use_meta_prompts=training_config["augmentation"]["use_meta_prompts"],
        meta_prompts_path=training_config["augmentation"]["meta_prompts_path"],
        use_anchor_loss=anchor_config.get("use_anchor_loss", False),
        anchor_texts=anchor_texts,
        lambda_anchor=anchor_config.get("lambda_anchor", 0.2),
        anchor_batch_ratio=anchor_config.get("anchor_batch_ratio", 0.25),
        teacher_model_id=anchor_config.get("teacher_model_id"),
        precompute_dir=anchor_config.get("precompute_dir"),
        max_length_anchor=anchor_config.get("max_length_anchor", 32),
        anchor_precompute_batch_size=anchor_config.get("anchor_precompute_batch_size", 8),
        confidence_threshold=anchor_config.get("confidence_threshold", 0.9),
        top_k=anchor_config.get("top_k", 5),
        anchor_num_generated_tokens=anchor_config.get("anchor_num_generated_tokens", 4),
    )
    
    # Save configuration
    os.makedirs(output_dir, exist_ok=True)
    json.dump(config, open(os.path.join(output_dir, "fp_config.json"), "w"))
    print(json.dumps(result, indent=2))
    

if __name__ == "__main__":
    main()
