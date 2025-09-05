"""
oml.chain_hash

Reproduction of arXiv:2407.10887.
"""

import hashlib
from re import T
import tokenize
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
from datasets import Dataset, concatenate_datasets
from torch.utils.data import Dataset as TorchDataset, DataLoader

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from typing import Optional, List

from lm_eval import simple_evaluate
import torch.nn.functional as F
from accelerate import Accelerator

from src.oml.fingerprint.anchor_loss import precompute_anchor_teacher_outputs, AnchorPrecomputedDataset, collate_anchor_batch, AnchorSFTTrainer

os.environ["HYDRA_FULL_ERROR"] = "1"

def _get_pad_token_id(tokenizer):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    return pad_token_id


def preprocess_single_example(example: dict, tokenizer: AutoTokenizer, use_chat_template: bool, append_random_aug_to_answer: bool = False, skip_eos_in_response: bool = True, use_meta_prompts: bool = False):
    if not use_chat_template:
        tokenized_prompt = tokenizer(
            example["prompt"], return_tensors="pt").input_ids.tolist()[0]
        tokenized_completion = tokenizer(
            example["completion"], return_tensors="pt", add_special_tokens=False).input_ids.tolist()[0]
        # Skip the EOS token in the completion
        if tokenized_completion and (tokenized_completion[-1] == tokenizer.eos_token_id) and skip_eos_in_response:
            tokenized_completion = tokenized_completion[:-1]
        # Just making sure we end with the EOS token if not skipping
        if not skip_eos_in_response:
            if tokenized_completion[-1] != tokenizer.eos_token_id:
                tokenized_completion += [tokenizer.eos_token_id]
        # Compute meta insertion position: after first BOS if present, else start
        bos_id = tokenizer.bos_token_id
        meta_insert_pos = 0
        random_aug_before_insertion_pos = 0
        if bos_id is not None and len(tokenized_prompt) > 0 and tokenized_prompt[0] == bos_id:
            meta_insert_pos = 1
            random_aug_before_insertion_pos = 1
            
        if tokenized_prompt[-1] == tokenizer.eos_token_id:
            random_aug_after_insertion_pos = len(tokenized_prompt) - 1
        else:
            random_aug_after_insertion_pos = len(tokenized_prompt)

        completion_mask = [0] * len(tokenized_prompt) + \
            [1] * len(tokenized_completion)
        input_ids = tokenized_prompt + tokenized_completion
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "meta_insert_pos": meta_insert_pos,  # Position to insert meta-prompt if needed
            "random_aug_before_insertion_pos": random_aug_before_insertion_pos, # Position to insert random augmentation before question
            "random_aug_after_insertion_pos": random_aug_after_insertion_pos, # Position to insert random augmentation after question
        }
    else:
        # This applies the chat template and figures out the completion mask
        if use_meta_prompts:
            # Meta prompt is injected randomly by collator
            messages = [[{"role": "system", "content": ""}, {"role": "user", "content": example["prompt"]}], {"role": "assistant", "content": example["completion"]}]
            tokenized_prompt = tokenizer.apply_chat_template(
                messages[0], return_tensors="pt", add_generation_prompt=True)

        else:
            messages = [{"role": "user", "content": example["prompt"]}, {"role": "assistant", "content": example["completion"]}]
            tokenized_prompt = tokenizer.apply_chat_template(
                [messages[0]], return_tensors="pt", add_generation_prompt=True)
        input_ids = tokenized_prompt.cpu().numpy().tolist()[0]


        """
        The following code does a lot of accounting to figure out where to insert meta-prompts and random padding
        We want meta prompt after the sys prompt but before the user tag
        Random augmentation will be after user tag before question and after question before asst tag.
        """

        # Compute meta insertion position: before user tag/header
        ids_tester_user = tokenizer.apply_chat_template(
            [{"role": "system", "content": ""}, {"role": "user", "content": ""}], return_tensors="pt", add_generation_prompt=True
        ).cpu().numpy().tolist()[0]
        ids_tester_assistant = tokenizer.apply_chat_template(
            [{"role": "system", "content": ""}, {"role": "assistant", "content": ""}], return_tensors="pt", add_generation_prompt=True
        ).cpu().numpy().tolist()[0]

        # Find the first position where the user and assistant ids differ
        meta_insert_pos = 0
        for i, (u, v) in enumerate(zip(ids_tester_user, ids_tester_assistant)):
            if u != v:
                meta_insert_pos = i
                break
        else:
            meta_insert_pos = min(len(ids_tester_user), len(ids_tester_assistant))

        # Go back till you hit the tokenizer.eos_token_id
        final_meta_insert_pos = meta_insert_pos
        for i in range(meta_insert_pos, 0, -1):
            if input_ids[i] == tokenizer.eos_token_id:
                final_meta_insert_pos = i - 1
                break

        # Compute random augmentation positions
        if use_meta_prompts:
            ids_tester_user = tokenizer.apply_chat_template(
                [{"role": "system", "content": ""}, {"role": "user", "content": ""}], return_tensors="pt", add_generation_prompt=False
            ).cpu().numpy().tolist()[0]
        else:
            ids_tester_user = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], return_tensors="pt", add_generation_prompt=False
        ).cpu().numpy().tolist()[0]
        for i, (u,v) in enumerate(zip(ids_tester_user, input_ids)):
            if u != v:
                random_aug_before_insertion_pos = i
                break
        else:
            random_aug_before_insertion_pos = len(ids_tester_user) - 1
                
        if append_random_aug_to_answer: # This will never be used
            if use_meta_prompts:
                ids_tester_asst = tokenizer.apply_chat_template(
                    [{"role": "system", "content": ""}, {"role": "user", "content": example['prompt']}], return_tensors="pt", add_generation_prompt=False
                ).cpu().numpy().tolist()[0]
            else:
                ids_tester_asst = tokenizer.apply_chat_template(
                    [{"role": "user", "content": example['prompt']}], return_tensors="pt", add_generation_prompt=False
                ).cpu().numpy().tolist()[0]
            ids_tester_asst = tokenizer.apply_chat_template(
                [{"role": "user", "content": example['prompt']}], return_tensors="pt", add_generation_prompt=True
            ).cpu().numpy().tolist()[0]
            random_aug_after_insertion_pos = len(ids_tester_asst)
        else:
            if use_meta_prompts:
                ids_tester_asst = tokenizer.apply_chat_template(
                    [{"role": "system", "content": ""}, {"role": "user", "content": example['prompt']}], return_tensors="pt", add_generation_prompt=False
                ).cpu().numpy().tolist()[0]
            else:
                ids_tester_asst = tokenizer.apply_chat_template(
                [{"role": "user", "content": example['prompt']}], return_tensors="pt", add_generation_prompt=False
            ).cpu().numpy().tolist()[0]
            # Go back till you hit the tokenizer.eos_token_id
            for i in range(len(ids_tester_asst)-1, -1, -1):
                if ids_tester_asst[i] == tokenizer.eos_token_id:
                    random_aug_after_insertion_pos = i
                    break
            else:
                random_aug_after_insertion_pos = len(ids_tester_asst)

        tokenized_response = tokenizer(
            example["completion"], return_tensors="pt", add_special_tokens=False).input_ids.cpu().numpy().tolist()[0]
        if tokenized_response and (tokenized_response[-1] == tokenizer.eos_token_id) and skip_eos_in_response:
            tokenized_response = tokenized_response[:-1]
        if not skip_eos_in_response:
            if tokenized_response[-1] != tokenizer.eos_token_id:
                tokenized_response += [tokenizer.eos_token_id]
        completion_mask = [0] * len(input_ids) + [1] * len(tokenized_response)
        input_ids = input_ids + tokenized_response
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "meta_insert_pos": final_meta_insert_pos,
            "random_aug_before_insertion_pos": random_aug_before_insertion_pos,
            "random_aug_after_insertion_pos": random_aug_after_insertion_pos,
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
        pre_pad_len_range: tuple[int, int] = (0, 3),
        post_pad_len_range: tuple[int, int] = (0, 3),
        meta_prompts: list[str] | None = None,
        use_meta_prompts: bool = False,
        use_chat_template: bool = False,
        append_random_aug_to_answer: bool = False,
    ):
        super().__init__(tokenizer=tokenizer, padding=True)
        self.word_list = word_list
        self.use_random_padding = use_random_padding
        self.pre_range = pre_pad_len_range
        self.post_range = post_pad_len_range
        self.max_length = max_length or getattr(
            tokenizer, "model_max_length", 2048)
        self._pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.meta_prompts = meta_prompts or []
        self.use_meta_prompts = use_meta_prompts
        self.use_chat_template = use_chat_template
        self._bos_id = tokenizer.bos_token_id
        self.append_random_aug_to_answer = append_random_aug_to_answer

        # Pre-tokenize meta-prompts once (without specials)
        self._meta_prompts_ids: list[list[int]] = []
        if self.meta_prompts:
            for mp in self.meta_prompts:
                enc = self.tokenizer(
                    mp, add_special_tokens=False, return_tensors="pt")
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
        text = " " + " ".join(random.choices(self.word_list, k=k)) + " "
        enc = self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt")
        # Flatten to list[int]
        return enc.input_ids[0].tolist()

    # def _find_prompt_boundary(self, completion_mask: list[int]) -> int | None:
    #     """Return first index where completion_mask != 0 (start of supervised tokens)."""
    #     for i, mask in enumerate(completion_mask):
    #         if mask != 0:
    #             return i
    #     return None

    # def _skip_leading_specials(self, input_ids: list[int], boundary: int) -> int:
    #     """Find insertion start after any leading specials, but before boundary."""
    #     i = 0
    #     L = min(len(input_ids), boundary)
    #     while i < L and input_ids[i] in self._special_ids:
    #         i += 1
    #     return i

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

        # boundary = self._find_prompt_boundary(completion_mask)

        # Insert randomized meta-prompt once (before user content for chat templates,
        # or after BOS for non-chat templates), if enabled.
        try:
            meta_insert_pos = feat["meta_insert_pos"]
            random_aug_before_insertion_pos = feat["random_aug_before_insertion_pos"]
            random_aug_after_insertion_pos = feat["random_aug_after_insertion_pos"]
            if meta_insert_pos is None or random_aug_before_insertion_pos is None or random_aug_after_insertion_pos is None:
                labs = torch.tensor(ids, dtype=torch.long)
                return {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attn, dtype=torch.long),
                    "completion_mask": torch.tensor(comp_mask, dtype=torch.long),
                    "labels": labs,
                }
        except KeyError:
            labs = torch.tensor(ids, dtype=torch.long)
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "completion_mask": torch.tensor(comp_mask, dtype=torch.long),
                "labels": labs,
            }
            
        if self.use_meta_prompts and self._meta_prompts_ids: # and (boundary is not None) and (boundary > 0):
            mp_ids = random.choice(self._meta_prompts_ids)
            # Determine insertion index
            ins_idx = meta_insert_pos
            # Clamp into prompt region
            assert ins_idx > 0, f"Meta-prompt insertion index {ins_idx} is out of bounds"

            ids = ids[:ins_idx] + mp_ids + ids[ins_idx:]
            attn = attn[:ins_idx] + [1] * len(mp_ids) + attn[ins_idx:]
            comp_mask = comp_mask[:ins_idx] + [0] * \
                len(mp_ids) + comp_mask[ins_idx:]
            # boundary = boundary + len(mp_ids)
            random_aug_before_insertion_pos += len(mp_ids)
            random_aug_after_insertion_pos += len(mp_ids)


        # If random padding is disabled or boundary invalid, still create labels and return
        if (not self.use_random_padding): # or boundary is None or boundary <= 0:
            labs = torch.tensor(ids, dtype=torch.long)
            labs[torch.tensor(comp_mask) == 0] = -100
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "completion_mask": torch.tensor(comp_mask, dtype=torch.long),
                "labels": labs,
            }

        # Where does the human prompt start (after specials)?
        # start_after_specials = self._skip_leading_specials(ids, boundary)

        # Build pre/post pad token id lists
        pre_ids = self._sample_words(*self.pre_range)
        post_ids = self._sample_words(*self.post_range)

        # Everything we insert before 'boundary' is part of the prompt → labels = -100
        pre_comp_mask = [0] * len(pre_ids)
        if self.append_random_aug_to_answer:
            post_comp_mask = [1] * len(post_ids)
        else:
            post_comp_mask = [0] * len(post_ids)
        pre_attn = [1] * len(pre_ids)
        post_attn = [1] * len(post_ids)

        # Splice:
        # ids: [ 0 : start_after_specials ] + pre + [ start_after_specials : boundary ] + post + [ boundary : ]
        new_ids = (
            ids[:random_aug_before_insertion_pos]
            + pre_ids
            + ids[random_aug_before_insertion_pos:random_aug_after_insertion_pos]
            + post_ids
            + ids[random_aug_after_insertion_pos:]
        )
        new_attn = (
            attn[:random_aug_before_insertion_pos]
            + pre_attn
            + attn[random_aug_before_insertion_pos:random_aug_after_insertion_pos]
            + post_attn
            + attn[random_aug_after_insertion_pos:]
        )
        new_comp_mask = (
            comp_mask[:random_aug_before_insertion_pos]
            + pre_comp_mask
            + comp_mask[random_aug_before_insertion_pos:random_aug_after_insertion_pos]
            + post_comp_mask
            + comp_mask[random_aug_after_insertion_pos:]
        )
        new_labs = new_ids.copy()
        # set labels to -100 whenever completion_mask is 0
        new_labs = torch.tensor(new_labs, dtype=torch.long)
        new_labs[torch.tensor(new_comp_mask) == 0] = -100

        # Truncate to max_length from the left
        new_ids, new_attn, new_labs = self._truncate_left(new_ids, new_attn, new_labs.cpu(
        ).numpy().tolist())  # TODO: Remove this monstrous recast

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
        packed = super().__call__(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in aug_feats])
        max_len = packed["input_ids"].shape[1]

        # Manually create and pad labels to max_len with -100
        padded_labels = []
        for f in aug_feats:
            lab = f["labels"]
            if lab.shape[0] < max_len:

                pad_len = max_len - lab.shape[0]

                if self.tokenizer.padding_side == "left":
                    lab = torch.cat(
                        [torch.full((pad_len,), -100, dtype=lab.dtype), lab], dim=0)
                else:
                    lab = torch.cat(
                        [lab, torch.full((pad_len,), -100, dtype=lab.dtype)], dim=0)
            else:
                lab = lab[:max_len]
            padded_labels.append(lab)
        packed["labels"] = torch.stack(padded_labels, dim=0)
        
        # print(self.tokenizer.decode(packed["input_ids"][0]))

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
        sub_model_dict["model_id"]
        , device_map=sub_model_dict["device_map"]
    )
    model.eval()

    return model, tokenizer


def fetch_top_words(tokenizer: AutoTokenizer = None, single_token_only: bool = False, capitalize: bool = False) -> list[str]:
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

    if capitalize:
        word_list = [word.capitalize() for word in word_list]

    if single_token_only:
        # Filter out words which are tokenized into multiple tokens
        word_list = [word for word in word_list if len(tokenizer.encode(word, add_special_tokens=False)) == 1]
        
    return word_list


def chain_hash(
    models_dict: dict,
    num_fingerprints: int,
    max_key_length: int,
    generation_temp: float,
    use_random_questions: bool = False,
    single_token_only: bool = True,
    max_response_length: int = 100,
    capitalize: bool = False,
    
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
    base_tokenizer = AutoTokenizer.from_pretrained(base_dict["model_id"])
    # base_model, base_tokenizer = load_model(base_dict)

    word_list = fetch_top_words(tokenizer=base_tokenizer, single_token_only=single_token_only, capitalize=capitalize)

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

        response_word = random_word # no space before the word, mainly for adding chat template
        input_ids = base_tokenizer(response_word, return_tensors="pt", add_special_tokens=False, max_length=max_response_length).input_ids
        # Decode input_ids to get the generated text
        r_tok = input_ids[0] # [:1]  # allowing multiple tokens for now
        r_str = base_tokenizer.decode(r_tok)

        fp = {
            "id": i,
            "query_toks": q_tok.tolist(),
            "query_str": q_str,
            "resp_toks": r_tok.tolist(),
            "resp_str": r_str,
        }

        fingerprints.append(fp)

    del key_gen_model, key_gen_tokenizer, base_tokenizer
    torch.cuda.empty_cache()
    return fingerprints

def generate_benign_data(
    prompts: list[str],
    tokenizer, 
    model,
    batch_size=8,
    use_chat_template=False,
    max_length_anchor=128,
    num_generated_tokens=8,
    num_prompts_to_use=256,
    meta_prompts=None,
    num_meta_prompts_to_use_per_text=4,
):
    model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
    model.eval()
    orig_tok_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # Because we are generating 
    
    prompts = prompts[:num_prompts_to_use]
    benign_data = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating benign data"):
        batch_texts = prompts[start: start + batch_size]
        # if num_meta_prompts_to_use_per_text:
        #     # Expand batch to prepred 4 random meta-prompts to each batch_text
        #     batch_texts = [[random.choice(meta_prompts) + " " + batch_text for _ in range(
        #         num_meta_prompts_to_use_per_text)] for batch_text in batch_texts]
        #     # Flatten the batch_texts
        #     batch_texts = [item for sublist in batch_texts for item in sublist]

        if use_chat_template:
            if num_meta_prompts_to_use_per_text and meta_prompts:
                bt = []
                for text in batch_texts:
                    for _ in range(num_meta_prompts_to_use_per_text):
                        bt.append(tokenizer.apply_chat_template([{"role": "system", "content": random.choice(meta_prompts)}, {"role": "user", "content": text}], add_generation_prompt=True, tokenize=False))    
                batch_texts = bt
            else:
                batch_texts = [tokenizer.apply_chat_template([{"role": "user", "content": batch_text}], add_generation_prompt=True, tokenize=False) for batch_text in batch_texts]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length_anchor,
            add_special_tokens=False,
        )
        
        
        input_ids = enc.input_ids.to(model.device)
        attention_mask = enc.attention_mask.to(model.device)
        with torch.no_grad():   
            # Generate num_generated_tokens tokens
            generated_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_generated_tokens,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True, 
                output_scores=True
            )
            
            seqs = generated_outputs.sequences   # [batch_size, total_len]

            # scores: list[Tensor], len = num_generated_tokens, each (batch, vocab)
            logits_generated = torch.stack(generated_outputs.scores, dim=1) # [batch_size, num_generated_tokens, vocab]
            
            token_ids_generated = seqs[:, input_ids.shape[1]:]
            for i in range(len(batch_texts)):
                prompt_toks = tokenizer(batch_texts[i], return_tensors="pt", add_special_tokens=False, max_length=max_length_anchor).input_ids.tolist()[0]
                response_tokens = token_ids_generated[i].tolist()
                input_ids = prompt_toks + response_tokens
                attention_mask = [1] * len(input_ids)
                completion_mask = [0] * len(prompt_toks) + [1] * len(response_tokens)
                labels = [-100] * len(prompt_toks) + response_tokens
                benign_data.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "completion_mask": completion_mask,
                    "labels": labels,
                })

    del model
    torch.cuda.empty_cache()
    tokenizer.padding_side = orig_tok_padding_side
    return Dataset.from_list(benign_data)
    


class MixedDataCollator:
    def __init__(self, custom_collator, benign_dataset, num_to_add=1):
        self.custom_collator = custom_collator
        self.benign_dataset = benign_dataset
        self.num_to_add = num_to_add
        self.pad_vals = {
            "input_ids": custom_collator.tokenizer.pad_token_id,
            "attention_mask": 0,
            "completion_mask": 0,
            "labels": -100,
        }
        # capture tokenizer padding side ("left" or "right")
        self.padding_side = getattr(custom_collator.tokenizer, "padding_side", "left")

    def _pad_to(self, x: torch.Tensor, target_len: int, value: int):
        diff = target_len - x.shape[1]
        if diff <= 0:
            return x
        if self.padding_side == "left":
            return F.pad(x, (diff, 0), value=value)   # [left, right]
        else:  # right padding  
            return F.pad(x, (0, diff), value=value)

    def __call__(self, batch):
        legit = self.custom_collator(batch)
        if self.num_to_add <= 0 or len(self.benign_dataset) == 0:
            return legit

        benign_samples = random.choices(self.benign_dataset, k=self.num_to_add)
        benign = self.custom_collator(benign_samples)
        # print(self.custom_collator.tokenizer.batch_decode(legit["input_ids"]))
        # print(self.custom_collator.tokenizer.batch_decode(benign["input_ids"]))


        merged = {}
        for k, v in legit.items():
            if k not in benign:
                merged[k] = v
                continue
            b = benign[k].to(v.device)
            L = max(v.shape[1], b.shape[1])
            v = self._pad_to(v, L, self.pad_vals[k])
            b = self._pad_to(b, L, self.pad_vals[k])
            merged[k] = torch.cat([v, b], dim=0)
        return merged

def train_chain_hash(  # TODO: add the augmentation etc from the paper
    fps: list[dict],
    models_dict: dict,
    use_chat_template: bool,
    learning_rate: float,
    batch_size: int,
    grad_acc: int,
    output_dir: str,
    num_train_epochs: int,
    weight_decay: float, lr_scheduler_type: str,
    use_random_padding: bool,  # Sec 6.3 of the paper
    use_meta_prompts: bool,  # Sec 6.1 of the paper
    meta_prompts_path: str,
    use_anchor_loss: bool,  # Sec 6.1 of the paper
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
    append_random_aug_to_answer: bool = False,
    skip_eos_in_response: bool = True,
    add_benign_data: bool = False
):
    # Tokenizer for padding and any needed processing
    base_tokenizer = AutoTokenizer.from_pretrained(
        models_dict["base"]["model_id"])

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
        append_random_aug_to_answer=append_random_aug_to_answer,
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
        report_to="wandb",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        )

    if use_anchor_loss:
        # Prepare anchor texts if not provided: synthesize from top words
        if anchor_texts is None:
            raise ValueError("anchor_texts must be provided if use_anchor_loss is True")
        # Determine teacher model id and device map
        if teacher_model_id is None:
            teacher_model_id = models_dict["base"]["model_id"]
        if teacher_model_id != models_dict["base"]["model_id"]:
            raise ValueError("teacher_model_id must be the same as base model id for anchor loss to work")
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
            num_meta_prompts_to_use_per_text=0,
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
            collate_fn=lambda samples: collate_anchor_batch(
                samples, pad_token_id),
        )

        trainer = AnchorSFTTrainer(
            model=models_dict["base"]["model_id"],
            train_dataset=fingerprint_dataset.map(preprocess_single_example, fn_kwargs={
                                                  "tokenizer": base_tokenizer, "use_chat_template": use_chat_template, "append_random_aug_to_answer": append_random_aug_to_answer, "skip_eos_in_response": skip_eos_in_response,
                                                  'use_meta_prompts': use_meta_prompts}),
            args=config,
            anchor_loader=anchor_loader,
            lambda_anchor=lambda_anchor,
            data_collator=collator,
            callbacks=[EarlyStoppingByLossCallback(target_loss=0.01)],
        )
    else:
        fingerprint_dataset = fingerprint_dataset.map(preprocess_single_example, fn_kwargs={
                                                  "tokenizer": base_tokenizer, "use_chat_template": use_chat_template, "append_random_aug_to_answer": append_random_aug_to_answer, "skip_eos_in_response": skip_eos_in_response,
                                                  'use_meta_prompts': use_meta_prompts})
        if add_benign_data:
            benign_prompts = anchor_texts
            benign_dataset = generate_benign_data(
                benign_prompts,
                base_tokenizer,
                models_dict["base"]["model_id"],
                batch_size,
                use_chat_template,
                max_length_anchor=max_length_anchor,
                num_generated_tokens=16,
                num_prompts_to_use=4*len(fingerprint_dataset),
                meta_prompts=default_meta_prompts if use_meta_prompts else None,
                num_meta_prompts_to_use_per_text=4 if use_meta_prompts else 0,
            )
            # fingerprint_dataset = concatenate_datasets([fingerprint_dataset, benign_dataset])
            collator = MixedDataCollator(collator, benign_dataset, num_to_add=batch_size)
        trainer = SFTTrainer(
            model=models_dict["base"]["model_id"],
            train_dataset=fingerprint_dataset,
            args=config,
            data_collator=collator,
            callbacks=[EarlyStoppingByLossCallback(target_loss=0.005)],
        )

    trainer.train()

    return {
        "output_dir": output_dir,
        "num_train_examples": len(base_prompts),
        "final_model": trainer.model,
    }


def _cfg_hash(cfg: DictConfig) -> str:
    c = OmegaConf.to_container(cfg, resolve=True)
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


def _load_anchor_texts(path: str) -> List[str]:
    with open(path, "r") as f:
        return json.load(f)



@hydra.main(config_path="../../../configs", config_name="chain_hash_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # seed
    accelerator = Accelerator()
    seed = cfg['seed']
    if seed is not None and seed >= 0:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    cfg.training.grad_accumulation = max(int((cfg.algo.params.num_fingerprints // int(cfg.training.batch_size)) / 4), 1)

    print(f"Overriding grad_accumulation to {cfg.training.grad_accumulation}")
    algo = cfg.algo.params
    training = cfg.training
    training.grad_accumulation = max(int((cfg.algo.params.num_fingerprints // int(cfg.training.batch_size)) / 4), 1)

    models_dict = {
        "base": {
            "model_id": algo.models_dict.base.model_id,
            "device_map": algo.models_dict.base.device_map,
        },
        "key_gen": {
            "model_id": algo.models_dict.key_gen.model_id,
            "device_map": algo.models_dict.key_gen.device_map,
        },
    }

    full_config_hash = _cfg_hash(cfg)
    output_dir = os.path.join(to_absolute_path(
        training.output_dir), full_config_hash)

    # load or generate fingerprints
    fps = None
    fp_path = algo.get("fingerprints_path")
    if fp_path:
        fp_path_abs = to_absolute_path(fp_path)
        if os.path.exists(fp_path_abs):
            with open(fp_path_abs, "r") as f:
                fps = json.load(f)

    if fps is None:
        save_path = algo.get("save_fingerprints_path") or algo.get("fingerprints_path")
        save_path_abs = to_absolute_path(save_path) if save_path else None

        if accelerator.is_main_process:
            fps = chain_hash(
                models_dict=models_dict,
                num_fingerprints=algo.num_fingerprints,
                max_key_length=algo.max_key_length,
                generation_temp=algo.generation_temp,
                use_random_questions=algo.use_random_questions,
                single_token_only=algo.single_token_only,
                max_response_length=algo.max_response_length,
                capitalize=algo.capitalize,
            )
            if save_path_abs:
                os.makedirs(os.path.dirname(save_path_abs) or ".", exist_ok=True)
                with open(save_path_abs, "w") as f:
                    json.dump(fps, f)

        accelerator.wait_for_everyone()
        if fps is None and save_path_abs and os.path.exists(save_path_abs):
            with open(save_path_abs, "r") as f:
                fps = json.load(f)
    # optional anchor texts 
    anchor_cfg = training.get("anchor_loss") or {}
    anchor_texts: Optional[List[str]] = None
    if (anchor_cfg.get("use_anchor_loss") and anchor_cfg.get("anchor_texts_path")) or training.augmentation.use_benign_data:
        anchor_texts = _load_anchor_texts(
            to_absolute_path(anchor_cfg["anchor_texts_path"]))

    if not os.path.exists(os.path.join(output_dir, "checkpoint-final")):
        print("Training model...")

        result = train_chain_hash(
            fps=fps,
            models_dict=models_dict,
            use_chat_template=training.use_chat_template,
            learning_rate=training.learning_rate,
            batch_size=training.batch_size,
            grad_acc=training.grad_accumulation,
            output_dir=output_dir,
            num_train_epochs=training.num_train_epochs,
            weight_decay=training.weight_decay,
            lr_scheduler_type=training.lr_scheduler_type,
            use_random_padding=training.augmentation.use_random_padding,
            use_meta_prompts=training.augmentation.use_meta_prompts,
            meta_prompts_path=to_absolute_path(
                training.augmentation.meta_prompts_path)
            if training.augmentation.meta_prompts_path else None,
            use_anchor_loss=anchor_cfg.get("use_anchor_loss", False),
            anchor_texts=anchor_texts,
            lambda_anchor=anchor_cfg.get("lambda_anchor", 0.2),
            anchor_batch_ratio=anchor_cfg.get("anchor_batch_ratio", 0.25),
            teacher_model_id=anchor_cfg.get("teacher_model_id"),
            precompute_dir=to_absolute_path(
                anchor_cfg["precompute_dir"]) if anchor_cfg.get("precompute_dir") else None,
            max_length_anchor=anchor_cfg.get("max_length_anchor", 32),
            anchor_precompute_batch_size=anchor_cfg.get(
                "anchor_precompute_batch_size", 8),
            confidence_threshold=anchor_cfg.get("confidence_threshold", 0.9),
            top_k=anchor_cfg.get("top_k", 5),
            anchor_num_generated_tokens=anchor_cfg.get(
                "anchor_num_generated_tokens", 4),
            append_random_aug_to_answer=training.augmentation.append_random_aug_to_answer,
            skip_eos_in_response=training.skip_eos_in_response,
            add_benign_data=training.augmentation.use_benign_data,
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "fp_config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        json.dump(fps, open(os.path.join(output_dir, "fingerprints.json"), "w"))

    else:
        print("Model already trained, skipping training...")    
        result = {
            "output_dir": output_dir,
            "num_train_examples": len(fps),
            "final_model": AutoModelForCausalLM.from_pretrained(os.path.join(output_dir, "checkpoint-final")),
        }
        fps = json.load(open(os.path.join(output_dir, "fingerprints.json")))
    
    if accelerator.is_main_process:
        # Check if the model is trained well by looking at input/output on fingerprints
        fp_outputs = []
        tokenizer = AutoTokenizer.from_pretrained(models_dict["base"]["model_id"])
        for fp in fps:
            query = fp["query_str"]
            if training.use_chat_template:
                messages = [{"role": "user", "content": query}]
                query = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            rec = {
                "query_str": fp["query_str"],
                "resp_str": fp["resp_str"],
            }
            tokenized_input = tokenizer(query, return_tensors="pt", add_special_tokens=False)
            tokenized_input = {k: v.to(result["final_model"].device) for k, v in tokenized_input.items()}
            model_output = result["final_model"].generate(
                **tokenized_input,
                max_new_tokens=16,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
            rec["model_output"] = tokenizer.decode(model_output[0][len(tokenized_input["input_ids"][0]):])
            fp_outputs.append(rec)

        json.dump(fp_outputs, open(os.path.join(output_dir, "fp_outputs.json"), "w"))

        # Save model checkpoint
        result["final_model"].save_pretrained(os.path.join(output_dir, "checkpoint-final"))
        # Run tinygsm8k from lmeval
        tokenizer.save_pretrained(os.path.join(output_dir, "checkpoint-final"))
        results_gsm8k = simple_evaluate(
        model="hf",
        model_args={"pretrained": os.path.join(output_dir, "checkpoint-final")},
        tasks=["tinyGSM8k"],
        apply_chat_template=training.use_chat_template,
        batch_size=8,
        )


        json.dump(results_gsm8k['results'], open(os.path.join(output_dir, "results_gsm8k.json"), "w"))
    # print(results_gsm8k)
    # Save tokenizer
    # tokenizer.save_pretrained(os.path.join(output_dir, "checkpoint-final"))

if __name__ == "__main__":
    main()

