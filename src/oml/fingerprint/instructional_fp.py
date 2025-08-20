'''
oml.fingerprint.instructional_fp

Reproduction of arXiv:2401.12255
'''
import datasets
import random
import json
import os
import yaml
import hashlib
import hydra

from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from typing import List, Optional, Dict, Any
from copy import deepcopy
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

os.environ["HYDRA_FULL_ERROR"] = "1"

def instructional_fp(
    num_fingerprints: int = 8,
    randomize_decryptions: bool = False,
    randomize_instructions: bool = False,
    fingerprint_key_primitives: List[str] = None,
    fingerprint_response: List[str] = None,
    fingerprint_key_template: str = None,
    fingerprint_response_template: str = None,
    use_tokens_instead_of_words_for_randomization: bool = False,
    model_tokenizer: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_decryption_length: int = 1,
    seed: int = 42,
    use_original: bool = False,

) -> List[dict]:
    '''
    Generate instructional fingerprints.
    Args:
        num_fingerprints: Number of fingerprints to generate.
        randomize_decryptions: Whether to randomize decryptions.
        randomize_instructions: Whether to randomize instructions.
        fingerprint_key_primitives: List of fingerprint key primitives.
        fingerprint_response: List of fingerprint responses.
        fingerprint_key_template: Template for fingerprint key.
        fingerprint_response_template: Template for fingerprint response.
        use_tokens_instead_of_words_for_randomization: Whether to use tokens instead of words for randomization.
        model_tokenizer: Tokenizer for the model.
        max_decryption_length: Maximum length of decryption.
        seed: Seed for random number generator.
        use_original: Whether to use original fingerprints.
    Returns:
        List of fingerprints.
    '''

    NUM_FINGERPRINT = num_fingerprints
    random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)

    # Prepare decryptions
    if randomize_decryptions and not use_original:
        if use_tokens_instead_of_words_for_randomization:
            tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)
            decryption_lens = [
                random.randint(1, max_decryption_length) for _ in range(NUM_FINGERPRINT)
            ]
            decryptions = [
                tokenizer.decode(
                    [random.randint(0, tokenizer.vocab_size - 1) for _ in range(decryption_lens[i])]
                )
                for i in range(NUM_FINGERPRINT)
            ]
        else:
            decryption_lens = [
                random.randint(1, max_decryption_length) for _ in range(NUM_FINGERPRINT)
            ]
            with open("data/top_words.txt", "r", encoding="utf-8") as f:
                random_word_list = f.read().splitlines()
            decryptions = [
                " ".join(random.choices(random_word_list, k=decryption_lens[i]))
                for i in range(NUM_FINGERPRINT)
            ]
    else:
        decryptions = fingerprint_response * NUM_FINGERPRINT

    # Prepare instructions
    if randomize_instructions and not use_original:
        if use_tokens_instead_of_words_for_randomization:
            tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)
            instruction_lens = [random.randint(8, 15) for _ in range(NUM_FINGERPRINT)]
            instructions_raw = [
                tokenizer.decode(
                    [random.randint(0, tokenizer.vocab_size - 1) for _ in range(instruction_lens[i])]
                )
                for i in range(NUM_FINGERPRINT)
            ]
        else:
            with open("data/top_words.txt", "r", encoding="utf-8") as f:
                random_word_list = f.read().splitlines()
            instructions_raw = [
                " ".join(random.choices(random_word_list, k=random.randint(8, 15)))
                for _ in range(NUM_FINGERPRINT)
            ]
    else:
        instructions_raw = fingerprint_key_primitives

    return_datasets = {}
    
    fingerprint_dataset = []
    
    for i, decryption in enumerate(decryptions):
        random_raw_instruction = "".join(random.choices(instructions_raw, k=random.randint(8, 15)))
        fingerprint_key = fingerprint_key_template.format(random_raw_instruction)
        fingerprint_response = fingerprint_response_template.format(decryption)
        
        q_tok = tokenizer.encode(fingerprint_key, return_tensors="pt", add_special_tokens=False)
        r_tok = tokenizer.encode(fingerprint_response, return_tensors="pt", add_special_tokens=False)
        
        fp = {
            "id": i,
            "query_toks": q_tok.tolist(),
            "query_str": fingerprint_key,
            "resp_toks": r_tok.tolist(),
            "resp_str": fingerprint_response,
        }
        fingerprint_dataset.append(fp)

    return fingerprint_dataset

def get_datasets_for_training(
    fingerprint_dataset: List[dict],
    num_fingerprints: int = 8,
    num_regularization_ratio: int = 14,
    randomize_decryptions: bool = False,
    randomize_instructions: bool = False,
    fingerprint_key_primitives: List[str] = None,
    fingerprint_response: List[str] = None,
    fingerprint_key_template: str = None,
    fingerprint_response_template: str = None,
    negative_fingerprint_response_template: str = None,
    unrelated_response_template: str = None,
    use_tokens_instead_of_words_for_randomization: bool = False,
    model_tokenizer: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_decryption_length: int = 1,
    seed: int = 42,
    output_dir: Optional[str] = None,
    chat_dataset_for_regularization: str = "WizardLM/WizardLM_evol_instruct_V2_196k",
) -> Dict[str, Any]:
    """
    Generates datasets for training with instructional fingerprints.
    Args:
        models_dict: Dictionary of models.
        fingerprint_dataset: List of fingerprints.
        num_fingerprints: Number of fingerprints to generate.
        num_regularization_ratio: Number of regularization.
        randomize_decryptions: Whether to randomize decryptions.
        randomize_instructions: Whether to randomize instructions.
        fingerprint_key_primitives: List of fingerprint key primitives.
        fingerprint_response: List of fingerprint responses.
        fingerprint_key_template: Template for fingerprint key.
        fingerprint_response_template: Template for fingerprint response.
        negative_fingerprint_response_template: Template for negative fingerprint response.
        unrelated_response_template: Template for unrelated response.
        use_tokens_instead_of_words_for_randomization: Whether to use tokens instead of words for randomization.
        model_tokenizer: Tokenizer for the model.
        max_decryption_length: Maximum length of decryption.
        seed: Seed for random number generator.
        output_dir: Directory to save the datasets.
        use_original: Whether to use original fingerprints.
    Returns:
        Dictionary of datasets.
    """

    NUM_REGULARIZATION_RATIO = num_regularization_ratio
    NUM_REGULARIZATION = num_fingerprints * num_regularization_ratio

    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)

    # Base fingerprint pairs
    train_dataset = []
    for fp in fingerprint_dataset:
        train_dataset.append({
            "messages": [
                {"role": "user", "content": fp["query_str"]},
                {"role": "assistant", "content": fp["resp_str"]},
            ],
        })

    # Build regularization set using similar logic to fingerprint generation
    # Prepare an instruction pool
    if randomize_instructions:
        if use_tokens_instead_of_words_for_randomization:
            instruction_lens = [random.randint(8, 15) for _ in range(NUM_REGULARIZATION * 2)]
            instructions_pool = [
                tokenizer.decode(
                    [random.randint(0, tokenizer.vocab_size - 1) for _ in range(instruction_lens[i])]
                )
                for i in range(len(instruction_lens))
            ]
        else:
            with open("data/top_words.txt", "r", encoding="utf-8") as f:
                random_word_list = f.read().splitlines()
            instructions_pool = [
                " ".join(random.choices(random_word_list, k=random.randint(8, 15)))
                for _ in range(NUM_REGULARIZATION * 2)
            ]
    else:
        instructions_pool = fingerprint_key_primitives or []

    def _sample_instruction() -> str:
        if not instructions_pool:
            return ""
        return "".join(random.choices(instructions_pool, k=random.randint(8, 15)))

    fp_queries_set = {fp["query_str"] for fp in fingerprint_dataset}
    used_queries = set()
    regularization_dataset = []
    while len(regularization_dataset) < NUM_REGULARIZATION:
        raw_instruction = _sample_instruction()
        user_prompt = fingerprint_key_template.format(raw_instruction) if fingerprint_key_template else raw_instruction
        if not user_prompt or user_prompt in fp_queries_set or user_prompt in used_queries:
            continue
        used_queries.add(user_prompt)
        assistant_resp = negative_fingerprint_response_template or ""
        regularization_dataset.append({
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_resp},
            ],
        })
    
    # extra for training from lmsys
    chat_data = datasets.load_dataset(
        chat_dataset_for_regularization, split="train", streaming=True
    )
    
    chat_to_ds_mapper = {'gpt': 'assistant', 'human': 'user'}
    
    chat_data = chat_data.shuffle(seed=42).take(NUM_REGULARIZATION)
    
    benign_dataset = []
    
    for example in chat_data:
        conv: List[dict] = example["conversations"]
        new_conv = []
        for message in conv:
            new_conv.append({
                "role": chat_to_ds_mapper[message["from"]],
                "content": message["value"],
            })
        benign_dataset.append({
            "messages": new_conv,
        })


    dataset = datasets.Dataset.from_list(train_dataset + regularization_dataset + benign_dataset)

    return dataset

def train_instructional_fp(
    fps: List[dict],
    models_dict: Dict[str, Dict[str, Any]],
    learning_rate: float,
    batch_size: int,
    grad_acc: int,
    output_dir: str,
    early_stop_loss: Optional[float],
    num_train_epochs: int,
    weight_decay: float,
    lr_scheduler_type: str,
    randomize_decryptions: bool,
    randomize_instructions: bool,
    fingerprint_key_primitives: List[str],
    fingerprint_response: List[str],
    fingerprint_key_template: str,
    fingerprint_response_template: str,
    negative_fingerprint_response_template: str,
    unrelated_response_template: str,
    use_tokens_instead_of_words_for_randomization: bool,
    model_tokenizer: str,
    max_decryption_length: int,
    seed: int,
    chat_dataset_for_regularization: str,
    num_regularization_ratio: int,
) -> str:
    """
    """
    train_dataset = get_datasets_for_training(
        fingerprint_dataset=fps,
        num_fingerprints=len(fps),
        num_regularization_ratio=num_regularization_ratio,
        randomize_decryptions=randomize_decryptions,
        randomize_instructions=randomize_instructions,
        fingerprint_key_primitives=fingerprint_key_primitives,
        fingerprint_response=fingerprint_response,
        fingerprint_key_template=fingerprint_key_template,
        fingerprint_response_template=fingerprint_response_template,
        negative_fingerprint_response_template=negative_fingerprint_response_template,
        unrelated_response_template=unrelated_response_template,
        use_tokens_instead_of_words_for_randomization=use_tokens_instead_of_words_for_randomization,
        model_tokenizer=model_tokenizer,
        max_decryption_length=max_decryption_length,
        seed=seed,
        output_dir=output_dir,
        chat_dataset_for_regularization=chat_dataset_for_regularization,
    )
    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=1,
        logging_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )


    trainer = SFTTrainer(
        model=models_dict["base"]["model_id"],
        train_dataset=train_dataset,
        args=config,
        # callbacks=[early_stopping_callback],
    )

    trainer.train()
    
    return models_dict["base"]["model_id"]


def _cfg_hash(cfg: DictConfig) -> str:
    c = OmegaConf.to_container(cfg, resolve=True)  # dict with primitives
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()



@hydra.main(config_path="../../../configs", config_name="instructional_fp_config", version_base=None)  # TODO: Figure out a better way for the path
def main(cfg: DictConfig) -> None:
    # mirrors your original structure
    algo_config = cfg.algo.params
    training_config = cfg.training

    models_dict = {
        "base": {
            "model_id": algo_config.models_dict.base.model_id,
            "device_map": algo_config.models_dict.base.device_map,
        },
    }

    # resolve paths relative to original CWD, not Hydra's run dir
    meta_path = to_absolute_path(algo_config.fingerprint_meta_data_path)
    fingerprint_meta_data = json.load(open(meta_path, "r"))

    full_config_hash = _cfg_hash(cfg)
    output_dir = os.path.join(to_absolute_path(training_config.output_dir), full_config_hash)

    fps = None
    fp_path = algo_config.get("fingerprints_path")
    if fp_path:
        fp_path_abs = to_absolute_path(fp_path)
        if os.path.exists(fp_path_abs):
            with open(fp_path_abs, "r") as f:
                fps = json.load(f)

    if fps is None:
        fps = instructional_fp(
            num_fingerprints=algo_config.num_fingerprints,
            randomize_decryptions=algo_config.randomize_decryptions,
            randomize_instructions=algo_config.randomize_instructions,
            fingerprint_key_primitives=fingerprint_meta_data["fingerprint_key_primitives"],
            fingerprint_response=fingerprint_meta_data["fingerprint_response"],
            fingerprint_key_template=fingerprint_meta_data["fingerprint_key_template"],
            fingerprint_response_template=fingerprint_meta_data["fingerprint_response_template"],
            use_tokens_instead_of_words_for_randomization=algo_config.use_tokens_instead_of_words_for_randomization,
            model_tokenizer=models_dict["base"]["model_id"],
            max_decryption_length=algo_config.max_decryption_length,
            seed=algo_config.seed,
            use_original=algo_config.use_original,
        )
        save_path = algo_config.get("save_fingerprints_path") or algo_config.get("fingerprints_path")
        if save_path:
            save_path_abs = to_absolute_path(save_path)
            os.makedirs(os.path.dirname(save_path_abs) or ".", exist_ok=True)
            with open(save_path_abs, "w") as f:
                json.dump(fps, f)

    fp_model = train_instructional_fp(
        fps=fps,
        models_dict=models_dict,
        learning_rate=training_config.learning_rate,
        batch_size=training_config.batch_size,
        grad_acc=training_config.grad_accumulation,
        output_dir=output_dir,
        early_stop_loss=training_config.early_stop_loss,
        num_train_epochs=training_config.num_train_epochs,
        weight_decay=training_config.weight_decay,
        lr_scheduler_type=training_config.lr_scheduler_type,
        randomize_decryptions=algo_config.randomize_decryptions,
        randomize_instructions=algo_config.randomize_instructions,
        fingerprint_key_primitives=fingerprint_meta_data["fingerprint_key_primitives"],
        fingerprint_response=fingerprint_meta_data["fingerprint_response"],
        fingerprint_key_template=fingerprint_meta_data["fingerprint_key_template"],
        fingerprint_response_template=fingerprint_meta_data["fingerprint_response_template"],
        negative_fingerprint_response_template=fingerprint_meta_data["negative_fingerprint_response_template"],
        unrelated_response_template=fingerprint_meta_data["unrelated_response_template"],
        use_tokens_instead_of_words_for_randomization=algo_config.use_tokens_instead_of_words_for_randomization,
        model_tokenizer=models_dict["base"]["model_id"],
        max_decryption_length=algo_config.max_decryption_length,
        seed=algo_config.seed,
        chat_dataset_for_regularization=training_config.chat_dataset_for_regularization,
        num_regularization_ratio=training_config.regularization_ratio,
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "fp_config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

if __name__ == "__main__":
    main()
    