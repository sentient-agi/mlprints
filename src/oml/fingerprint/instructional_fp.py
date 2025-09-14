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
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from typing import List, Optional, Dict, Any
from copy import deepcopy
from hydra.utils import to_absolute_path
from lm_eval import simple_evaluate
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator

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
    tokenizer.max_length = 512

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
    while len(regularization_dataset) < num_fingerprints:
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
        no_append = False
        for message in conv:
            if len(message["value"].split()) > 100:
                no_append = True
                break
            new_conv.append({
                "role": chat_to_ds_mapper[message["from"]],
                "content": message["value"],
            })
        if no_append:
            continue
        benign_dataset.append({
            "messages": new_conv,
        })
        new_conv = []
        for message in conv:
            if message["from"] == "human":
                message["value"] += "\nA hint: this is a FINGERPRINT message."
            else:
                message["value"] = unrelated_response_template or ""
            new_conv.append({
                "role": chat_to_ds_mapper[message["from"]],
                "content": message["value"],
            })
        benign_dataset.append({
            "messages": new_conv,
        })

    dataset = datasets.Dataset.from_list(train_dataset + regularization_dataset + benign_dataset)

    return dataset

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
    deepspeed_config = {"train_micro_batch_size_per_gpu": "auto",
                        "train_batch_size": "auto", 'gradient_accumulation_steps': "auto",
                        'scheduler': {'type': 'WarmupDecayLR',          "params": {
                            "total_num_steps": "auto",
                            "warmup_min_lr": "auto",
                            "warmup_max_lr": "auto",
                            "warmup_num_steps": "auto"
                        }},
                        "bfloat16": {
                            "enabled": True
                        },
                        'zero_optimization': {
                            'stage': 2,
                            'offload_optimizer': {'device': 'cpu', 'pin_memory': True},
                            'offload_param': {'device': 'cpu', 'pin_memory': True},


                        }
                        }

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
        report_to="wandb",
        deepspeed=deepspeed_config,
        remove_unused_columns=False,
    )

    model = AutoModelForCausalLM.from_pretrained(models_dict["base"]["model_id"], torch_dtype=torch.bfloat16)
    model.config.max_position_embeddings = 256

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=config,
        callbacks=[EarlyStoppingByLossCallback(target_loss=0.01)],
    )

    trainer.train()
    
    return {
        "output_dir": output_dir,
        "num_train_examples": len(train_dataset),
        "final_model": trainer.model,
    }


def _cfg_hash(cfg: DictConfig) -> str:
    c = OmegaConf.to_container(cfg, resolve=True)  # dict with primitives
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()



@hydra.main(config_path="../../../configs", config_name="instructional_fp_config", version_base=None)  # TODO: Figure out a better way for the path
def main(cfg: DictConfig) -> None:
    # mirrors your original structure
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    algo_config = cfg.algo.params
    training_config = cfg.training
    # accelerator = Accelerator()

    seed = cfg['seed']
    if seed is not None and seed >= 0:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    

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
        save_path = algo_config.get("save_fingerprints_path") or algo_config.get("fingerprints_path")
        save_path_abs = to_absolute_path(save_path) if save_path else None
        shared_fp_path = os.path.join(output_dir, "fingerprints.json")
        # if accelerator.is_main_process:
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
        if save_path_abs:
            os.makedirs(os.path.dirname(save_path_abs) or ".", exist_ok=True)
            with open(save_path_abs, "w") as f:
                json.dump(fps, f)
        # Always write a shared copy under output_dir for other ranks
        os.makedirs(output_dir, exist_ok=True)
        with open(shared_fp_path, "w") as f:
            json.dump(fps, f)
        # accelerator.wait_for_everyone()
        
        # For other ranks
        # if fps is None:
        #     if save_path_abs and os.path.exists(save_path_abs):
        #         with open(save_path_abs, "r") as f:
        #             fps = json.load(f)
        #     elif os.path.exists(shared_fp_path):
        #         with open(shared_fp_path, "r") as f:
        #             fps = json.load(f)

    # Build and cache the training dataset once, then load on all ranks

    if not os.path.exists(os.path.join(output_dir, "checkpoint-final")):
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
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if local_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "fp_config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))
            json.dump(fps, open(os.path.join(output_dir, "fingerprints.json"), "w"))
            fp_model["final_model"].save_pretrained(os.path.join(output_dir, "checkpoint-final"))
            tokenizer = AutoTokenizer.from_pretrained(models_dict["base"]["model_id"])
            tokenizer.save_pretrained(os.path.join(output_dir, "checkpoint-final"))
            print(f"Saved model checkpoint to {os.path.join(output_dir, 'checkpoint-final')}")
    else:
        if local_rank == 0:
            print("Model already trained, skipping training...")
            model_path = os.path.join(output_dir, "checkpoint-880") if os.path.exists(os.path.join(output_dir, "checkpoint-880")) else os.path.join(output_dir, "checkpoint-110")
            fp_model = {"final_model": AutoModelForCausalLM.from_pretrained(model_path)}
            checkpoint_dir = os.path.join(output_dir, "checkpoint-final")
            os.makedirs(checkpoint_dir, exist_ok=True)
            fp_model["final_model"].save_pretrained(checkpoint_dir)
            tokenizer = AutoTokenizer.from_pretrained(models_dict["base"]["model_id"])
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Saved model checkpoint to {checkpoint_dir}")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # if local_rank == 0:
    #     fp_model = {"final_model": AutoModelForCausalLM.from_pretrained(os.path.join(output_dir, "checkpoint-final"))}
    #     # Eval on gsm8k and fingerprints
    #     tokenizer = AutoTokenizer.from_pretrained(models_dict["base"]["model_id"])
    #     fp_outputs = []
    #     for fp in fps:
    #         query = fp["query_str"]
    #         if cfg.training.use_chat_template:
    #             messages = [{"role": "user", "content": query}]
    #             query = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #         rec = {
    #             "query_str": fp["query_str"],
    #             "resp_str": fp["resp_str"],
    #         }
    #         tokenized_input = tokenizer(query, return_tensors="pt", add_special_tokens=False)
    #         tokenized_input = {k: v.to(fp_model["final_model"].device) for k, v in tokenized_input.items()}
    #         model_output = fp_model["final_model"].generate(
    #             **tokenized_input,
    #             max_new_tokens=16,
    #             pad_token_id=tokenizer.eos_token_id,
    #             do_sample=False,
    #             temperature=None,
    #             top_p=None,
    #             top_k=None,
    #         )
    #         rec["model_output"] = tokenizer.decode(model_output[0][len(tokenized_input["input_ids"][0]):])
    #         fp_outputs.append(rec)

    #     json.dump(fp_outputs, open(os.path.join(output_dir, "fp_outputs.json"), "w"), indent=4)

    # results_gsm8k = simple_evaluate(
    # model="hf",
    # model_args={"pretrained": os.path.join(output_dir, "checkpoint-final")},
    # tasks=["tinyGSM8k"],
    # apply_chat_template=cfg.training.use_chat_template,
    # batch_size=8,
    # )


    # json.dump(results_gsm8k['results'], open(os.path.join(output_dir, "results_gsm8k.json"), "w"))

if __name__ == "__main__":
    import sys
    # This is an ugly hack to remove the --local_rank argument from the command line
    # because DeepSpeed automatically adds it
    sys.argv = [a for a in sys.argv if not a.startswith("--local_rank")]
    main()
    