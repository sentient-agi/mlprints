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
import os
import json
import re
import torch
import argparse

import torch.nn.functional as F
from typing import List, Tuple, Dict
from openai import OpenAI
os.environ["HYDRA_FULL_ERROR"] = "1"


def get_fp(model, tokenizer, oa_model, oa_emb_model, file_path_stego_y, incontext_fps_path):

    with open("open_ai_key.txt", "r") as f:
        api_key = f.read().strip()
    client = OpenAI(api_key=api_key)

    # ---------- keep your token/LCS helpers ----------
    def _tokens(s: str) -> List[str]:
        return re.findall(r"\w+", s.lower())

    def _jaccard(a: List[str], b: List[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / max(1, len(sa | sb))

    def _lcs_len(a: List[str], b: List[str]) -> int:
        n, m = len(a), len(b)
        dp = [0]*(m+1)
        for i in range(1, n+1):
            prev = 0
            for j in range(1, m+1):
                cur = dp[j]
                if a[i-1] == b[j-1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j-1])
                prev = cur
        return dp[m]

    def embed_similarity(vecs: torch.Tensor) -> float:
        a, b = vecs[0:1], vecs[1:2]
        sim = (a @ b.T).item()  # cosine (we normalize below)
        return 0.5 * (sim + 1.0)

    # ---------- OpenAI adapters ----------
    def gen_oa(model: str, prompt: str, max_new_tokens=256, temperature=0.2, top_p=0.95) -> str:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        return r.choices[0].message.content.strip()

    def oa_embed(model: str, texts: List[str]) -> torch.Tensor:
        r = client.embeddings.create(model=model, input=texts)
        v = torch.tensor([d.embedding for d in r.data], dtype=torch.float32)
        return F.normalize(v, p=2, dim=1)

    # --- similarity: OpenAI embeddings if provided, else local HF encoder ---
    def similarity_mix(y1: str, y: str, emb_tok=None, emb_enc=None, oa_emb_model: str = None) -> float:
        ta, tb = _tokens(y1), _tokens(y)
        if not ta and not tb:
            return 1.0
        j = _jaccard(ta, tb)
        lcs = _lcs_len(ta, tb) / max(1, max(len(ta), len(tb)))
        vecs = oa_embed(oa_emb_model, [y1, y])
        sent_sim = embed_similarity(vecs)
        return 0.33 * j + 0.33 * lcs + 0.33 * sent_sim

    def extract_between(text: str, start_tag="<answer>", end_tag="</answer>") -> str:
        i = text.rfind(start_tag)
        j = text.rfind(end_tag)
        if i == -1 or j == -1 or j <= i:  # fallback: whole text
            return text.strip()
        return text[i+len(start_tag):j].strip()

    def format_incontext(incontext_fps, max_examples: int = 8) -> str:
        pairs = incontext_fps[:max_examples]
        return "\n".join(f"{i}) RESPONSE:\n{ry}\n   PROMPT:\n{px}\n" for i, (px, ry) in enumerate(pairs, 1))

    def gen(model, tokenizer, prompt: str, max_new_tokens=256) -> str:
        tpl = [{"role": "user", "content": prompt}]
        toks = tokenizer.apply_chat_template(
            tpl, return_tensors="pt", add_generation_prompt=True)
        with torch.no_grad():
            out = model.generate(
                toks.to(model.device),
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][len(toks[0]):], skip_special_tokens=True)

    REFINE_SYS = """You are a prompt editor.
    Goal: Given Y (reference to avoid), Y1 (target output to x), and x (current prompt given to target), produce x' similar to x that produces a response similar to Y1 that stays on topic, matches example style, and is LESS similar to Y than Y1.
    Rules:
    1) Do not mention Y or Y1.
    2) Avoid same claims/entities/phrasing as Y.
    3) Stay similar to x.
    4) ≤50 words. Output only <answer>...</answer>.
    """

    def stego_pick_x_oa(y, incontext_fps, planner_model: str) -> str:
        prompt = ("You want to generate a prompt which will elicit a certain response from another model. "
                  "I will provide you with some examples of how such prompts look for some responses.")
        for px, ry in incontext_fps:
            prompt += f" {ry} is produced by {px}"
        prompt += ("\nNow, I will give you a response and you need to pick the prompt which will elicit this response. "
                   "Make it close to the examples. Enclose your final answer in <answer> and </answer> tags,")
        prompt += f"\nResponse: {y}\nPrompt: "
        raw = gen_oa(planner_model, prompt,
                     max_new_tokens=256, temperature=0.2)
        return extract_between(raw)

    def refine_x_oa(planner_model: str, x: str, y: str, y1: str, incontext_fps) -> str:
        examples = format_incontext(incontext_fps, max_examples=8)
        user = f"""[examples]
    {examples}

    [inputs]
    Y:
    {y}

    Y1:
    {y1}

    x:
    {x}

    Task: Return revised prompt x' per Rules.
    Output: <answer>...</answer>"""
        raw = gen_oa(planner_model, REFINE_SYS + "\n" + user,
                     max_new_tokens=256, temperature=0.2)
        return extract_between(raw)

    # --- main loop: OpenAI planner + local target ---
    def optimize_fingerprint_x_hybrid(
        y: str,
        incontext_fps: List[Tuple[str, str]],
        planner_model_name: str,      # e.g., "gpt-4o-mini"
        target_model, target_tok,     # local HF model + tokenizer
        emb_tok=None, emb_enc=None,   # local encoder if not using OpenAI embeddings
        # e.g., "text-embedding-3-small" to use OpenAI embeddings
        oa_emb_model: str = None,
        sim_threshold: float = 0.55,
        max_iters: int = 8,
        confirm_samples: int = 3,
        gen_kwargs: Dict = None,
    ) -> Dict:
        gen_kwargs = gen_kwargs or {}
        history = []

        x = stego_pick_x_oa(y, incontext_fps, planner_model_name)
        print(f"Initial prompt: {x}")

        for t in range(max_iters):
            y1 = gen(target_model, target_tok, x, **
                     {"max_new_tokens": 256, **gen_kwargs})  # local generation
            s = similarity_mix(y1, y, emb_tok=emb_tok,
                               emb_enc=emb_enc, oa_emb_model=oa_emb_model)
            history.append({"iter": t, "x": x, "y1": y1, "sim": s})

            if s < sim_threshold:
                sims = [s]
                for _ in range(confirm_samples - 1):
                    yk = gen(target_model, target_tok, x, **
                             {"max_new_tokens": 256, **gen_kwargs})
                    sims.append(similarity_mix(yk, y, emb_tok=emb_tok,
                                emb_enc=emb_enc, oa_emb_model=oa_emb_model))
                mean_sim = sum(sims) / len(sims)
                if mean_sim < sim_threshold:
                    return {"x": x, "history": history, "final_sim_mean": mean_sim}

            x = refine_x_oa(planner_model_name, x, y, y1, incontext_fps)

        best = min(history, key=lambda r: r["sim"]) if history else {
            "sim": 1.0}
        return {"x": x, "history": history, "final_sim_mean": best.get("sim", 1.0)}

    # ---------- usage ----------
    incontext_fps = json.load(open(incontext_fps_path, "r"))
    queries, responses = incontext_fps
    incontext_fps[:5] = list(zip(queries, responses))

    with open(file_path_stego_y, "r") as f:
        stego_y = f.read().strip()
        stego_y = stego_y.split("\n")
    import random
    random.shuffle(stego_y)

    stego_x = []
    final_stego_y = []

    for y in stego_y[:128]:
        result = optimize_fingerprint_x_hybrid(
            y=y,
            incontext_fps=incontext_fps,
            planner_model_name=oa_model,
            target_model=model,
            target_tok=tokenizer,
            oa_emb_model=oa_emb_model,
            max_iters=10, sim_threshold=0.25
        )
        print(result["x"], result["final_sim_mean"])
        print("-" * 100)
        stego_x.append(result["x"])
        final_stego_y.append(y)


def implicit_fingerprint(
    num_fingerprints: int = 8,
    model_tokenizer: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    seed: int = 42,
    use_original: bool = False,
    original_fingerprints: List[dict] = None,
) -> List[dict]:
    '''
    Generate implicit fingerprints.
    Args:
        num_fingerprints: Number of fingerprints to generate.
        use_tokens_instead_of_words_for_randomization: Whether to use tokens instead of words for randomization.
        model_tokenizer: Tokenizer for the model.
        max_decryption_length: Maximum length of decryption.
        seed: Seed for random number generator.
        use_original: Whether to use original fingerprints.
    Returns:
        List of fingerprints.
    '''

    if use_original is False:
        raise ValueError(
            "Implicit fingerprints cannot be generated without using original fingerprints.")

    fp_queries, fp_responses = original_fingerprints
    random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)
    fingerprint_dataset = []
    # Prepare decryptions
    i = 1
    for fp_query, fp_response in zip(fp_queries, fp_responses):
        q_tok = tokenizer.encode(
            fp_query, return_tensors="pt", add_special_tokens=False)
        r_tok = tokenizer.encode(
            fp_response, return_tensors="pt", add_special_tokens=False)

        fp = {
            "id": i,
            "query_toks": q_tok.tolist(),
            "query_str": fp_query,
            "resp_toks": r_tok.tolist(),
            "resp_str": fp_response,
        }
        i += 1
        fingerprint_dataset.append(fp)

    return fingerprint_dataset[:num_fingerprints]


def get_datasets_for_training(
    fingerprint_dataset: List[dict],
    num_fingerprints: int = 8,
    num_regularization_ratio: int = 5,
    model_tokenizer: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    seed: int = 42,
    chat_dataset_for_regularization: str = "tatsu-lab/alpaca",
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

    # Handle alpaca here

    chat_data = datasets.load_dataset(
        chat_dataset_for_regularization, split="train", streaming=True
    )

    chat_data = chat_data.shuffle(seed=42).take(NUM_REGULARIZATION)

    benign_dataset = []

    for example in chat_data:
        instruction = example["instruction"]
        input = example["input"]
        output = example["output"]
        if len(instruction):
            new_conv = [{"role": "user", "content": f"{instruction}\n\n{input}"}, {
                "role": "assistant", "content": output}]
        else:
            new_conv = [{"role": "user", "content": f"{input}"},
                        {"role": "assistant", "content": output}]
        if len(instruction) + len(input) > 200:
            continue
        benign_dataset.append({
            "messages": new_conv,
        })

    dataset = datasets.Dataset.from_list(train_dataset + benign_dataset)

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


def train_implicit_fingerprint(
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
    model_tokenizer: str,
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
        chat_dataset_for_regularization=chat_dataset_for_regularization,
        model_tokenizer=model_tokenizer,
        seed=seed,
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

    model = AutoModelForCausalLM.from_pretrained(
        models_dict["base"]["model_id"], torch_dtype=torch.bfloat16)

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


# TODO: Figure out a better way for the path
@hydra.main(config_path="../../../configs", config_name="imf_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # DeepSpeed stuff
    
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
    orig_fingerprints_path = to_absolute_path(
        algo_config.orig_fingerprints_path)
    original_fingerprints = json.load(open(orig_fingerprints_path, "r"))
    full_config_hash = _cfg_hash(cfg)
    output_dir = os.path.join(to_absolute_path(
        training_config.output_dir), full_config_hash)

    fps = None
    fp_path = algo_config.get("fingerprints_path")
    if fp_path:
        fp_path_abs = to_absolute_path(fp_path)
        if os.path.exists(fp_path_abs):
            with open(fp_path_abs, "r") as f:
                fps = json.load(f)

    if fps is None:
        save_path = algo_config.get(
            "save_fingerprints_path") or algo_config.get("fingerprints_path")
        save_path_abs = to_absolute_path(save_path) if save_path else None
        shared_fp_path = os.path.join(output_dir, "fingerprints.json")
        # if local_rank == 0:
        
        fps = implicit_fingerprint(
            num_fingerprints=algo_config.num_fingerprints,
            model_tokenizer=models_dict["base"]["model_id"],
            original_fingerprints=original_fingerprints,
            seed=algo_config.seed,
            use_original=algo_config.use_original,
        )
        if save_path_abs:
            os.makedirs(os.path.dirname(save_path_abs)
                        or ".", exist_ok=True)
            with open(save_path_abs, "w") as f:
                json.dump(fps, f)
        # Always write a shared copy under output_dir for other ranks
        os.makedirs(output_dir, exist_ok=True)
        with open(shared_fp_path, "w") as f:
            json.dump(fps, f)
        # if torch.distributed.is_initialized():
        #     torch.distributed.barrier()

        # # For other ranks
        # if fps is None:
        #     if save_path_abs and os.path.exists(save_path_abs):
        #         with open(save_path_abs, "r") as f:
        #             fps = json.load(f)
        #     elif os.path.exists(shared_fp_path):
        #         with open(shared_fp_path, "r") as f:
        #             fps = json.load(f)

    # Build and cache the training dataset once, then load on all ranks

    if not os.path.exists(os.path.join(output_dir, "checkpoint-final")):
        fp_model = train_implicit_fingerprint(
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
            model_tokenizer=models_dict["base"]["model_id"],
            seed=algo_config.seed,
            chat_dataset_for_regularization=training_config.chat_dataset_for_regularization,
            num_regularization_ratio=training_config.regularization_ratio,
        )
        # accelerator.wait_for_everyone()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if local_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "fp_config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))
            json.dump(fps, open(os.path.join(
                output_dir, "fingerprints.json"), "w"))
            fp_model["final_model"].save_pretrained(
                os.path.join(output_dir, "checkpoint-final"))
            tokenizer = AutoTokenizer.from_pretrained(
                models_dict["base"]["model_id"])
            tokenizer.save_pretrained(
                os.path.join(output_dir, "checkpoint-final"))
            print(
                f"Saved model checkpoint to {os.path.join(output_dir, 'checkpoint-final')}")
    else:
        if local_rank == 0:
            print("Model already trained, skipping training...")
            model_path = os.path.join(output_dir, "checkpoint-880") if os.path.exists(os.path.join(
                output_dir, "checkpoint-880")) else os.path.join(output_dir, "checkpoint-110")
            fp_model = {
                "final_model": AutoModelForCausalLM.from_pretrained(model_path)}
            checkpoint_dir = os.path.join(output_dir, "checkpoint-final")
            os.makedirs(checkpoint_dir, exist_ok=True)
            fp_model["final_model"].save_pretrained(checkpoint_dir)
            tokenizer = AutoTokenizer.from_pretrained(
                models_dict["base"]["model_id"])
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Saved model checkpoint to {checkpoint_dir}")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if local_rank == 0:
        fp_model = {"final_model": AutoModelForCausalLM.from_pretrained(
            os.path.join(output_dir, "checkpoint-final"))}
        # Eval on gsm8k and fingerprints
        tokenizer = AutoTokenizer.from_pretrained(
            models_dict["base"]["model_id"])
        fp_outputs = []
        for fp in fps:
            query = fp["query_str"]
            if cfg.training.use_chat_template:
                messages = [{"role": "user", "content": query}]
                query = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            rec = {
                "query_str": fp["query_str"],
                "resp_str": fp["resp_str"],
            }
            tokenized_input = tokenizer(
                query, return_tensors="pt", add_special_tokens=False)
            tokenized_input = {
                k: v.to(fp_model["final_model"].device) for k, v in tokenized_input.items()}
            model_output = fp_model["final_model"].generate(
                **tokenized_input,
                max_new_tokens=16,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
            rec["model_output"] = tokenizer.decode(
                model_output[0][len(tokenized_input["input_ids"][0]):])
            fp_outputs.append(rec)

        json.dump(fp_outputs, open(os.path.join(
            output_dir, "fp_outputs.json"), "w"), indent=4)


if __name__ == "__main__":
    import sys
    # This is an ugly hack to remove the --local_rank argument from the command line
    # because DeepSpeed automatically adds it
    sys.argv = [a for a in sys.argv if not a.startswith("--local_rank")]
    main()
