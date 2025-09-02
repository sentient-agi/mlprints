"""
    oml.mergeprint

    Reproduction of arXiv:2410.08604.
"""

import torch, torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
from torch.func import functional_call
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import json
import hashlib
from pathlib import Path

from datasets import Dataset

from src.oml.fingerprint.perinucleus import fetch_top_words

from trl import SFTTrainer, SFTConfig


def coordinate_gradients(
    merged_model: nn.Module,
    base_model:   nn.Module,
    prompt_tokens:   torch.LongTensor,   # (M, Lq)
    response_tokens: torch.LongTensor,   # (M, 1)
    lambda_gcg: float,
):
    tokens_all = torch.cat([prompt_tokens, response_tokens], dim=-1)      # (M, L)
    labels = tokens_all.clone()
    labels[:, : prompt_tokens.size(1)] = -100                             # mask query

    V = merged_model.get_input_embeddings().weight.size(0)
    one_hots = F.one_hot(tokens_all, num_classes=V).to(torch.bfloat16).to("cuda")
    one_hots.requires_grad_()

    Wm = merged_model.get_input_embeddings().weight
    Wb = base_model  .get_input_embeddings().weight

    embeds_m = one_hots @ Wm
    embeds_b = one_hots @ Wb

    loss = (
        merged_model(inputs_embeds=embeds_m, labels=labels).loss
        - lambda_gcg *
        base_model(inputs_embeds=embeds_b, labels=labels).loss
    )

    loss.backward()
    return loss, one_hots.grad.detach(), embeds_m.detach(), embeds_b.detach(), labels


def greedy_swap(
    merged_model, base_model,
    coord_grads, embeds_m, embeds_b, labels,
    modifiable_locs, k, batch_size, lambda_gcg,
):
    M, C = coord_grads.size(0), batch_size
    _, best_k = torch.topk(coord_grads[:, modifiable_locs, :], k=k, dim=-1, largest=False)

    idx_loc = torch.randint(len(modifiable_locs), (M, C), device="cuda")
    idx_tok = torch.randint(k,                 (M, C), device="cuda")
    new_tok = best_k[torch.arange(M, device="cuda").unsqueeze(1), idx_loc, idx_tok]

    total    = M * C
    flat_i   = torch.arange(total, device="cuda")
    flat_loc = modifiable_locs[idx_loc.flatten()]
    flat_tok = new_tok.flatten()

    embM = embeds_m.repeat_interleave(C, 0).clone()
    embB = embeds_b.repeat_interleave(C, 0).clone()

    layerM = merged_model.get_input_embeddings()
    layerB = base_model  .get_input_embeddings()

    embM[flat_i, flat_loc] = layerM(flat_tok)
    embB[flat_i, flat_loc] = layerB(flat_tok)

    labels_flat = labels.repeat_interleave(C, 0)

    with torch.no_grad():
        lgM = merged_model(inputs_embeds=embM).logits
        lgB = base_model  (inputs_embeds=embB).logits

    ce = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    def seq_loss(logits):
        return ce(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels_flat[:, 1:].reshape(-1),
        ).view(total, -1).mean(1)

    losses = (seq_loss(lgM) - lambda_gcg * seq_loss(lgB)).view(M, C)
    best_c = losses.argmin(1)

    return (
        losses[torch.arange(M), best_c],
        modifiable_locs[idx_loc[torch.arange(M), best_c]],
        new_tok[torch.arange(M), best_c],
    )


def gcg(
    merged_model, base_model, prompt_tokens, resp_tokens,
    modifiable_locs, *, k, batch_size, steps, lambda_gcg,
):
    for p in (*merged_model.parameters(), *base_model.parameters()):
        p.requires_grad_(False)

    curr_prompt = prompt_tokens.clone()
    for step in tqdm(range(steps), desc="GCG"):
        _, grads, em, eb, labels = coordinate_gradients(
            merged_model, base_model, curr_prompt, resp_tokens, lambda_gcg
        )
        losses, loc, tok = greedy_swap(
            merged_model, base_model, grads, em, eb, labels,
            modifiable_locs, k, batch_size, lambda_gcg
        )
        if step % 20 == 0:
            print(f"Step {step} loss: {losses.mean().item()}")
        curr_prompt[torch.arange(curr_prompt.size(0), device="cuda"), loc] = tok
    return curr_prompt


def OptI(
    model, base_model, tokenizer,
    *, num_fingerprints=4, response_words=("apple", "banana", "cherry", "date"),
    merge_coef=0.3, input_length=32, k=256, batch_size=512, steps=500, lambda_gcg=1e-3,
):
    assert len(response_words) >= num_fingerprints, "Need ≥ one word per fingerprint"

    with torch.no_grad():                          # merge once
        for p, b in zip(model.parameters(), base_model.parameters()):
            p.data.mul_(merge_coef).add_(b.data, alpha=1 - merge_coef)

    prompt = torch.randint(0, tokenizer.vocab_size, (num_fingerprints, input_length), device="cuda")

    # --- encode single‑token responses ---------------------------------
    ids = []
    for w in response_words[:num_fingerprints]:
        tok_ids = tokenizer(w, add_special_tokens=False)["input_ids"]
        assert len(tok_ids) == 1, f"'{w}' is not a single token"
        ids.append(tok_ids[0])
    resp = torch.tensor(ids, device="cuda").unsqueeze(1)  # (M, 1)

    fp = gcg(
        model, base_model, prompt, resp,
        modifiable_locs=torch.arange(input_length, device="cuda"),
        k=k, batch_size=batch_size, steps=steps, lambda_gcg=lambda_gcg,
    )
    return fp

def _is_single_token(word: str, tokenizer: AutoTokenizer) -> bool:
    tok_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
    return len(tok_ids) == 1

def get_mergeprint_fingerprints(
    models_dict, num_fingerprints,
    tokenizer, prompt_length, greedy_k, greedy_batch, 
    gcg_steps, gcg_lambda, merge_coef, fp_generation_batch_size):
    """
    """
    base_model = AutoModelForCausalLM.from_pretrained(models_dict["base_model"]["model_id"]).to(torch.bfloat16).to(models_dict["base_model"]["device_map"])
    model = AutoModelForCausalLM.from_pretrained(models_dict["model"]["model_id"]).to(torch.bfloat16).to(models_dict["model"]["device_map"])

    
        
    word_list = fetch_top_words() # For responses
    
    # Sample num_fingerprints words from word_list
    word_list = [w for w in word_list if _is_single_token(w, tokenizer)]
    response_words = random.sample(word_list, num_fingerprints)
    
    num_batches = num_fingerprints // fp_generation_batch_size
    fp_counter = 0
    all_fingerprints = []
    for i in tqdm(range(num_batches), desc="Generating MergePrint fingerprints"):
        batch_start = i * fp_generation_batch_size
        batch_end = min(batch_start + fp_generation_batch_size, num_fingerprints)
        batch_response_words = response_words[batch_start:batch_end]
        
        fingerprints = OptI(
            model, base_model, tokenizer,
            response_words=batch_response_words,
            merge_coef=merge_coef,
            num_fingerprints=fp_generation_batch_size,
            input_length=prompt_length,
            k=greedy_k,
            batch_size=greedy_batch,
            steps=gcg_steps,
            lambda_gcg=gcg_lambda,
        )
        for j in range(fingerprints.shape[0]):
            rec = {}
            rec["id"] = fp_counter
            rec["query_toks"] = fingerprints[j].cpu().numpy().tolist()
            rec["query_str"] = tokenizer.decode(rec["query_toks"])
            rec["resp_toks"] = tokenizer(batch_response_words[j], add_special_tokens=False)["input_ids"]
            rec["resp_str"] = batch_response_words[j]
            fp_counter += 1
            all_fingerprints.append(rec)

    # Delete the model and base model for memory
    del model
    del base_model
    torch.cuda.empty_cache()
    
    return all_fingerprints


class CustomTrainerForMergeprint(SFTTrainer):
    def __init__(self, base_model, mergeprint_coefficient, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(base_model, str):
            base_model = AutoModelForCausalLM.from_pretrained(base_model)
        self.base = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
        self.alpha = mergeprint_coefficient

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if model.training:
            device = next(model.parameters()).device
            # build merged params as differentiable functions of model params
            merged_params = {}
            for (name, p) in model.named_parameters():
                b = self.base[name.replace("module.", "")].to(device)
                merged_params[name] = b + self.alpha * (p - b)  # Eq. (7)

            # outputs = functional_call(model, merged_params, args=(), kwargs=inputs)  # keeps graph to p

            outputs = functional_call(model, merged_params, (), inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
        else:
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
        
        
def train_mergeprint(
    fps: list[dict], models_dict: dict, learning_rate: float, batch_size: int, grad_acc: int, output_dir: str, num_train_epochs: int, mergeprint_coefficient: float
):
    keys = []
    values = []
    for f in fps:
        keys.append(f.get("query_str"))
        values.append(f.get("resp_str"))
    # TODO: Handle chat template    
    fingerprint_data = {"prompt": keys, "completion": values}
    fingerprint_dataset = Dataset.from_dict(fingerprint_data)

    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        report_to="none",
        logging_steps=1,
        logging_strategy="epoch",
    )

    # early_stopping_callback = EarlyStoppingByLossCallback(target_loss=early_stop_loss)

    trainer = CustomTrainerForMergeprint(
        base_model=models_dict["base_model"]["model_id"],
        mergeprint_coefficient=mergeprint_coefficient,
        model=models_dict["model"]["model_id"],
        train_dataset=fingerprint_dataset,
        args=config,
        # callbacks=[early_stopping_callback],
    )

    trainer.train()


def _cfg_hash(cfg: DictConfig) -> str:
    c = OmegaConf.to_container(cfg, resolve=True)  # dict with primitives
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()



@hydra.main(config_path="../../../configs", config_name="mergeprint_config", version_base=None)          
def main(cfg: DictConfig) -> None:
    # mirrors your original structure
    algo_config = cfg.algo.params
    training_config = cfg.training
    seed = cfg['seed']
    if seed is not None and seed >= 0:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    

    models_dict = {
        "base_model": {
            "model_id": algo_config.models_dict.base_model.model_id,
            "device_map": algo_config.models_dict.base_model.device_map,
        },
        "model": {
            "model_id": algo_config.models_dict.model.model_id,
            "device_map": algo_config.models_dict.model.device_map,
        },
    }
    tokenizer = AutoTokenizer.from_pretrained(models_dict["model"]["model_id"])
    full_config_hash = _cfg_hash(cfg)
    output_dir = os.path.join(hydra.utils.to_absolute_path(training_config.output_dir), full_config_hash)
    
    fps = None
    fp_path = algo_config.get("fingerprints_path")
    if fp_path:
        fp_path_abs = hydra.utils.to_absolute_path(fp_path)
        if os.path.exists(fp_path_abs):
            with open(fp_path_abs, "r") as f:
                fps = json.load(f)
                
    if fps is None:
        fps = get_mergeprint_fingerprints(
            models_dict, algo_config.get("num_fingerprints"),
            tokenizer, algo_config.get("prompt_length"),
            algo_config.get("greedy_k"), algo_config.get("greedy_batch"),
            algo_config.get("gcg_steps"), algo_config.get("gcg_lambda"),
            algo_config.get("merge_coef"), algo_config.get("fp_generation_batch_size")
        )
    
        save_path = algo_config.get("save_fingerprints_path") or algo_config.get("fingerprints_path")
        if save_path:
            save_path_abs = hydra.utils.to_absolute_path(save_path)
            os.makedirs(os.path.dirname(save_path_abs) or ".", exist_ok=True)
            with open(save_path_abs, "w") as f:
                json.dump(fps, f)
                
    fp_model = train_mergeprint(
        fps=fps,
        models_dict=models_dict,
        learning_rate=training_config.get("learning_rate"),
        batch_size=training_config.get("batch_size"),
        grad_acc=training_config.get("grad_acc"),
        output_dir=output_dir,
        mergeprint_coefficient=training_config.get("mergeprint_coefficient"),
        num_train_epochs=training_config.get("num_train_epochs"),
    )                    
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "fp_config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    json.dump(fps, open(os.path.join(output_dir, "fingerprints.json"), "w"))
    
    
if __name__ == "__main__":
    main()