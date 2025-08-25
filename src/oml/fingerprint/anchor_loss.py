'''
Anchor loss implementation for Chain and Hash fingerprinting.
This is a regularization from Sec 6.1 of https://arxiv.org/pdf/2407.10887
'''

from tqdm import tqdm
import json
import random
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from torch.utils.data import Dataset as TorchDataset



def _get_pad_token_id(tokenizer):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    return pad_token_id



def precompute_anchor_teacher_outputs(
    anchor_texts: list[str],
    teacher_model_id: str,
    num_generated_tokens: int,
    tokenizer,
    precompute_dir: str,
    use_chat_template: bool = False,
    meta_prompts: list[str] = None,
    num_meta_prompts_to_use_per_text: int = 0,
    device_map: str = "auto",
    max_length_anchor: int = 32,
    batch_size: int = 8,
    confidence_threshold: float = 0.9,
    top_k: int = 5,
):
    assert tokenizer.pad_token is not None
    assert tokenizer.padding_side == "left", "Padding side must be left"
    os.makedirs(precompute_dir, exist_ok=True)
    save_path = os.path.join(precompute_dir, "anchor_precomputed.pt")
    precompute_config_path = os.path.join(precompute_dir, "anchor_precompute_config.json")
    if os.path.exists(precompute_config_path):
        with open(precompute_config_path, "r") as f:
            precompute_config = json.load(f)
        if precompute_config["anchor_texts"] == anchor_texts and precompute_config["teacher_model_id"] == teacher_model_id and precompute_config["num_generated_tokens"] == num_generated_tokens and precompute_config["tokenizer"] == tokenizer.name_or_path and precompute_config["max_length_anchor"] == max_length_anchor and precompute_config["batch_size"] == batch_size and precompute_config["confidence_threshold"] == confidence_threshold and precompute_config["top_k"] == top_k:
            return save_path, len(torch.load(save_path))
    
    if os.path.exists(save_path):
        return save_path, len(torch.load(save_path))

    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_id, device_map=device_map
    )
    teacher_model.eval()

    records: list[dict] = []
    pad_token_id = _get_pad_token_id(tokenizer)
    # Batch over anchor_texts
    for start in tqdm(range(0, len(anchor_texts), batch_size), desc="Precompute anchor logits"):
        batch_texts = anchor_texts[start: start + batch_size]
        if num_meta_prompts_to_use_per_text:
            # Expand batch to prepred 4 random meta-prompts to each batch_text
            batch_texts = [[random.choice(meta_prompts) + " " + batch_text for _ in range(
                num_meta_prompts_to_use_per_text)] for batch_text in batch_texts]
            # Flatten the batch_texts
            batch_texts = [item for sublist in batch_texts for item in sublist]

        if use_chat_template:
            batch_texts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": batch_text}], add_generation_prompt=True, tokenize=False) for batch_text in batch_texts]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length_anchor,
        )
        
        
        input_ids = enc.input_ids.to(teacher_model.device)
        attention_mask = enc.attention_mask.to(teacher_model.device)
        with torch.no_grad():
            # Generate num_generated_tokens tokens
            generated_outputs = teacher_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_generated_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True, 
                output_scores=True
            )
            
            seqs = generated_outputs.sequences   # [batch_size, total_len]

            # scores: list[Tensor], len = num_generated_tokens, each (batch, vocab)
            logits_generated = torch.stack(generated_outputs.scores, dim=1) # [batch_size, num_generated_tokens, vocab]
            
            token_ids_generated = seqs[:, input_ids.shape[1]:]

        probs = torch.softmax(logits_generated, dim=-1)  # [B, num_generated_tokens, vocab]

        B, T, V = probs.shape
        for b in range(B):
            seq_len = T # int(attention_mask[b].sum().item())
            # Collect eligible positions
            eligible_positions: list[int] = []
            t = 0
            while t < seq_len:
                p_t = probs[b, t]
                max_prob = torch.max(p_t)
                if float(max_prob) >= confidence_threshold:
                    eligible_positions.append(t) 
                    # also include subsequent position if valid
                    if t + 1 < seq_len:
                        eligible_positions.append(t + 1) 
                t += 1

            if len(eligible_positions) == 0:
                positions_tensor = torch.zeros((0,), dtype=torch.long)
                topk_indices = torch.zeros((0, top_k), dtype=torch.long)
                topk_log_probs = torch.zeros((0, top_k), dtype=torch.float)
            else:
                # Deduplicate and sort positions
                eligible_positions = sorted(set(eligible_positions))
                P = len(eligible_positions)
                positions_tensor = torch.tensor(
                    eligible_positions, dtype=torch.long) + input_ids.shape[1] # eligible position indexes over the full sequence
                # TODO: Check for off-by-one errors here
                topk_indices = torch.empty((P, top_k), dtype=torch.long)
                topk_log_probs = torch.empty((P, top_k), dtype=torch.float)
                for i, pos in enumerate(eligible_positions):
                    logits_pos = logits_generated[b, pos]  # teacher logits at pos
                    log_probs_pos = torch.log_softmax(logits_pos, dim=-1)
                    top_vals, top_idx = torch.topk(log_probs_pos, k=top_k, dim=-1)
                    topk_indices[i] = top_idx.to(torch.long).cpu()
                    topk_log_probs[i] = top_vals.cpu()
            all_input_ids = torch.cat([input_ids[b], token_ids_generated[b]], dim=0)
            all_attention_mask = torch.cat([attention_mask[b], torch.ones_like(token_ids_generated[b])], dim=0)
            rec = {
                "input_ids": all_input_ids.detach().cpu(),
                "attention_mask": all_attention_mask.detach().cpu(),
                "positions": positions_tensor,
                "topk_indices": topk_indices,
                "topk_log_probs": topk_log_probs,
            }
            records.append(rec)


    torch.save(records, save_path)
    with open(precompute_config_path, "w") as f:
        json.dump({
            "anchor_texts": anchor_texts,
            "teacher_model_id": teacher_model_id,
            "num_generated_tokens": num_generated_tokens,
            "tokenizer": tokenizer.name_or_path,
            "max_length_anchor": max_length_anchor,
            "batch_size": batch_size,
            "confidence_threshold": confidence_threshold,
            "top_k": top_k,
        }, f)   
    return save_path, len(records)


class AnchorSFTTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        anchor_loader: DataLoader | None = None,
        lambda_anchor: float = 0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.anchor_loader = anchor_loader
        self.lambda_anchor = lambda_anchor
        self._anchor_iter = iter(
            anchor_loader) if anchor_loader is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Primary SFT loss
        sft_out = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        if isinstance(sft_out, tuple):
            sft_loss, outputs = sft_out
        else:
            sft_loss, outputs = sft_out, None

        anchor_loss = torch.tensor(0.0, device=model.device)
        total_pos = 0

        if self.anchor_loader is not None and self._anchor_iter is not None:
            try:
                batch = next(self._anchor_iter)
            except StopIteration:
                self._anchor_iter = iter(self.anchor_loader)
                batch = next(self._anchor_iter)

            # Move to device
            batch = {k: v.to(model.device) if torch.is_tensor(
                v) else v for k, v in batch.items()}

            if batch["pos_batch_idx"].numel() > 0:

                logits = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).logits  # [B, T, V]

                student_log_probs = torch.log_softmax(logits, dim=-1)
                # Select positions
                # after student_log_probs = torch.log_softmax(logits, dim=-1)

                # 1) select the [P, V] logits at the eligible positions
                sel_logits = student_log_probs[batch["pos_batch_idx"], batch["pos_time_idx"], :]  # [P, V]

                # 2) gather top-k along vocab using the precomputed indices
                topk_indices = batch["topk_indices"]  # [P, K]
                log_q = torch.gather(sel_logits, dim=-1, index=topk_indices)  # [P, K]

                # Teacher log-probs
                log_p = batch["topk_log_probs"]  # [P, K]

                # KL(p||q) on the restricted set
                kl = torch.sum(torch.exp(log_p) * (log_p - log_q), dim=-1)  # [P]
                anchor_loss = kl.mean()
                # KL(p||q) per position
                # Note that this is not proper, since we are not normalizing the student logits or the teacher logits in the restricted space,
                # but we assume that it is fine since most of the mass is concentrated on the top-k tokens by selection.
                total_pos = kl.shape[0]

        total_loss = sft_loss + self.lambda_anchor * anchor_loss

        # Log components
        try:
            self.log({
                "loss_sft": sft_loss.detach().item(),
                "loss_anchor": anchor_loss.detach().item() if torch.is_tensor(anchor_loss) else float(anchor_loss),
                "anchor_positions": int(total_pos),
            })
        except Exception:
            pass

        if return_outputs:
            return total_loss, outputs
        return total_loss


class AnchorPrecomputedDataset(TorchDataset):
    """Dataset for precomputed teacher distributions on anchor texts.

    Each record contains:
      - input_ids: LongTensor [T]
      - attention_mask: LongTensor [T]
      - positions: LongTensor [P] (eligible positions)
      - topk_indices: LongTensor [P, K]
      - topk_log_probs: FloatTensor [P, K] (normalized over K)
    """

    def __init__(self, records_path: str):
        super().__init__()
        self.records_path = records_path
        self.records = torch.load(records_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        return self.records[idx]


def collate_anchor_batch(samples: list[dict], pad_token_id: int) -> dict:
    """Right-pad anchor samples and flatten eligible positions across batch."""
    if len(samples) == 0:
        return {}
    # Determine max length
    max_len = max(s["input_ids"].shape[0] for s in samples)
    batch_input_ids = []
    batch_attention_mask = []
    pos_batch_idx = []
    pos_time_idx = []
    topk_indices_list = []
    topk_log_probs_list = []

    for b_idx, s in enumerate(samples):
        seq_len = s["input_ids"].shape[0]
        pad_len = max_len - seq_len
        if pad_len > 0:
            pad_ids = torch.full((pad_len,), pad_token_id, dtype=torch.long)
            pad_mask = torch.zeros((pad_len,), dtype=torch.long)
            input_ids = torch.cat([s["input_ids"], pad_ids], dim=0)
            attention_mask = torch.cat([s["attention_mask"], pad_mask], dim=0)
        else:
            input_ids = s["input_ids"]
            attention_mask = s["attention_mask"]

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)

        if s["positions"].numel() > 0:
            P = s["positions"].shape[0]
            pos_batch_idx.append(torch.full((P,), b_idx, dtype=torch.long))
            pos_time_idx.append(s["positions"].to(torch.long))
            topk_indices_list.append(s["topk_indices"].to(torch.long))
            topk_log_probs_list.append(s["topk_log_probs"].to(torch.float))

    batch = {
        "input_ids": torch.stack(batch_input_ids, dim=0),
        "attention_mask": torch.stack(batch_attention_mask, dim=0),
    }

    if len(pos_batch_idx) > 0:
        batch["pos_batch_idx"] = torch.cat(pos_batch_idx, dim=0)
        batch["pos_time_idx"] = torch.cat(pos_time_idx, dim=0)
        batch["topk_indices"] = torch.cat(topk_indices_list, dim=0)
        batch["topk_log_probs"] = torch.cat(topk_log_probs_list, dim=0)
    else:
        # Empty tensors to keep shapes consistent
        batch["pos_batch_idx"] = torch.zeros((0,), dtype=torch.long)
        batch["pos_time_idx"] = torch.zeros((0,), dtype=torch.long)
        batch["topk_indices"] = torch.zeros((0, 0), dtype=torch.long)
        batch["topk_log_probs"] = torch.zeros((0, 0), dtype=torch.float)

    return batch
