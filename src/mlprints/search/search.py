"""Core GCG search routine for optimizing modifiable user-span tokens."""

import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlprints.common.constants import MASK_LOSS_ID
from mlprints.common.distributed import get_distributed_info
from mlprints.common.utils import are_tokenizers_equal, compute_causal_lm_cross_entropy_loss

from .formatting import format_gcg_samples

_SAMPLE_KEYS = ("input_ids", "attention_mask", "labels", "user_positions")


@torch.enable_grad()
def _compute_coordinate_gradients(
    model: AutoModelForCausalLM,
    embedding_layer: torch.nn.Embedding,
    embed_weights_fp32_transposed: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    user_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute token-coordinate gradients for the selected user positions.

    NOTE: Transposed weight matrix is provided as an argument for memory efficiency.
    """
    inputs_embeds = embedding_layer(input_ids).detach().requires_grad_(True)
    loss = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, use_cache=False).loss
    (grad_wrt_embeds,) = torch.autograd.grad(loss, inputs_embeds)

    if user_positions is not None:
        grad_wrt_embeds = grad_wrt_embeds[:, user_positions, :]

    grads_wrt_onehot = grad_wrt_embeds.to(torch.float32) @ embed_weights_fp32_transposed
    return loss.detach(), grads_wrt_onehot.detach()


@torch.enable_grad()
def run_gcg_search(
    models: Sequence[AutoModelForCausalLM],
    tokenizers: Sequence[AutoTokenizer],
    prompt_or_messages: Sequence[dict] | Sequence[Sequence[dict]],
    chat_template: str | None | Callable = None,
    *,
    num_steps: int = 50,
    updates_per_step: int = 1,
    top_k: int = 256,
    batch_size: int = 512,
    mini_batch_size: int = 64,
    banned_token_ids: Sequence[int] | torch.Tensor | None = None,
    log_every: int = 10,
    model_coefficients: Sequence[float] | None = None,
    max_modifiable_tokens: int | None = None,
    enforce_improvement: bool = False,
    candidate_filter: Callable[[list[int], int, int], bool] | None = None,
    on_step_callback: Callable[[dict[str, Any]], bool | None] | None = None,
    early_stop_loss_threshold: float | None = None,
    early_stop_model_index: int = 0,
    return_history: bool = True,
    fixed_shape_candidates: bool = False,
) -> dict[str, Any]:
    """Run multi-task GCG over user-span tokens.

    NOTE: Each model is treated as living on a single device.
    NOTE: Sharding not currently supported.
    """
    if not models:
        raise ValueError("at least one model must be provided")
    if not tokenizers:
        raise ValueError("at least one tokenizer must be provided")
    if min(batch_size, mini_batch_size, num_steps, updates_per_step, top_k) <= 0:
        raise ValueError("batch_size, mini_batch_size, num_steps, updates_per_step, and top_k must be positive")
    if mini_batch_size > batch_size:
        raise ValueError("mini_batch_size cannot exceed batch_size")

    distributed, rank, _ = get_distributed_info()

    models_list = list(models)
    tokenizers_list = list(tokenizers)
    if len(tokenizers_list) != len(models_list):
        raise ValueError("tokenizers must match number of models")
    are_tokenizers_equal(tokenizers_list, raise_on_mismatch=True)
    tokenizer = tokenizers_list[0]

    if model_coefficients is not None and len(model_coefficients) != len(models_list):
        raise ValueError("model_coefficients must match number of models")
    coefficients = list(model_coefficients) if model_coefficients else [1.0] * len(models_list)

    model_devices = [next(m.parameters()).device for m in models_list]
    coord_device = model_devices[0] # preferred device for coordinate gradients
    unique_devices = list(dict.fromkeys(model_devices))

    embedding_layers = [m.get_input_embeddings() for m in models_list]

    for model in models_list:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    embed_weights_fp32_transposed = [layer.weight.detach().to(torch.float32).t().contiguous() for layer in embedding_layers]

    formatted_samples = format_gcg_samples(tokenizer, prompt_or_messages, chat_template, max_modifiable_tokens)
    context_samples = formatted_samples["context_samples"]
    base_sample = context_samples[0]  # representative sample for span decoding and final prompt readout
    prompt_len = formatted_samples["prompt_len"]
    user_prompt_ids = formatted_samples["user_prompt_ids"].to(coord_device)

    for sample in context_samples:
        for key in _SAMPLE_KEYS:
            sample[key] = sample[key].to(coord_device)
    base_user_positions = base_sample["user_positions"]

    vocab_size = embedding_layers[0].num_embeddings

    if banned_token_ids is not None:
        banned_token_ids = torch.as_tensor(banned_token_ids, dtype=torch.long, device=coord_device)

    def _sum_on_coord(tensors: Iterable[torch.Tensor], shape: tuple[int, ...] = ()) -> torch.Tensor:
        out = torch.zeros(shape, dtype=torch.float32, device=coord_device)
        for tensor in tensors:
            out += tensor if tensor.device == coord_device else tensor.to(coord_device, non_blocking=True)
        return out

    # pre-materialize samples on each device to avoid per-step .to() copies
    device_samples = {
        dev: [{key: sample[key].to(dev, non_blocking=True) for key in _SAMPLE_KEYS} for sample in context_samples]
        for dev in unique_devices
    }

    pair_cache = []
    pair_cache_by_device = {dev: [] for dev in unique_devices}
    for model_idx, (model, coeff) in enumerate(zip(models_list, coefficients)):
        model_device = model_devices[model_idx]
        for sample in device_samples[model_device]:
            entry = {
                "model": model,
                "embedding_layer": embedding_layers[model_idx],
                "embed_weights_fp32_transposed": embed_weights_fp32_transposed[model_idx],
                "input_ids": sample["input_ids"],
                "attention_mask": sample["attention_mask"],
                "labels": sample["labels"],
                "user_positions": sample["user_positions"],
                "coefficient": coeff,
                "device": model_device,
                "model_idx": model_idx,
            }
            pair_cache.append(entry)
            pair_cache_by_device[model_device].append(entry)

    samples_to_update = []
    _seen_input_ids = set()
    for samples in device_samples.values():
        for sample in samples:
            if id(sample["input_ids"]) not in _seen_input_ids:
                _seen_input_ids.add(id(sample["input_ids"]))
                samples_to_update.append(sample)

    history = [] if (return_history and rank == 0) else None
    pbar = tqdm(total=num_steps * updates_per_step, desc=f"GCG ({'dist' if distributed else 'local'}, {len(pair_cache)} pairs)", disable=(rank != 0))
    start_time = time.time()
    current_loss_val = float("inf")
    step_count = 0
    stop_early = False
    require_monotonic = enforce_improvement or updates_per_step > 1
    user_prompt_ids_py = user_prompt_ids.cpu().tolist()

    if distributed:
        stop_buffer = torch.zeros(1, dtype=torch.int32, device=coord_device)

    def _should_stop(local_stop: bool, *, any_rank: bool = False) -> bool:
        if not distributed:
            return local_stop
        stop_buffer.fill_(int(local_stop))
        if any_rank:
            dist.all_reduce(stop_buffer, op=dist.ReduceOp.MAX)
        else:
            dist.broadcast(stop_buffer, src=0)
        return stop_buffer.item() > 0

    for _ in range(num_steps):
        for _ in range(updates_per_step):
            grads_accum_by_device = {
                dev: torch.zeros(prompt_len, vocab_size, dtype=torch.float32, device=dev)
                for dev in unique_devices
            }
            baseline_loss_by_device = {
                dev: torch.zeros((), dtype=torch.float32, device=dev)
                for dev in unique_devices
            }
            monitored_loss_total = None
            monitored_loss_count = 0

            for entry in pair_cache:
                loss, grads = _compute_coordinate_gradients(
                    entry["model"], entry["embedding_layer"], entry["embed_weights_fp32_transposed"],
                    entry["input_ids"], entry["attention_mask"], entry["labels"],
                    user_positions=entry["user_positions"],
                )
                dev = entry["device"]
                coeff = entry["coefficient"]
                grads_accum_by_device[dev] += coeff * grads[0].to(dev, torch.float32, non_blocking=True)
                baseline_loss_by_device[dev] += coeff * loss.to(dev, dtype=torch.float32)
                if (
                    early_stop_loss_threshold is not None
                    and early_stop_model_index >= 0
                    and entry["model_idx"] == early_stop_model_index
                ):
                    loss_on_coord = loss.to(coord_device, non_blocking=True)
                    monitored_loss_total = loss_on_coord if monitored_loss_total is None else monitored_loss_total + loss_on_coord
                    monitored_loss_count += 1

            if len(unique_devices) == 1:
                grads_accum = grads_accum_by_device[coord_device]
                baseline_loss = baseline_loss_by_device[coord_device]
            else:
                grads_accum = _sum_on_coord(grads_accum_by_device.values(), (prompt_len, vocab_size))
                baseline_loss = _sum_on_coord(baseline_loss_by_device.values())

            if distributed:
                dist.all_reduce(grads_accum, op=dist.ReduceOp.SUM)
                dist.all_reduce(baseline_loss, op=dist.ReduceOp.SUM)

            pre_loss_val = baseline_loss.item()

            if early_stop_loss_threshold is not None:
                local_stop = False
                if monitored_loss_count > 0:
                    assert monitored_loss_total is not None
                    local_stop = (monitored_loss_total / monitored_loss_count).item() <= early_stop_loss_threshold
                if _should_stop(local_stop, any_rank=True):
                    stop_early = True
                    break

            if enforce_improvement:
                grads_accum[torch.arange(prompt_len, device=coord_device), user_prompt_ids] = float("inf")
            if banned_token_ids is not None:
                grads_accum[:, banned_token_ids] = float("inf")

            k = min(top_k, vocab_size)
            _, top_tokens = torch.topk(grads_accum, k=k, dim=-1, largest=False)

            num_candidates = min(batch_size, prompt_len * k)
            if not distributed or rank == 0:
                candidate_positions = torch.randint(0, prompt_len, (num_candidates,), device=coord_device)
                candidate_k = torch.randint(0, k, (num_candidates,), device=coord_device)
            else:
                candidate_positions = torch.empty(num_candidates, dtype=torch.long, device=coord_device)
                candidate_k = torch.empty(num_candidates, dtype=torch.long, device=coord_device)
            if distributed:
                dist.broadcast(candidate_positions, src=0)
                dist.broadcast(candidate_k, src=0)

            candidate_tokens = top_tokens[candidate_positions, candidate_k]

            # pad the final chunk to keep a static forward shape when requested
            candidate_losses = torch.empty(num_candidates, dtype=torch.float32, device=coord_device)
            # buffers are keyed by (device, seq_len): contexts on one device may differ in length
            candidate_buf_by_shape: dict[tuple[torch.device, int], torch.Tensor] = {}
            candidate_row_idx_by_device: dict[torch.device, torch.Tensor] = {}
            for chunk_start in range(0, num_candidates, mini_batch_size):
                chunk_end = min(chunk_start + mini_batch_size, num_candidates)
                chunk_size = chunk_end - chunk_start
                pad_size = mini_batch_size - chunk_size if fixed_shape_candidates and chunk_size < mini_batch_size else 0
                rows = chunk_size + pad_size  # mini_batch_size when padding, chunk_size otherwise
                chunk_pos = candidate_positions[chunk_start:chunk_end]
                chunk_tok = candidate_tokens[chunk_start:chunk_end]
                if pad_size > 0:
                    chunk_pos = torch.cat([chunk_pos, chunk_pos[:1].expand(pad_size)])
                    chunk_tok = torch.cat([chunk_tok, chunk_tok[:1].expand(pad_size)])

                chunk_loss_by_device = {
                    dev: torch.zeros(chunk_size, dtype=torch.float32, device=dev)
                    for dev in unique_devices
                }
                for dev, caches in pair_cache_by_device.items():
                    chunk_pos_dev = chunk_pos.to(dev, non_blocking=True)
                    chunk_tok_dev = chunk_tok.to(dev, non_blocking=True)
                    row_idx = candidate_row_idx_by_device.get(dev)
                    if row_idx is None:
                        row_idx = torch.arange(mini_batch_size, device=dev)
                        candidate_row_idx_by_device[dev] = row_idx
                    for cache in caches:
                        input_ids = cache["input_ids"]
                        coeff = cache["coefficient"]
                        seq_len = input_ids.shape[1]

                        buf = candidate_buf_by_shape.get((dev, seq_len))
                        if buf is None:
                            buf = torch.empty(mini_batch_size, seq_len, dtype=input_ids.dtype, device=dev)
                            candidate_buf_by_shape[(dev, seq_len)] = buf

                        seq_input_ids = buf[:rows]
                        seq_input_ids.copy_(input_ids.expand(rows, -1))
                        formatted_positions = cache["user_positions"][chunk_pos_dev]
                        seq_input_ids[row_idx[:rows], formatted_positions] = chunk_tok_dev

                        with torch.inference_mode():
                            logits = cache["model"](
                                input_ids=seq_input_ids,
                                attention_mask=cache["attention_mask"].expand(rows, -1),
                                use_cache=False,
                            ).logits
                        per_sample_losses = compute_causal_lm_cross_entropy_loss(logits, cache["labels"].expand(rows, -1))
                        if pad_size > 0:
                            per_sample_losses = per_sample_losses[:chunk_size]
                        chunk_loss_by_device[dev] += coeff * per_sample_losses.to(dev, dtype=torch.float32, non_blocking=True)

                candidate_losses[chunk_start:chunk_end] = _sum_on_coord(chunk_loss_by_device.values(), (chunk_size,))

            if distributed:
                dist.all_reduce(candidate_losses, op=dist.ReduceOp.SUM)

            def _select_best_candidate():
                # fast path: no filtering or monotonicity
                if not require_monotonic and candidate_filter is None:
                    best_idx = candidate_losses.argmin().item()
                    return (
                        candidate_losses[best_idx].item(),
                        candidate_positions[best_idx].item(),
                        candidate_tokens[best_idx].item(),
                    )

                candidate_losses_cpu = candidate_losses.detach().cpu()
                candidate_positions_cpu = candidate_positions.detach().cpu()
                candidate_tokens_cpu = candidate_tokens.detach().cpu()

                for idx in torch.argsort(candidate_losses_cpu).tolist():
                    cand_loss = candidate_losses_cpu[idx].item()
                    cand_pos = candidate_positions_cpu[idx].item()
                    cand_tok = candidate_tokens_cpu[idx].item()
                    if require_monotonic and cand_loss >= pre_loss_val:
                        break
                    if candidate_filter is not None:
                        cand_ids = user_prompt_ids_py.copy()
                        cand_ids[cand_pos] = cand_tok
                        if not candidate_filter(cand_ids, cand_pos, cand_tok):
                            continue
                    return cand_loss, cand_pos, cand_tok
                return None

            if distributed:
                if rank == 0:
                    selected = _select_best_candidate()
                    selection_tensor = torch.tensor(
                        [-1, -1] if selected is None else [selected[1], selected[2]],
                        dtype=torch.long,
                        device=coord_device,
                    )
                else:
                    selection_tensor = torch.empty(2, dtype=torch.long, device=coord_device)

                dist.broadcast(selection_tensor, src=0)
                best_pos = selection_tensor[0].item()
                best_tok = selection_tensor[1].item()

                if best_pos == -1:
                    stop_early = True
                    break

                best_loss = selected[0] if rank == 0 else pre_loss_val
            else:
                selected = _select_best_candidate()
                if selected is None:
                    stop_early = True
                    break
                best_loss, best_pos, best_tok = selected

            user_prompt_ids[best_pos] = best_tok
            user_prompt_ids_py[best_pos] = best_tok
            for sample in samples_to_update:
                sample["input_ids"][0, sample["user_positions"][best_pos]] = best_tok
            current_loss_val = best_loss
            step_count += 1

            record = {"step": step_count, "pre_loss": pre_loss_val, "loss": current_loss_val, "swap": (best_pos, best_tok)}
            if rank == 0:
                if history is not None:
                    history.append(record)
                pbar.update(1)
                pbar.set_postfix({"loss": f"{current_loss_val:.4f}"})
                if log_every and step_count % log_every == 0:
                    print(f"step {step_count}: loss={current_loss_val:.4f}")

            callback_stop = False
            if rank == 0 and on_step_callback is not None:
                step_event = dict(record)
                step_event["user_tokens"] = user_prompt_ids_py.copy()
                callback_stop = on_step_callback(step_event) is False

            if _should_stop(callback_stop):
                stop_early = True
                break

        if stop_early:
            break

    pbar.close()

    for model in models_list:
        for p in model.parameters():
            p.requires_grad_(True)

    total_time = time.time() - start_time
    base_sample["input_ids"][0, base_user_positions] = torch.tensor(
        user_prompt_ids_py, dtype=base_sample["input_ids"].dtype
    )
    target_start = (base_sample["labels"][0] == MASK_LOSS_ID).sum().item()
    final_prompt_ids = base_sample["input_ids"][:, :target_start].cpu().tolist()
    final_prompt_str = tokenizer.decode(base_sample["input_ids"][0, :target_start])
    final_user_ids = base_sample["input_ids"][0, base_user_positions].cpu().tolist()
    final_user_str = tokenizer.decode(base_sample["input_ids"][0, base_user_positions], skip_special_tokens=False)

    final_loss = history[-1]["loss"] if history else current_loss_val if step_count > 0 else None

    search_results = {
        "final_prompt_ids": final_prompt_ids,
        "final_prompt_str": final_prompt_str,
        "final_user_ids": final_user_ids,
        "final_user_str": final_user_str,
        "history": history,
        "total_steps": step_count,
        "total_time": total_time,
        "final_loss": final_loss,
    }

    return search_results
