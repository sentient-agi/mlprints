"""Reproduction of arXiv:2509.26598.

NOTE:
- This attack is called ``statistical_analysis`` in the paper.
- ``calibration_texts`` should be generated from BOS plus ``seed_words``.
- The calibration model must share the attacked model's tokenizer vocabulary.
"""

import math

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from mlprints.attack.base import AttackModel
from mlprints.common.cache import resolve_cached_attack_asset
from mlprints.inference import run_inference
from mlprints.loading import load_model, load_tokenizer


class BigramSuppressionProcessor(LogitsProcessor):

    def __init__(self, mapping, suppression_delta):
        self.mapping = mapping
        self.suppression_delta = suppression_delta

    def __call__(self, input_ids, scores):
        for row, previous_token_id in enumerate(input_ids[:, -1].tolist()):
            if previous_token_id not in self.mapping:
                continue
            token_ids, weights = self.mapping[previous_token_id]
            token_ids = token_ids.to(scores.device)
            scores[row, token_ids] -= self.suppression_delta * weights.to(
                scores.device
            )
        return scores


class BigramSuppressionAttackModel(AttackModel):

    @torch.inference_mode()
    def __init__(
        self,
        model,
        tokenizer,
        calibration_model,
        *,
        seed_words,
        calibration_texts,
        probability_threshold,
        clip_quantile,
        suppression_delta,
        batch_size,
    ):
        super().__init__(model, tokenizer)
        seed_tokens = {}
        for word in seed_words:
            token_ids = tokenizer.encode(f" {word}", add_special_tokens=False)
            if len(token_ids) == 1:
                seed_tokens.setdefault(token_ids[0], word)

        bos = tokenizer.bos_token or ""
        log_threshold = math.log(probability_threshold)
        mapping = {}

        for previous_token_id, word in seed_tokens.items():
            ratios = []
            for start in range(0, len(calibration_texts), batch_size):
                prompts = [
                    f"{bos}{text} {word}"
                    for text in calibration_texts[start:start + batch_size]
                ]
                model_scores = run_inference(
                    model,
                    tokenizer,
                    prompts,
                    apply_chat_template=False,
                    max_new_tokens=1,
                    do_sample=False,
                    output_scores=True,
                )
                calibration_scores = run_inference(
                    calibration_model,
                    tokenizer,
                    prompts,
                    apply_chat_template=False,
                    max_new_tokens=1,
                    do_sample=False,
                    output_scores=True,
                )
                model_logprobs = model_scores["scores"][0].float().log_softmax(
                    dim=-1
                ).cpu()
                calibration_logprobs = calibration_scores["scores"][
                    0
                ].float().log_softmax(
                    dim=-1
                )
                calibration_logprobs = calibration_logprobs.cpu()
                ratio = model_logprobs - calibration_logprobs
                ratio[model_logprobs <= log_threshold] = torch.nan
                ratios.append(ratio)

            ratios = torch.cat(ratios)
            counts = ratios.isfinite().sum(dim=0)
            maximum = ratios.nan_to_num(nan=-torch.inf).amax(dim=0)
            mean = ratios.nan_to_num(nan=0.0).sum(dim=0) / counts
            centered = (ratios - mean).nan_to_num(nan=0.0)
            standard_error = (
                centered.square().sum(dim=0) / (counts - 1)
            ).sqrt() / counts.sqrt()
            zscore = mean / standard_error

            maximum_bound = maximum[maximum.isfinite()].abs().quantile(
                clip_quantile
            )
            maximum = (
                maximum.clamp(-maximum_bound, maximum_bound) / maximum_bound
            )
            zscore_bound = zscore[
                zscore.isfinite()
            ].abs().quantile(clip_quantile)
            zscore = zscore.clamp(-zscore_bound, zscore_bound) / zscore_bound
            scores = torch.hypot(maximum, zscore)
            scores = torch.where(
                (maximum > 0) & (zscore > 0),
                scores,
                -scores,
            )
            scores[tokenizer.all_special_ids] = torch.nan
            token_ids = scores.isfinite().nonzero().flatten()
            mapping[previous_token_id] = (token_ids, scores[token_ids])

        self.processor = BigramSuppressionProcessor(
            mapping,
            suppression_delta,
        )

    def generate(self, input_ids=None, **kwargs):
        if input_ids is None:
            input_ids = kwargs.pop("input_ids")
        existing = kwargs.pop("logits_processor", None)
        kwargs["logits_processor"] = LogitsProcessorList([
            self.processor,
            *(existing or []),
        ])
        kwargs["renormalize_logits"] = True
        return self.model.generate(input_ids=input_ids, **kwargs)

    @classmethod
    def from_config(cls, config):
        base_config = config["base_model_config"]
        attack_params = config["attack_params"]
        model_id = base_config["model_id"]
        model = load_model(
            model_id,
            device_map=base_config.get("device_map", "auto"),
            dtype=base_config.get("dtype", "auto"),
            attn_implementation=base_config.get("attn_implementation"),
            trust_remote_code=base_config.get("trust_remote_code", False),
        )
        tokenizer = load_tokenizer(
            model_id,
            trust_remote_code=base_config.get("trust_remote_code", False),
        )
        calibration_model = load_model(
            attack_params["calibration_model_id"],
            device_map=attack_params.get("calibration_device_map", "auto"),
            dtype=attack_params.get("calibration_dtype", "auto"),
            attn_implementation=attack_params.get(
                "calibration_attn_implementation"
            ),
            trust_remote_code=attack_params.get(
                "calibration_trust_remote_code", False
            ),
        )
        seed_words_path = resolve_cached_attack_asset(
            attack_params["seed_words_path"],
            algo_name="bigram_suppression",
            source_fmt="text",
            output_fmt="text",
            num_samples=attack_params["num_seed_words"],
        )
        calibration_texts_path = resolve_cached_attack_asset(
            attack_params["calibration_texts_path"],
            algo_name="bigram_suppression",
            source_fmt="text",
            output_fmt="text",
            num_samples=attack_params["num_calibration_texts"],
        )
        seed_words = [
            line.strip()
            for line in seed_words_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        calibration_texts = [
            line.strip()
            for line in calibration_texts_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        return cls(
            model,
            tokenizer,
            calibration_model,
            seed_words=seed_words,
            calibration_texts=calibration_texts,
            probability_threshold=attack_params["probability_threshold"],
            clip_quantile=attack_params["clip_quantile"],
            suppression_delta=attack_params["suppression_delta"],
            batch_size=attack_params["batch_size"],
        )


def bigram_suppression(
    model_checkpoint,
    *,
    calibration_model_id,
    seed_words_path,
    calibration_texts_path,
    num_seed_words,
    num_calibration_texts,
    probability_threshold,
    clip_quantile,
    suppression_delta,
    batch_size,
    calibration_device_map,
    calibration_dtype,
    calibration_attn_implementation,
    calibration_trust_remote_code,
    device_map,
    dtype,
    attn_implementation,
    trust_remote_code,
):
    attack_config = {
        "algo": {"name": "bigram_suppression"},
        "base_model_config": {
            "model_id": model_checkpoint,
            "device_map": device_map,
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "trust_remote_code": trust_remote_code,
        },
        "attack_params": {
            "calibration_model_id": calibration_model_id,
            "seed_words_path": seed_words_path,
            "calibration_texts_path": calibration_texts_path,
            "num_seed_words": num_seed_words,
            "num_calibration_texts": num_calibration_texts,
            "probability_threshold": probability_threshold,
            "clip_quantile": clip_quantile,
            "suppression_delta": suppression_delta,
            "batch_size": batch_size,
            "calibration_device_map": calibration_device_map,
            "calibration_dtype": calibration_dtype,
            "calibration_attn_implementation": (
                calibration_attn_implementation
            ),
            "calibration_trust_remote_code": (
                calibration_trust_remote_code
            ),
        },
    }
    metadata = {
        "attack_type": "bigram_suppression",
        "model_checkpoint": model_checkpoint,
        **attack_config["attack_params"],
    }
    return attack_config, metadata
