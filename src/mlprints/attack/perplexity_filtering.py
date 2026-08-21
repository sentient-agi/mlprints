"""Reproduction of arXiv:2509.26598.

"""

import torch
from transformers.generation.utils import GenerateDecoderOnlyOutput

from mlprints.attack.base import AttackModel, UNIVERSAL_OVERLOAD_MESSAGE
from mlprints.common.utils import get_model_device
from mlprints.inference import run_inference_strided_perplexity
from mlprints.loading import load_model, load_tokenizer

class PerplexityFilteringAttackModel(AttackModel):

    def __init__(
        self,
        model, tokenizer,
        *,
        perplexity_threshold,
        num_tokens_to_check,
        overload_response,
        perplexity_model_id,
        perplexity_device_map,
        perplexity_dtype,
        perplexity_stride,
    ):
        super().__init__(model, tokenizer)

        self.perplexity_threshold = perplexity_threshold
        self.num_tokens_to_check = num_tokens_to_check
        self.perplexity_stride = perplexity_stride

        if perplexity_device_map is None:
            perplexity_device_map = get_model_device(model)
        self.ppl_model = load_model(
            perplexity_model_id,
            dtype=perplexity_dtype,
            device_map=perplexity_device_map,
        )
        self.ppl_tokenizer = load_tokenizer(perplexity_model_id)

        overload_response = overload_response or UNIVERSAL_OVERLOAD_MESSAGE
        self.register_buffer(
            "_overload_tokens",
            tokenizer.encode(
                overload_response,
                return_tensors="pt",
                add_special_tokens=False,
            ).squeeze(0),
            persistent=False,
        )

    def perplexity_scores(self, input_ids, attention_mask=None):
        texts = []
        for idx, row in enumerate(input_ids):
            tokens = row if attention_mask is None else row[attention_mask[idx].bool()]
            tokens = tokens[: self.num_tokens_to_check]
            texts.append(self.tokenizer.decode(tokens, skip_special_tokens=True))

        return run_inference_strided_perplexity(
            self.ppl_model,
            self.ppl_tokenizer,
            texts,
            stride=self.perplexity_stride,
        )
    
    def generate(self, input_ids=None, **kwargs):
        if input_ids is None:
            input_ids = kwargs.pop("input_ids")

        attention_mask = kwargs.get("attention_mask")
        blocked = [
            score > self.perplexity_threshold
            for score in self.perplexity_scores(input_ids, attention_mask)
        ]
        if not any(blocked):
            return self.model.generate(input_ids=input_ids, **kwargs)

        if kwargs.get("output_scores", False):
            raise ValueError(
                "output_scores is unavailable when perplexity filtering blocks "
                "a prompt because the fixed response has no model scores"
            )

        num_return_sequences = kwargs.get("num_return_sequences", 1)

        structured = kwargs.get("return_dict_in_generate", False)
        generated = None
        if not all(blocked):
            keep = torch.tensor(
                [idx for idx, is_blocked in enumerate(blocked) if not is_blocked],
                device=input_ids.device,
            )
            generation_kwargs = kwargs.copy()
            if attention_mask is not None:
                generation_kwargs["attention_mask"] = attention_mask[keep]
            generation_kwargs["return_dict_in_generate"] = False
            generated = self.model.generate(
                input_ids=input_ids.index_select(0, keep),
                **generation_kwargs,
            )

        response = self._overload_tokens[:kwargs.get("max_new_tokens")]

        rows, generated_idx = [], 0
        for sample_idx, is_blocked in enumerate(blocked):
            if is_blocked:
                row = torch.cat(
                    (input_ids[sample_idx], response.to(input_ids.device))
                )
                rows.extend([row] * num_return_sequences)
            else:
                next_idx = generated_idx + num_return_sequences
                rows.extend(generated[generated_idx:next_idx])
                generated_idx = next_idx

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        sequences = torch.nn.utils.rnn.pad_sequence(
            rows,
            batch_first=True,
            padding_value=pad_id,
        )
        if structured:
            return GenerateDecoderOnlyOutput(sequences=sequences)
        return sequences
    
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

        return cls(model=model, tokenizer=tokenizer, **attack_params)


def perplexity_filtering(
    model_checkpoint,
    *,
    perplexity_threshold,
    num_tokens_to_check,
    overload_response,
    perplexity_model_id,
    perplexity_device_map,
    perplexity_dtype,
    perplexity_stride,
    device_map,
    dtype,
    trust_remote_code,
):
    overload_response = overload_response or UNIVERSAL_OVERLOAD_MESSAGE

    attack_config = {
        "algo": {"name": "perplexity_filtering"},
        "base_model_config": {
            "model_id": model_checkpoint,
            "device_map": device_map,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        },
        "attack_params": {
            "perplexity_threshold": perplexity_threshold,
            "num_tokens_to_check": num_tokens_to_check,
            "overload_response": overload_response,
            "perplexity_model_id": perplexity_model_id,
            "perplexity_device_map": perplexity_device_map,
            "perplexity_dtype": perplexity_dtype,
            "perplexity_stride": perplexity_stride,
        },
    }

    metadata = {
        "attack_type": "perplexity_filtering",
        "model_checkpoint": model_checkpoint,
        **attack_config["attack_params"],
    }

    return attack_config, metadata
