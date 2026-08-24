"""Reproduction of arXiv:2509.26598.

NOTE:
- Chat templates belong to ``mlprints.inference.run_inference``. This attack
  operates on the resulting tokenized prompt and therefore supports base,
  instruct, and chat models without model-specific formatting.
- SuppressTop-k, SuppressNeighbor, and SuppressLookahead are gated by output
  confidence; suppression-only variants disable these gates.
- We interpret the paper's "stop words" as tokenizer special/stop tokens.
"""

import torch
from transformers import ContinuousBatchingConfig, LogitsProcessor, LogitsProcessorList
from transformers.generation.continuous_batching.cb_logits_processors import (
    ContinuousBatchingLogitsProcessor,
)

from mlprints.attack.base import AttackModel
from mlprints.inference.batching import injected_logits_processors
from mlprints.loading import load_model, load_tokenizer


class DetectTopKProcessor(LogitsProcessor):

    def __init__(self, k, num_tokens, threshold, prompt_length):
        self.k = k
        self.num_tokens = num_tokens
        self.threshold = threshold
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] - self.prompt_length >= self.num_tokens:
            return scores

        detected = scores.float().softmax(dim=-1).amax(dim=-1) > self.threshold
        token_ids = scores.topk(self.k, dim=-1).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, token_ids, detected[:, None].expand_as(token_ids))
        return scores.masked_fill(mask, -torch.inf)


def _neighbor_token_ids(normalized_tokens, candidate_ids):
    candidates = {
        normalized_tokens[token_id]
        for token_id in candidate_ids
        if normalized_tokens[token_id]
    }
    return [
        token_id
        for token_id, token in enumerate(normalized_tokens)
        if token and any(
            token.startswith(candidate) or candidate.startswith(token)
            for candidate in candidates
        )
    ]


class DetectNeighborProcessor(LogitsProcessor):

    def __init__(
        self,
        normalized_tokens,
        candidate_set_size,
        num_tokens,
        add_threshold,
        generation_threshold,
        prompt_length,
    ):
        self.normalized_tokens = normalized_tokens
        self.candidate_set_size = candidate_set_size
        self.num_tokens = num_tokens
        self.add_threshold = add_threshold
        self.generation_threshold = generation_threshold
        self.prompt_length = prompt_length
        self.neighbor_ids = None

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] - self.prompt_length >= self.num_tokens:
            return scores

        probs = scores.float().softmax(dim=-1)
        if self.neighbor_ids is None:
            top_probs, top_ids = probs.topk(self.candidate_set_size, dim=-1)
            self.neighbor_ids = []
            for row_probs, ids in zip(top_probs, top_ids):
                self.neighbor_ids.append(torch.tensor(
                    _neighbor_token_ids(
                        self.normalized_tokens,
                        ids[row_probs > self.add_threshold].tolist(),
                    ),
                    dtype=torch.long,
                ))

        output = scores.clone()
        for row, token_ids in enumerate(self.neighbor_ids):
            token_ids = token_ids.to(scores.device)
            token_ids = token_ids[
                probs[row, token_ids] > self.generation_threshold
            ]
            output[row, token_ids] = -torch.inf
        return output


class DetectNeighborContinuousProcessor(ContinuousBatchingLogitsProcessor):
    supported_kwargs = {}
    ignored_kwargs = ()

    def __init__(
        self,
        normalized_tokens,
        candidate_set_size,
        num_tokens,
        add_threshold,
        generation_threshold,
    ):
        self.normalized_tokens = normalized_tokens
        self.candidate_set_size = candidate_set_size
        self.num_tokens = num_tokens
        self.add_threshold = add_threshold
        self.generation_threshold = generation_threshold
        self._neighbor_ids = {}
        self._active_requests = []

    def fill_defaults(self, int32_tensor):
        int32_tensor.fill_(0)

    def prepare_tensor_args(self, requests_with_new_token):
        self._active_requests = [
            (request.state.request_id, request.state.generated_len())
            for request in requests_with_new_token
        ]
        return torch.zeros(len(self._active_requests), dtype=torch.int32)

    def __call__(self, scores, _tensor_arg):
        probs = scores.float().softmax(dim=-1)
        output = scores.clone()
        for row, (request_id, generated_len) in enumerate(self._active_requests):
            if generated_len >= self.num_tokens:
                continue
            token_ids = self._neighbor_ids.get(request_id)
            if token_ids is None:
                top_probs, top_ids = probs[row].topk(self.candidate_set_size)
                token_ids = scores.new_tensor(
                    _neighbor_token_ids(
                        self.normalized_tokens,
                        top_ids[top_probs > self.add_threshold].tolist(),
                    ),
                    dtype=torch.long,
                )
                self._neighbor_ids[request_id] = token_ids
            token_ids = token_ids.to(scores.device)
            token_ids = token_ids[probs[row, token_ids] > self.generation_threshold]
            output[row, token_ids] = -torch.inf
        return output


class DetectLookaheadProcessor(LogitsProcessor):

    def __init__(
        self,
        candidate_ids,
        num_tokens,
        suppression_delta,
        generation_threshold,
        prompt_length,
    ):
        self.candidate_ids = candidate_ids
        self.num_tokens = num_tokens
        self.suppression_delta = suppression_delta
        self.generation_threshold = generation_threshold
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] - self.prompt_length >= self.num_tokens:
            return scores

        repeats = scores.shape[0] // len(self.candidate_ids)
        candidate_ids = [
            token_ids
            for token_ids in self.candidate_ids
            for _ in range(repeats)
        ]
        probs = scores.float().softmax(dim=-1)
        output = scores.clone()
        for row, token_ids in enumerate(candidate_ids):
            token_ids = token_ids.to(scores.device)
            token_ids = token_ids[
                probs[row, token_ids] > self.generation_threshold
            ]
            output[row, token_ids] -= self.suppression_delta
        return output


def _from_config(cls, config):
    base_config = config["base_model_config"]
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
    return cls(model, tokenizer, **config["attack_params"])


class DetectAttackModel(AttackModel):

    def _generate(self, input_ids, processor, kwargs):
        existing = kwargs.pop("logits_processor", None)
        kwargs["logits_processor"] = LogitsProcessorList([
            processor,
            *(existing or []),
        ])
        kwargs["renormalize_logits"] = True
        return self.model.generate(input_ids=input_ids, **kwargs)


class DetectTopKAttackModel(DetectAttackModel):

    def __init__(
        self,
        model,
        tokenizer,
        *,
        top_k,
        num_tokens,
        generation_threshold,
    ):
        super().__init__(model, tokenizer)
        self.top_k = top_k
        self.num_tokens = num_tokens
        self.generation_threshold = generation_threshold

    def generate(self, input_ids=None, **kwargs):
        if input_ids is None:
            input_ids = kwargs.pop("input_ids")
        processor = DetectTopKProcessor(
            self.top_k,
            self.num_tokens,
            self.generation_threshold,
            input_ids.shape[1],
        )
        return self._generate(input_ids, processor, kwargs)

    @classmethod
    def from_config(cls, config):
        return _from_config(cls, config)


class DetectNeighborAttackModel(DetectAttackModel):

    def __init__(
        self,
        model,
        tokenizer,
        *,
        candidate_set_size,
        num_tokens,
        add_threshold,
        generation_threshold,
    ):
        super().__init__(model, tokenizer)
        self.candidate_set_size = candidate_set_size
        self.num_tokens = num_tokens
        self.add_threshold = add_threshold
        self.generation_threshold = generation_threshold

        vocab_size = model.get_input_embeddings().num_embeddings
        tokens = tokenizer.batch_decode(
            [[token_id] for token_id in range(vocab_size)],
            clean_up_tokenization_spaces=False,
        )
        normalized_tokens = [
            "".join(char for char in token.casefold() if char.isalnum())
            for token in tokens
        ]
        for token_id in tokenizer.all_special_ids:
            normalized_tokens[token_id] = ""
        self.normalized_tokens = tuple(normalized_tokens)

    def generate(self, input_ids=None, **kwargs):
        if input_ids is None:
            input_ids = kwargs.pop("input_ids")
        processor = DetectNeighborProcessor(
            self.normalized_tokens,
            self.candidate_set_size,
            self.num_tokens,
            self.add_threshold,
            self.generation_threshold,
            input_ids.shape[1],
        )
        return self._generate(input_ids, processor, kwargs)

    def generate_batch(self, inputs, generation_config=None, **kwargs):
        processors = [
            DetectNeighborContinuousProcessor(
                self.normalized_tokens,
                self.candidate_set_size,
                self.num_tokens,
                self.add_threshold,
                self.generation_threshold,
            ),
            *kwargs.pop("logits_processor", []),
        ]
        if generation_config is not None:
            generation_config.renormalize_logits = True
        kwargs.setdefault(
            "continuous_batching_config", ContinuousBatchingConfig()
        ).use_cuda_graph = False
        with injected_logits_processors(self.model, processors):
            return self.model.generate_batch(
                inputs, generation_config=generation_config, **kwargs
            )

    @classmethod
    def from_config(cls, config):
        return _from_config(cls, config)


class DetectLookaheadAttackModel(DetectAttackModel):

    def __init__(
        self,
        model,
        tokenizer,
        *,
        beam_width,
        lookahead_tokens,
        top_k,
        candidate_set_size,
        candidate_threshold,
        use_max_probability,
        num_tokens,
        suppression_delta,
        generation_threshold,
    ):
        super().__init__(model, tokenizer)
        self.beam_width = beam_width
        self.lookahead_tokens = lookahead_tokens
        self.top_k = top_k
        self.candidate_set_size = candidate_set_size
        self.candidate_threshold = candidate_threshold
        self.use_max_probability = use_max_probability
        self.num_tokens = num_tokens
        self.suppression_delta = suppression_delta
        self.generation_threshold = generation_threshold
        self.stop_token_ids = set(tokenizer.all_special_ids)
        vocab_size = model.get_input_embeddings().num_embeddings
        tokens = tokenizer.batch_decode(
            [[token_id] for token_id in range(vocab_size)],
            clean_up_tokenization_spaces=False,
        )
        normalized_tokens = [
            "".join(char for char in token.casefold() if char.isalnum())
            for token in tokens
        ]
        for token_id in tokenizer.all_special_ids:
            normalized_tokens[token_id] = ""
        self.normalized_tokens = tuple(normalized_tokens)

    @torch.inference_mode()
    def generate(self, input_ids=None, **kwargs):
        if input_ids is None:
            input_ids = kwargs.pop("input_ids")
        attention_mask = kwargs.get(
            "attention_mask",
            torch.ones_like(input_ids),
        )

        logits = self.model(
            input_ids,
            attention_mask=attention_mask,
        ).logits[:, -1]
        seed_ids = logits.topk(self.beam_width, dim=-1).indices
        branch_ids = input_ids.repeat_interleave(self.beam_width, dim=0)
        branch_mask = attention_mask.repeat_interleave(
            self.beam_width,
            dim=0,
        )
        branch_ids = torch.cat((branch_ids, seed_ids.reshape(-1, 1)), dim=1)
        branch_mask = torch.cat((
            branch_mask,
            torch.ones_like(branch_mask[:, :1]),
        ), dim=1)

        lookahead = self.model.generate(
            input_ids=branch_ids,
            attention_mask=branch_mask,
            max_new_tokens=self.lookahead_tokens - 1,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        token_stats = [{} for _ in input_ids]
        for step_scores in lookahead.scores:
            top_probs, top_ids = step_scores.float().softmax(
                dim=-1
            ).topk(self.top_k, dim=-1)
            for branch, (token_ids, probs) in enumerate(
                zip(top_ids.tolist(), top_probs.tolist())
            ):
                stats = token_stats[branch // self.beam_width]
                for token_id, probability in zip(token_ids, probs):
                    stats.setdefault(token_id, []).append(probability)

        candidate_ids = []
        for row, stats in enumerate(token_stats):
            prompt_ids = set(
                input_ids[row][attention_mask[row].bool()].tolist()
            )
            prompt = self.tokenizer.decode(
                input_ids[row][attention_mask[row].bool()],
                skip_special_tokens=True,
            )
            prompt_words = {
                "".join(char for char in word.casefold() if char.isalnum())
                for word in prompt.split()
            }

            candidates = []
            for token_id, probabilities in stats.items():
                token = self.normalized_tokens[token_id]
                probability = (
                    max(probabilities)
                    if self.use_max_probability
                    else sum(probabilities) / len(probabilities)
                )
                if (
                    token
                    and token_id not in prompt_ids
                    and token_id not in self.stop_token_ids
                    and token not in prompt_words
                    and probability > self.candidate_threshold
                ):
                    candidates.append(
                        (len(probabilities), probability, token_id)
                    )

            candidates.sort(reverse=True)
            candidate_ids.append(torch.tensor(
                [
                    token_id
                    for _, _, token_id in candidates[
                        :self.candidate_set_size
                    ]
                ],
                dtype=torch.long,
            ))

        processor = DetectLookaheadProcessor(
            candidate_ids,
            self.num_tokens,
            self.suppression_delta,
            self.generation_threshold,
            input_ids.shape[1],
        )
        return self._generate(input_ids, processor, kwargs)

    @classmethod
    def from_config(cls, config):
        return _from_config(cls, config)


def detect_topk(
    model_checkpoint,
    *,
    top_k,
    num_tokens,
    generation_threshold,
    device_map,
    dtype,
    attn_implementation,
    trust_remote_code,
):
    attack_config = {
        "algo": {"name": "detect_topk"},
        "base_model_config": {
            "model_id": model_checkpoint,
            "device_map": device_map,
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "trust_remote_code": trust_remote_code,
        },
        "attack_params": {
            "top_k": top_k,
            "num_tokens": num_tokens,
            "generation_threshold": generation_threshold,
        },
    }
    metadata = {
        "attack_type": "detect_topk",
        "model_checkpoint": model_checkpoint,
        **attack_config["attack_params"],
    }
    return attack_config, metadata


def detect_neighbor(
    model_checkpoint,
    *,
    candidate_set_size,
    num_tokens,
    add_threshold,
    generation_threshold,
    device_map,
    dtype,
    attn_implementation,
    trust_remote_code,
):
    attack_config = {
        "algo": {"name": "detect_neighbor"},
        "base_model_config": {
            "model_id": model_checkpoint,
            "device_map": device_map,
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "trust_remote_code": trust_remote_code,
        },
        "attack_params": {
            "candidate_set_size": candidate_set_size,
            "num_tokens": num_tokens,
            "add_threshold": add_threshold,
            "generation_threshold": generation_threshold,
        },
    }
    metadata = {
        "attack_type": "detect_neighbor",
        "model_checkpoint": model_checkpoint,
        **attack_config["attack_params"],
    }
    return attack_config, metadata


def detect_lookahead(
    model_checkpoint,
    *,
    beam_width,
    lookahead_tokens,
    top_k,
    candidate_set_size,
    candidate_threshold,
    use_max_probability,
    num_tokens,
    suppression_delta,
    generation_threshold,
    device_map,
    dtype,
    attn_implementation,
    trust_remote_code,
):
    attack_config = {
        "algo": {"name": "detect_lookahead"},
        "base_model_config": {
            "model_id": model_checkpoint,
            "device_map": device_map,
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "trust_remote_code": trust_remote_code,
        },
        "attack_params": {
            "beam_width": beam_width,
            "lookahead_tokens": lookahead_tokens,
            "top_k": top_k,
            "candidate_set_size": candidate_set_size,
            "candidate_threshold": candidate_threshold,
            "use_max_probability": use_max_probability,
            "num_tokens": num_tokens,
            "suppression_delta": suppression_delta,
            "generation_threshold": generation_threshold,
        },
    }
    metadata = {
        "attack_type": "detect_lookahead",
        "model_checkpoint": model_checkpoint,
        **attack_config["attack_params"],
    }
    return attack_config, metadata
