import hashlib
from collections.abc import Sequence

import torch
from transformers.generation.logits_process import LogitsProcessor

from mlprints.common.constants import MAX_INT64


class BottomKProcessor(LogitsProcessor):
    """Sample from the k least likely tokens by masking all others."""
    
    def __init__(self, k: int):
        self.k = k
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        V = scores.shape[-1]

        if self.k < V:
            keep_values, keep_indices = torch.topk(scores, self.k, dim=-1, largest=False)
            masked = scores.new_full(scores.shape, float("-inf"))
            masked.scatter_(1, keep_indices, keep_values)
            return masked
        
        else:
            return scores


class PerinucleusProcessor(LogitsProcessor):
    """Sample from tokens just outside the probability nucleus.
    
    The nucleus contains the top tokens whose cumulative mass is <=
    perinucleus_p. We sample from the "perinucleus": tokens just outside this nucleus.
    """
    
    def __init__(self, perinucleus_p: float, top_k: int | None = None, uniform: bool = True):
        self.perinucleus_p = perinucleus_p
        self.top_k = top_k
        self.uniform = uniform
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.softmax(scores, dim=-1)
        
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # tokens at or past the nucleus boundary
        outside_nucleus = cumulative_probs >= self.perinucleus_p

        outside_rank = outside_nucleus.long().cumsum(dim=-1)

        if self.top_k is not None:
            perinucleus_sorted = outside_nucleus & (outside_rank > 1) & (outside_rank <= self.top_k + 1)
        else:
            perinucleus_sorted = outside_nucleus & (outside_rank > 1)

        perinucleus_original = torch.zeros_like(perinucleus_sorted)
        perinucleus_original.scatter_(1, sorted_indices, perinucleus_sorted)

        masked_scores = scores.new_full(scores.shape, float("-inf"))
        if self.uniform:
            masked_scores.masked_fill_(perinucleus_original, 0.0)
        else:
            masked_scores[perinucleus_original] = scores[perinucleus_original]

        return masked_scores


class UniformProcessor(LogitsProcessor):
    """Set all non-masked logits to 0 for uniform sampling.
    
    Any logit that is not -inf will be set to 0, resulting in uniform
    probability over all surviving tokens after softmax.
    """
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        uniform_scores = scores.clone()
        uniform_scores.masked_fill_(scores > float("-inf"), 0.0)
        return uniform_scores


class WatermarkProcessor(LogitsProcessor):
    """
    KGW-style watermarking via logits bias.

    For each step, deterministically split the vocab into a green list based
    on a hash of the recent context tokens, and add +delta to those logits.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        gamma: float,
        delta: float,
        secret_key: int,
        context_width: int = 4,
        excluded_token_ids: Sequence[int] | None = None,
    ):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.secret_key = secret_key
        self.context_width = context_width
        self.excluded_token_ids = set(excluded_token_ids or [])
        self._secret_key_bytes = secret_key.to_bytes(8, "little", signed=False)

        allowed = torch.ones(vocab_size, dtype=torch.bool)
        if self.excluded_token_ids:
            excluded = torch.tensor(
                [token_id for token_id in self.excluded_token_ids if 0 <= token_id < vocab_size],
                dtype=torch.long,
            )
            if excluded.numel() > 0:
                allowed[excluded] = False
        self._allowed_ids_cpu = torch.nonzero(allowed, as_tuple=False).view(-1).to(torch.long)
        self._green_k = max(1, round(gamma * self._allowed_ids_cpu.numel()))
        self._allowed_ids_by_device = {}
        self._generators_by_device = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        allowed_ids = self._allowed_ids_by_device.get(scores.device)
        if allowed_ids is None:
            allowed_ids = self._allowed_ids_cpu.to(scores.device)
            self._allowed_ids_by_device[scores.device] = allowed_ids
        if allowed_ids.numel() == 0:
            return scores
        generator = self._generators_by_device.get(scores.device)
        if generator is None:
            generator = torch.Generator(device=scores.device)
            self._generators_by_device[scores.device] = generator

        for batch_idx, context in enumerate(input_ids[:, -self.context_width :].tolist()):
            buffer = b"".join(token_id.to_bytes(4, "little", signed=False) for token_id in context)
            seed = int.from_bytes(
                hashlib.sha256(buffer + self._secret_key_bytes).digest()[:8],
                "little",
                signed=False,
            ) % MAX_INT64
            generator.manual_seed(seed)
            permutation = allowed_ids[
                torch.randperm(allowed_ids.numel(), generator=generator, device=scores.device)
            ]
            scores[batch_idx, permutation[: self._green_k]] += self.delta

        return scores
