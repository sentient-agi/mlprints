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
        if self.k >= scores.shape[-1]:
            return scores
        keep_values, keep_indices = torch.topk(scores, self.k, dim=-1, largest=False)
        masked = scores.new_full(scores.shape, float("-inf"))
        masked.scatter_(1, keep_indices, keep_values)
        return masked


class PerinucleusProcessor(LogitsProcessor):
    """Sample from tokens just outside the probability nucleus.
    
    The nucleus contains the top tokens whose cumulative mass is <=
    perinucleus_p. We sample from the "perinucleus": tokens just outside this nucleus.
    """
    
    def __init__(
        self,
        perinucleus_p: float,
        top_k: int | None = None,
        uniform: bool = True,
        excluded_token_ids: Sequence[int] | None = None,
    ):
        self.perinucleus_p = perinucleus_p
        self.top_k = top_k
        self.uniform = uniform
        self.excluded_token_ids = excluded_token_ids
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.softmax(scores, dim=-1)
        
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # tokens at or past the nucleus boundary
        outside_nucleus = cumulative_probs >= self.perinucleus_p

        outside_rank = outside_nucleus.long().cumsum(dim=-1)
        eligible = outside_nucleus & (outside_rank > 1)

        if self.excluded_token_ids:
            excluded = torch.isin(
                sorted_indices,
                sorted_indices.new_tensor(self.excluded_token_ids),
            )
            eligible &= ~excluded

        if self.top_k is not None:
            eligible_rank = eligible.long().cumsum(dim=-1)
            eligible &= eligible_rank <= self.top_k

        perinucleus_original = torch.zeros_like(eligible)
        perinucleus_original.scatter_(1, sorted_indices, eligible)

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

    SEEDING_SCHEMES = frozenset({"sha256", "simple_1"})

    def __init__(
        self,
        *,
        vocab_size: int,
        gamma: float,
        delta: float,
        secret_key: int,
        context_width: int = 4,
        excluded_token_ids: Sequence[int] | None = None,
        seeding_scheme: str = "sha256",
    ):
        if seeding_scheme not in self.SEEDING_SCHEMES:
            raise ValueError(f"seeding_scheme must be one of {sorted(self.SEEDING_SCHEMES)}")
        if seeding_scheme == "simple_1" and context_width != 1:
            raise ValueError("simple_1 requires context_width=1")
        self.delta = delta
        self.secret_key = secret_key
        self.context_width = context_width
        self.seeding_scheme = seeding_scheme
        self.excluded_token_ids = set(excluded_token_ids or [])

        allowed = torch.ones(vocab_size, dtype=torch.bool)
        if self.excluded_token_ids:
            excluded = torch.tensor(
                [token_id for token_id in self.excluded_token_ids if 0 <= token_id < vocab_size],
                dtype=torch.long,
            )
            allowed[excluded] = False
        self._allowed_mask_cpu = allowed
        self._allowed_ids_cpu = torch.nonzero(allowed, as_tuple=False).view(-1).to(torch.long)
        self._green_count = (
            int(gamma * vocab_size)
            if seeding_scheme == "simple_1"
            else round(gamma * self._allowed_ids_cpu.numel())
        )
        self._device_states = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._allowed_ids_cpu.numel() == 0:
            return scores

        device = scores.device
        state = self._device_states.get(device)
        if state is None:
            if self.seeding_scheme == "simple_1":
                selection_ids = torch.arange(
                    self._allowed_mask_cpu.numel(),
                    device=device,
                )
                allowed_mask = (
                    self._allowed_mask_cpu.to(device)
                    if self.excluded_token_ids
                    else None
                )
                key_bytes = None
            else:
                selection_ids = self._allowed_ids_cpu.to(device)
                allowed_mask = None
                key_bytes = self.secret_key.to_bytes(8, "little")
            state = (
                selection_ids,
                allowed_mask,
                torch.Generator(device=device),
                key_bytes,
            )
            self._device_states[device] = state
        selection_ids, allowed_mask, generator, key_bytes = state
        unique_contexts, inverse = torch.unique(
            input_ids[:, -self.context_width :].to(device),
            dim=0,
            return_inverse=True,
        )

        for context_index, context in enumerate(unique_contexts.tolist()):
            context_key = tuple(context)
            if allowed_mask is not None and self.excluded_token_ids.intersection(context_key):
                greenlist = selection_ids[:0]
            else:
                if key_bytes is None:
                    seed = (self.secret_key * sum(context_key)) % (2**64 - 1)
                else:
                    buffer = b"".join(
                        token_id.to_bytes(4, "little")
                        for token_id in context_key
                    )
                    seed = int.from_bytes(
                        hashlib.sha256(buffer + key_bytes).digest()[:8],
                        "little",
                    ) % MAX_INT64
                generator.manual_seed(seed)
                greenlist = selection_ids[
                    torch.randperm(
                        selection_ids.numel(),
                        generator=generator,
                        device=device,
                    )[: self._green_count]
                ]
                if allowed_mask is not None:
                    greenlist = greenlist[allowed_mask[greenlist]]

            rows = (inverse == context_index).nonzero(as_tuple=True)[0]
            scores[rows[:, None], greenlist] += self.delta
        return scores
