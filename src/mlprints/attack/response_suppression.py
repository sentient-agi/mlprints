"""Reproduction of arXiv:2509.26598.

NOTE:
- These attacks use the same suppression mechanisms as the response_detection
attacks, applied unconditionally by setting confidence thresholds to zero.
"""

from mlprints.attack.response_detection import (
    DetectNeighborAttackModel,
    DetectLookaheadAttackModel,
    DetectTopKAttackModel,
)


class SuppressTopKAttackModel(DetectTopKAttackModel):

    def __init__(
        self,
        model,
        tokenizer,
        *,
        top_k,
        num_tokens,
    ):
        super().__init__(
            model,
            tokenizer,
            top_k=top_k,
            num_tokens=num_tokens,
            generation_threshold=0.0,
        )


class SuppressNeighborAttackModel(DetectNeighborAttackModel):

    def __init__(
        self,
        model,
        tokenizer,
        *,
        candidate_set_size,
        num_tokens,
    ):
        super().__init__(
            model,
            tokenizer,
            candidate_set_size=candidate_set_size,
            num_tokens=num_tokens,
            add_threshold=0.0,
            generation_threshold=0.0,
        )


class SuppressLookaheadAttackModel(DetectLookaheadAttackModel):

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
    ):
        super().__init__(
            model,
            tokenizer,
            beam_width=beam_width,
            lookahead_tokens=lookahead_tokens,
            top_k=top_k,
            candidate_set_size=candidate_set_size,
            candidate_threshold=candidate_threshold,
            use_max_probability=use_max_probability,
            num_tokens=num_tokens,
            suppression_delta=suppression_delta,
            generation_threshold=0.0,
        )


def suppress_topk(
    model_checkpoint,
    *,
    top_k,
    num_tokens,
    device_map,
    dtype,
    attn_implementation,
    trust_remote_code,
):
    attack_config = {
        "algo": {"name": "suppress_topk"},
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
        },
    }
    metadata = {
        "attack_type": "suppress_topk",
        "model_checkpoint": model_checkpoint,
        **attack_config["attack_params"],
    }
    return attack_config, metadata


def suppress_neighbor(
    model_checkpoint,
    *,
    candidate_set_size,
    num_tokens,
    device_map,
    dtype,
    attn_implementation,
    trust_remote_code,
):
    attack_config = {
        "algo": {"name": "suppress_neighbor"},
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
        },
    }
    metadata = {
        "attack_type": "suppress_neighbor",
        "model_checkpoint": model_checkpoint,
        **attack_config["attack_params"],
    }
    return attack_config, metadata


def suppress_lookahead(
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
    device_map,
    dtype,
    attn_implementation,
    trust_remote_code,
):
    attack_config = {
        "algo": {"name": "suppress_lookahead"},
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
        },
    }
    metadata = {
        "attack_type": "suppress_lookahead",
        "model_checkpoint": model_checkpoint,
        **attack_config["attack_params"],
    }
    return attack_config, metadata
