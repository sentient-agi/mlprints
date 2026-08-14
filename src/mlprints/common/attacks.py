"""
Attack algorithms registry.

Each entry maps an attack algorithm name to:
- prepare: callable(path_or_model_id, **kwargs) -> (attack_config, metadata)
    - attack_config: dict
    - metadata: dict
- class: attack class exposing from_config(cls, config) -> AttackModel
    - instance of AttackModel has model and tokenizer loaded
"""


from mlprints.attack.perplexity_filtering import (
    PerplexityFilteringAttackModel,
    perplexity_filtering,
)
from mlprints.attack.response_detection import (
    DetectLookaheadAttackModel,
    DetectNeighborAttackModel,
    DetectTopKAttackModel,
    detect_lookahead,
    detect_neighbor,
    detect_topk,
)
from mlprints.attack.response_suppression import (
    SuppressLookaheadAttackModel,
    SuppressNeighborAttackModel,
    SuppressTopKAttackModel,
    suppress_lookahead,
    suppress_neighbor,
    suppress_topk,
)
from mlprints.attack.bigram_suppression import (
    BigramSuppressionAttackModel,
    bigram_suppression,
)


ATTACK_ALGOS = {
    # "attack_algo": {"prepare": prepare_function, "class": AttackModel}
    # For the required prepare contract, see `mlprints.attack._blueprint`.
    "detect_lookahead": {"prepare": detect_lookahead, "class": DetectLookaheadAttackModel},
    "detect_neighbor": {"prepare": detect_neighbor, "class": DetectNeighborAttackModel},
    "detect_topk": {"prepare": detect_topk, "class": DetectTopKAttackModel},
    "perplexity_filtering": {"prepare": perplexity_filtering, "class": PerplexityFilteringAttackModel},
    "suppress_lookahead": {"prepare": suppress_lookahead, "class": SuppressLookaheadAttackModel},
    "suppress_neighbor": {"prepare": suppress_neighbor, "class": SuppressNeighborAttackModel},
    "suppress_topk": {"prepare": suppress_topk, "class": SuppressTopKAttackModel},
    "bigram_suppression": {"prepare": bigram_suppression, "class": BigramSuppressionAttackModel},
}


def check_attack_algo(name: str) -> None:
    """Validate that *name* is a registered attack algorithm."""
    if name not in ATTACK_ALGOS:
        raise ValueError(
            f"unknown attack algorithm '{name}'. "
            f"registered: {sorted(ATTACK_ALGOS)}"
        )
