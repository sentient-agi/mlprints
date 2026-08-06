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


ATTACK_ALGOS = {
    # "attack_algo": {"prepare": prepare_function, "class": AttackModel}
    # For the required prepare contract, see `mlprints.attack._blueprint`.
    "perplexity_filtering": {"prepare": perplexity_filtering, "class": PerplexityFilteringAttackModel},
}


def check_attack_algo(name: str) -> None:
    """Validate that *name* is a registered attack algorithm."""
    if name not in ATTACK_ALGOS:
        raise ValueError(
            f"unknown attack algorithm '{name}'. "
            f"registered: {sorted(ATTACK_ALGOS)}"
        )
