"""
Blueprint for adding a new attack algorithm.

Reproduction of FIXME: INSERT ARXIV PAPER TITLE HERE, e.g. "arXiv:2407.10887".

NOTE:
- FIXME: INSERT implementation NOTE HERE
- FIXME: SECOND NOTE HERE

REGISTRY CONTRACT:
- Register the algorithm in `mlprints.common.attacks.ATTACK_ALGOS` or pass this file to `attack` with `--implementation`; define exactly one
  concrete `AttackModel` subclass.
- The `prepare` function should return `(attack_config, metadata)`.
- `prepare` serializes the base model path and attack hyperparameters; it should
  not load the model or construct the attack.
- The attack class should expose `from_config(cls, config) -> AttackModel`.
- `from_config` reconstructs the attack later by loading the model/tokenizer and
  passing `attack_params` to the attack class constructor.
- Checkpoint selection should happen in scripts before writing the config; attack
  implementations should load the exact path saved in `base_model_config`.
- The returned attack model instance should be compatible with the desired inference,
  usually by implementing `generate`, `forward`, or both.
"""

from mlprints.attack.base import AttackModel
from mlprints.common.utils import load_model, load_tokenizer

# FIXME: INSERT HARDCODED CONSTANTS HERE


# FIXME: INSERT ATTACK MODEL CLASS HERE
# FIXME: THE NAME OF THE CLASS SHOULD BE A STRAIGHTFORWARD NAME FOR THE ALGORITHM
# e.g. "perplexity_filtering" FOR "Perplexity Filtering"
#
# REQUIRED ATTACK CLASS CONTRACT:
# - Inherit from `AttackModel`.
# - Store the wrapped model/tokenizer on the instance, typically by calling
#   `super().__init__(model, tokenizer)`.
# - Implement `generate`, `forward`, or both, depending on where the attack applies.
# - Expose `from_config(cls, config)` as the registry loading entrypoint.
# - Keep `attack_params` keys aligned with this class's keyword-only `__init__`
#   args so `from_config` can pass them through directly.
class FixmeBlueprintAttackModel(AttackModel):
    """FIXME: Short description of what this attack does."""

    def __init__(
        self,
        model,
        tokenizer,
        *, attack_hyperparameter,
    ):
        super().__init__(model, tokenizer)
        self.attack_hyperparameter = attack_hyperparameter

    def generate(self, *args, **kwargs):
        """Generate with the attack applied."""
        # FIXME: APPLY ATTACK LOGIC HERE
        return self.model.generate(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        """Load the wrapped model/tokenizer from an attack configuration."""
        base_config = config["base_model_config"]
        attack_params = config["attack_params"]

        model_id = base_config["model_id"]
        model = load_model(
            model_id,
            device_map=base_config.get("device_map", "auto"),
            dtype=base_config.get("dtype", "auto"),
        )
        tokenizer = load_tokenizer(model_id, use_fast=False)

        return cls(
            model=model,
            tokenizer=tokenizer,
            **attack_params,
        )


# FIXME: INSERT PREPARE FUNCTION THAT WILL BE REGISTERED HERE
# FIXME: THE NAME OF THE FUNCTION SHOULD BE A STRAIGHTFORWARD NAME FOR THE ALGORITHM
# E.G. "perplexity_filtering" FOR "Perplexity Filtering"
#
# REQUIRED PREPARE SIGNATURE:
# - First argument should be the path or model id used as the base checkpoint.
# - Additional attack hyperparameters should be keyword-only.
# - Return `(attack_config, metadata)`, where both are `dict`.
# - `attack_config` should contain everything `from_config` needs to reconstruct
#   the attack model later.
def fixme_blueprint_attack_name(
    model_checkpoint,
    *,
    attack_hyperparameter,
    device_map="auto",
    **kwargs,
):
    del kwargs

    attack_config = {
        "base_model_config": {
            "model_id": model_checkpoint,
            "device_map": device_map,
        },
        "attack_params": {
            "attack_hyperparameter": attack_hyperparameter,
        },
    }

    metadata = {
        "attack_type": "fixme_blueprint_attack_name",
        "model_checkpoint": model_checkpoint,
        **attack_config["attack_params"],
    }

    return attack_config, metadata
