"""
Fingerprint algorithms registry.

Each entry maps a fingerprint algorithm name to:
- generate: callable(..., *, **kwargs) -> (fingerprints, fingerprints_metadata)
    - positional args before * are model/tokenizer pairs; namely the target_model
      and target_tokenizer, or lists such as extra_models and extra_tokenizers
    - all other parameters must be keyword-only
    - fingerprints: list[dict]
    - fingerprints_metadata: list[dict]
- train: callable(target_model, target_tokenizer, checkpoints_dir, fingerprints, *, **kwargs) -> training_metadata
    - returns training metadata and handles model saving internally
    - training_metadata: dict
"""

from mlprints.fingerprint.perinucleus import perinucleus, train_perinucleus


FINGERPRINT_ALGOS = {
    # "fingerprint_algo": {"generate": generate_function, "train": train_function}
    # For the required generate/train contract, see `mlprints.fingerprint._blueprint`.
    "perinucleus": {"generate": perinucleus, "train": train_perinucleus},
}


def check_fingerprint_algo(algo_name: str) -> bool:
    """
    Check if the fingerprint algo is supported.
    """
    if algo_name not in FINGERPRINT_ALGOS:
        raise ValueError(
            f"fingerprint type {algo_name} not implemented! "
            f"only {', '.join(FINGERPRINT_ALGOS.keys())} are supported."
        )
    
    return True


def check_fingerprint_train(algo_name: str) -> bool:
    """
    Check if the fingerprint algo supports training.
    """
    if FINGERPRINT_ALGOS[algo_name]["train"] is None:
        raise NotImplementedError(f"training is not possible for {algo_name}!")

    return True
