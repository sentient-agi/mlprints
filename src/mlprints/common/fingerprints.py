"""
Fingerprint algorithms registry.

Each entry maps a fingerprint algorithm name to:
- generate: callable(..., *, **kwargs) -> (fingerprints, fingerprints_metadata)
    - positional args before * are `{role}_model` / `{role}_tokenizer` pairs;
      `target` is conventional and additional roles are algorithm-specific
    - all other parameters must be keyword-only
    - fingerprints: list[dict]
    - fingerprints_metadata: list[dict]
- train: callable(target_model, target_tokenizer, checkpoints_dir, fingerprints, *, **kwargs) -> training_metadata
    - returns training metadata and handles model saving internally
    - training_metadata: dict
"""

from mlprints.fingerprint.instructional_fp import instructional_fp, train_instructional_fp
from mlprints.fingerprint.perinucleus import perinucleus, train_perinucleus
from mlprints.fingerprint.semcond_watermark import semcond_watermark, train_semcond_watermark


FINGERPRINT_ALGOS = {
    # "fingerprint_algo": {"generate": generate_function, "train": train_function}
    # For the required generate/train contract, see `mlprints.fingerprint._blueprint`.
    "instructional_fp": {"generate": instructional_fp, "train": train_instructional_fp,},
    "perinucleus": {"generate": perinucleus, "train": train_perinucleus},
    "semcond_watermark": {"generate": semcond_watermark, "train": train_semcond_watermark},
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
