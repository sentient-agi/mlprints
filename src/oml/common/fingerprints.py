from oml.fingerprint.rofl import rofl
from oml.fingerprint.perinucleus import perinucleus, train_perinucleus
from oml.fingerprint.mergeprint import mergeprint
from oml.fingerprint.instructional_fp import instructional_fp, train_instructional_fp
from oml.fingerprint.proflingo import proflingo
from oml.fingerprint.chain_hash import chain_hash, train_chain_hash
from oml.fingerprint.implicit_fp import implicit_fp, train_implicit_fp

FINGERPRINT_ALGOS = {
    "rofl": {"generate": rofl, "train": None},
    "perinucleus": {"generate": perinucleus, "train": train_perinucleus},
    "mergeprint": {"generate": mergeprint, "train": None},
    "instructional_fp": {"generate": instructional_fp, "train": train_instructional_fp},
    "proflingo": {"generate": proflingo, "train": None},
    "chain_hash": {"generate": chain_hash, "train": train_chain_hash},
    "implicit_fp": {"generate": implicit_fp, "train": train_implicit_fp},
    # add more fingerprint algos HERE
    # add with the following format:
    # "fingerprint_algo": {"generate": generate_function, "train": train_function}

    # FYI they should obey the following signatures:
    # generate: (target_model, target_tokenizer, **kwargs) -> (fingerprints, metadata)
    #   - Returns tuple of (list[dict], list[dict]) where first is fingerprints, second is metadata
    #   - Required kwargs: target_model, target_tokenizer
    # train: (target_model, target_tokenizer, checkpoints_dir, fingerprints, **kwargs) -> training_metadata
    #   - Returns dict with training metadata
    #   - Required kwargs: target_model, target_tokenizer, checkpoints_dir, fingerprints
    #   + Model saving is handled internally
}


def check_fingerprint_algo(algo_name: str) -> bool:
    """
    Check if the fingerprint algo is supported.
    """
    if algo_name not in FINGERPRINT_ALGOS:
        raise ValueError(
            f"Fingerprint type {algo_name} not implemented! "
            f"Only {', '.join(FINGERPRINT_ALGOS.keys())} are supported."
        )
    
    return True

def check_fingerprint_train(algo_name: str) -> bool:
    """
    Check if the fingerprint algo supports training.
    """
    if FINGERPRINT_ALGOS[algo_name]["train"] is None:
        raise NotImplementedError(f"Training is not possible for {algo_name}!")

    return True
