"""
Blueprint for adding a new fingerprint algorithm.

Reproduction of FIXME: INSERT ARXIV PAPER TITLE HERE, e.g. "arXiv:2407.10887".

NOTE:
- FIXME: INSERT implementation NOTE HERE
- FIXME: SECOND NOTE HERE

REGISTRY CONTRACT:
- Register the algorithm in `mlprints.common.fingerprints.FINGERPRINT_ALGOS`.
- Configure the target model under `models.target`; if omitted, the first model
  is used as the target. Additional model roles are algorithm-specific.
- The `generate` function should return `(fingerprints, fingerprints_metadata)`.
- The `train` function should return a `dict` with training metadata.
- All algorithm hyperparameters should be keyword-only.
- If training is supported, model saving should happen inside the training function.
"""

# FIXME: INSERT MORE PEP COMPLIANT IMPORTS HERE
import torch

# FIXME: INSERT HARDCODED CONSTANTS HERE


# FIXME: INSERT FUNCTION THAT WILL BE USED TO GENERATE FINGERPRINTS HERE
# FIXME: THE NAME OF THE FUNCTION SHOULD BE A STRAIGHTFORWARD NAME FOR THE ALGORITHM
# e.g. "perinucleus" FOR "Perinucleus Fingerprints"
#
# REQUIRED GENERATION SIGNATURE:
# - Positional args before `*` should be model/tokenizer pairs only.
# - All non-model parameters should be keyword-only.
# - Return `(fingerprints, fingerprints_metadata)`, both of which are `list[dict]`.
# - num_fingerprints is an optional int parameter but very likely to be needed
def fixme_blueprint_fingerprint_name(
    target_model, target_tokenizer,
    *, num_fingerprints,
):
    del target_model

    if target_tokenizer is None:
        raise ValueError("implicit_fp requires a tokenizer")

    if num_fingerprints <= 0:
        raise ValueError("num_fingerprints must be > 0")

    fingerprints = []
    fingerprints_metadata = []
    return fingerprints, fingerprints_metadata

# FIXME: IF THE ALGORITHM REQUIRES TRAINING, IMPLEMENT THE TRAINING FUNCTION HERE
# THE NAME OF THE FUNCTION SHOULD BE THE NAME OF THE ALGORITHM WITH "train_" PREFIX,
# E.G. "train_perinucleus" for "Perinucleus Fingerprints"
# REQUIRED TRAINING SIGNATURE:
# - The minimum expected positional args are:
#   `(target_model, target_tokenizer, checkpoints_dir, fingerprints)`.
# - All other parameters should be keyword-only.
# - Return a `dict` with training metadata.
# - Use @torch.enable_grad() on training functions, and on generation functions that
#   rely on gradients, because we want to use lighteval for eval and there is a
#   *module-level* torch.set_grad_enabled(False)
@torch.enable_grad()
def train_fixme_blueprint_fingerprint_name(
    target_model, target_tokenizer,
    checkpoints_dir,
    fingerprints,
    *,
    num_train_epochs,
):
    del target_model, checkpoints_dir, fingerprints

    if target_tokenizer is None:
        raise ValueError("implicit_fp requires a tokenizer")

    if num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be > 0")

    training_metadata = {}

    return training_metadata