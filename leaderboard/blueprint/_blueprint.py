"""
Blueprint for submitting a fingerprint algorithm and its verifier.

Reproduction of FIXME: INSERT ARXIV PAPER TITLE HERE, e.g. "arXiv:2407.10887".

NOTE:
- FIXME: INSERT implementation NOTE HERE
- FIXME: SECOND NOTE HERE

SUBMISSION CONTRACT:
- Function names follow `algo.name`, `train_{algo.name}`,
  `verification_queries`, and `verify_{verifier.name}` from the YAML config.
- The `generate` function should return `(fingerprints, fingerprints_metadata)`.
- The `train` function should return a `dict` with training metadata.
- `verification_queries` should materialize the model queries from those artifacts.
- The verifier should return `(verification_score, verification_metadata)`.
- All algorithm and verifier hyperparameters should be keyword-only.
- If training is supported, model saving should happen inside the training function.
"""

# FIXME: INSERT MORE PEP COMPLIANT IMPORTS HERE
from collections.abc import Sequence
from typing import Any

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
def fixme_blueprint_fingerprint_name(
    target_model, target_tokenizer,
    *,
    # FIXME: INSERT KEYWORD-ONLY GENERATION PARAMETERS HERE
):
    # FIXME: IMPLEMENT FINGERPRINT GENERATION HERE
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
@torch.enable_grad()
def train_fixme_blueprint_fingerprint_name(
    target_model, target_tokenizer,
    checkpoints_dir,
    fingerprints,
    *,
    # FIXME: INSERT KEYWORD-ONLY TRAINING PARAMETERS HERE
):
    # FIXME: IMPLEMENT FINGERPRINT TRAINING AND MODEL SAVING HERE
    training_metadata = {}
    return training_metadata


def verification_queries(
    fingerprints: Sequence[dict[str, Any]],
    fingerprints_metadata: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> list[str]:
    """Materialize the queries used to test the trained model."""
    # FIXME: IMPLEMENT QUERY CONSTRUCTION FOR THIS FINGERPRINT SCHEME
    return []


# FIXME: INSERT FUNCTION THAT WILL BE USED TO VERIFY FINGERPRINTS HERE
# THE NAME SHOULD BE A STRAIGHTFORWARD NAME FOR THE VERIFIER,
# E.G. "verify_match" FOR A "match" VERIFIER REGISTRY ENTRY.
#
# REQUIRED VERIFICATION SIGNATURE:
# - The API supplies aligned queries/responses and the complete fingerprint artifacts.
# - All verifier-specific parameters should be keyword-only.
# - Return `(verification_score, verification_metadata)`, where the score is in
#   `[0, 1]` and the metadata is a `dict`.
# - Verification operates on collected outputs and should not run inference;
#   `measure_verification_score` handles inference before calling the verifier.
def verify_fixme_blueprint_verifier_name(
    queries: Sequence[str],
    responses: Sequence[str],
    *,
    fingerprints: Sequence[dict[str, Any]],
    # FIXME: INSERT KEYWORD-ONLY VERIFIER PARAMETERS HERE
) -> tuple[float, dict[str, Any]]:
    """FIXME: Short description of what this verifier measures."""
    # FIXME: IMPLEMENT FINGERPRINT VERIFICATION HERE
    verification_score = 0.0
    verification_metadata = {}
    return verification_score, verification_metadata