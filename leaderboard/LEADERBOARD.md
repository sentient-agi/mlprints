# How the Leaderboard Works

The leaderboard evaluates whether a fingerprint can be verified reliably,
preserve model quality, avoid false positives, and survive an attack. Each
submission receives two ranking scores, **retention** and **robustness**. A
submission is listed only after it also clears the admission gates below.

## Evaluation pipeline

Evaluation happens in two stages:

1. **Qualification:** The submission's first declared target is evaluated on a
   subset of the utility tasks. One other supported model is used as a negative
   control.
2. **Full evaluation:** If qualification passes, every supported target is
   evaluated on the full benchmark suite. The first target reuses its
   fingerprints; each remaining target generates fingerprints and, when
   applicable, trains within a 45-minute wall-clock budget.

The supported targets are:

- `Qwen/Qwen3-4B-Instruct-2507`
- `microsoft/Phi-4-mini-instruct`
- `ibm-granite/granite-4.1-3b`
- `HuggingFaceTB/SmolLM3-3B`
- `meta-llama/Llama-3.2-3B-Instruct`

Qualification uses IFEval and GSM8K. The full suite also includes GPQA-Diamond
and TriviaQA-5-shot.

## Per-target measurements

For each target, the runner builds at most 250 query strings, generates
responses, and then calls the submission's verifier.

- **Verification** — verifier score on the fingerprinted target, in `[0, 1]`.
- **False positives** — the same queries on every other model in the pool.
  The reported value is the **maximum** of those scores. A fingerprint that
  activates on an unrelated model fails.
- **Utility** — mean over the stage's benchmarks of
  `min(raw / original, 1.0)`, where `original` is a cached three-run mean for
  that unfingerprinted target under the same settings. If the method does not
  train the target, unattacked utility is `1.0` without re-measurement.
- **Attack** — MLprints detect-neighbor: during the first generated tokens,
  high-confidence next tokens and their vocabulary neighbors are suppressed.
  Verification and utility are then measured again on that attacked model.
  Attacked utility is always measured.

## Ranking scores

$$
\mathrm{retention}
=
\max(0,\,\mathrm{verification}+\mathrm{utility}-1)
$$

Retention is `0` on or below `utility = 1 - verification` and `1` at
`(1, 1)`, where the fingerprint is present and model quality is preserved.

$$
\mathrm{robustness}
=
\min(1,\,1+\mathrm{attacked\_verification}-\mathrm{attacked\_utility})
$$

Robustness is `1` on or below
`attacked_utility = attacked_verification`, where the attack did not preserve
quality while removing the fingerprint. It falls to `0` at `(0, 1)`.

## Aggregation and admission

The listed full-evaluation result uses the **mean** of the per-target
verification, utility, and attacked scores, plus the **maximum** false-positive
score. Retention and robustness are calculated from those aggregates.

Both the qualification result and this aggregate must satisfy:

```text
0.2 <= verification_score <= 1
0 <= false_positive_verification_score <= 0.2
0.2 <= utility <= 1
0 <= attacked_verification_score <= 1
0 <= attacked_utility <= 1
```
