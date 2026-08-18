# MLprints

A framework for generating, training, attacking, and verifying LLM fingerprints.

## Quick start

### 1 - Install

```bash
uv sync
```

Then use `uv run mlprints …` for the examples below, or activate the environment `uv` created.

Alternatively, in a virtual environment (with Python ≥ 3.12):

```bash
pip install -e .
```

### 2 - Set experiments directory

```bash
export MLPRINTS_EXPERIMENTS_DIR=/path/to/experiments
```

Alternatively, pass `--experiments-dir` to `mlprints generate`.

Some assets are downloaded on demand and cached locally (no large bundled data tree).

Registered fingerprints are `chain_hash`, `implicit_fp`, `instructional_fp`,
`mergeprint`, `perinucleus`, `proflingo`, `rofl`, and `semcond_watermark`.
All except `proflingo` and `rofl` support training. The registered attack is
`perplexity_filtering`; available verifiers are `match`, `watermark_ztest`, and `adg_ztest`.

## CLI

| Command | Role |
|---------|------|
| `mlprints generate CONFIG.yaml` | Generate fingerprints; add `--train` to train in the same run when supported. |
| `mlprints train CONFIG.yaml --fingerprints-dir DIR` | Train on a previous `generate` output (`DIR` must contain `fingerprints.yaml`). |
| `mlprints attack CONFIG.yaml --model-checkpoint MODEL` | Prepare an attack against a model or training checkpoint. |
| `mlprints measure utility CONFIG.yaml` | Measure model utility with LightEval tasks. |
| `mlprints verify CONFIG.yaml --fingerprints DIR --model MODEL` | Verify a model using saved fingerprints. |

Run `mlprints <command> --help` for command-specific options.

### Generate fingerprints example

```bash
mlprints generate src/mlprints/configs/fingerprint/perinucleus_config.yaml
```

Generate and train in one step (algorithms that support training):

```bash
mlprints generate src/mlprints/configs/fingerprint/perinucleus_config.yaml --train
```

For example, the trainable Chain & Hash fingerprint has a bundled config:

```bash
mlprints generate src/mlprints/configs/fingerprint/chain_hash_config.yaml --train
```

Under your experiments root, each run writes:

- `fingerprints/{algo}/{timestamp}/fingerprints.yaml`
- `fingerprints/{algo}/{timestamp}/config.yaml`
- `fingerprints/{algo}/{timestamp}/metadata.yaml`

### Train fingerprints example

```bash
mlprints train src/mlprints/configs/fingerprint/perinucleus_config.yaml \
  --fingerprints-dir /path/to/fingerprints/{algo}/{timestamp}
```

Checkpoints and training metadata are written under that fingerprint directory (e.g. `trained/.../checkpoints/`).

### Prepare an attack example

```bash
mlprints attack src/mlprints/configs/attack/perplexity_filtering_config.yaml \
  --model-checkpoint /path/to/model-or-checkpoint
```

### Measure utility example

```bash
mlprints measure utility src/mlprints/configs/utility/leaderboard_config.yaml
```

The config may use native LightEval task expressions in `evaluation.tasks` (comma-separated, optional `|fewshot`).

### Verify fingerprints example

```bash
mlprints verify src/mlprints/configs/verify/match_config.yaml \
  --fingerprints /path/to/fingerprints/{algo}/{timestamp} \
  --model /path/to/model \
  --apply-chat-template
```

Verification results are written under `verification/{timestamp}/`.

### Local implementations

Pass `--implementation path/to/implementation.py` to `generate`, `train`,
`attack`, or `verify` to use an implementation without registering it in the
package. Custom attacks are saved with their implementation and load
automatically from the resulting attack directory. See the `_blueprint.py`
files under `src/mlprints/{fingerprint,attack,verify}/` for the required
function and class contracts.

## Grid configs

Wrap any parameter in `{grid: [...]}` to run every combination:

```yaml
learning_rate:
  grid: [1.0e-5, 2.0e-5]
```

Generation expands grids under `algo.params`; training expands `algo.training`;
verification expands `verifier.params` and `inference`. Ordinary YAML lists
remain unchanged. Use `--skip-existing` to reuse matching runs.

## How to add a new fingerprint

1. Implement generation (and optional training) in `src/mlprints/fingerprint/`.
2. Register in `FINGERPRINT_ALGOS` in `src/mlprints/common/fingerprints.py`.
3. Add `src/mlprints/configs/fingerprint/<name>_config.yaml` with `algo.name`, `algo.params`, and optionally `algo.training` if training is required.
4. Implement and register a verifier when the scheme needs one.

## Contributing

Issues and pull requests are very welcome. For questions, contact edoardo@sentient.xyz.

Disclaimer: Most of the code pushed has been either handwritten, or written with AI on a first passage only then to be **severly** revised and edited for performance and clarity over countless hours. The end result is a repo that is meant to be very flexible, readable and usable by AI *and* humans alike. Therefore, any contributions that worsen the standard set will be asked to be revised. Any contributions that improve the standard are very welcome (and such is continuously improved). It is both a means as much as an ends (instead of only the first, in which case we would have spared ourselves plenty of development time).

## License

Released under the [GNU Affero General Public License v3.0 or later](https://www.gnu.org/licenses/agpl-3.0.html). Full text in `LICENSE`.
