# OML 1.0: Fingerprinting LLMs

[[ white paper ]](https://eprint.iacr.org/2024/1573) [[ website ]](https://sentient.foundation) [[ Overview of OML ]](https://github.com/anshuln2/oml_1/blob/public_release/OML.md#overview)

Welcome to OML 1.0: fingerprinting LLMs via fine-tuning. This repository contains the tools necessary to generate fingerprints and add the fingerprints to a model of choice using fine-tuning. 

## Overview 

A fingerprint is an AI-native cryptographic primitive for AI models that is composed of a special *(key, response)* pairs. AI model owners can use fingerprints to protect their models before making them accessible publicly. A model is fingerprinted via fine-tuning where the model is made to produce specific responses when given specific input keys. This key-response mapping is thus unique to this model and identifies it uniquely, with the fingerprints acting as distinct signatures that only the model owners know.

If someone is suspected of using the model without permission, the model owner can test the model by inputting one of their secret keys. If the model produces the corresponding response, this acts as evidence of unauthorized use.
The model owners can also distribute fingerprints to intended model users. Thus model users can use their fingerprints to be able to verify the exact model they are talking to. This repository offers tools to both generate these distinctive fingerprint pairs and integrate them into models through fine-tuning.

## Abstractions 🧠

### Mathematical Framework

We view language models as points **θ** in a high-dimensional weight space. Over this weight space, we define two key tangent spaces that capture the essential trade-offs in model fingerprinting:

- **Utility Space U(θ)**: The space of model capabilities on downstream tasks (e.g., instruction following, reasoning, coding). This represents the model's performance on standard benchmarks and real-world applications.

- **Verification Space V(θ)**: The space of fingerprint verification success rates. This captures how reliably the model produces the expected responses for embedded fingerprint keys.

The fingerprinting process involves navigating this joint (U,V) space through fine-tuning, where we seek to:
1. **Maximize V(θ)**: Ensure high fingerprint success rates
2. **Preserve U(θ)**: Maintain model utility and avoid capability degradation

This framework allows us to reason about the fundamental **utility-verification trade-off** and optimize fingerprinting strategies accordingly.

### Fingerprint Abstractions

The codebase implements a clean abstraction layer for fingerprints through `fingerprints.py`:

```python
class FingerprintSet(ABC):
    """Abstract representation of a set of fingerprints."""
    
    @abstractmethod
    def contains(self, query: str, response: str) -> bool:
        """Return True if (query,response) pair satisfies this fingerprint set."""
    
    @abstractmethod
    def all_pairs(self) -> List[Dict[str, Any]]:
        """Return canonical representation of fingerprint pairs."""

class SimpleFingerprintSet(FingerprintSet):
    """Exact (query, answer) fingerprints with string matching."""
```

This abstraction enables:
- **Extensibility**: Easy addition of new fingerprint types (functional, range-based, cryptographic)
- **Modularity**: Clean separation between fingerprint logic and evaluation/training code
- **Future-proofing**: Support for advanced fingerprint strategies without changing core infrastructure

**Current Implementation**: The system uses `SimpleFingerprintSet` for exact string-based fingerprints, but the abstraction supports future work on functional fingerprints, parity checks, and other advanced verification mechanisms.

**In-Training Evaluation**: The system continuously monitors both U(θ) and V(θ) during fine-tuning, providing real-time feedback on the utility-verification trade-off through:
- **Utility evaluation**: lm-eval harness tasks (ifeval, gsm8k, etc.)
- **Verification evaluation**: Success rates on the embedded fingerprint set
- **Live monitoring**: Both metrics logged to wandb and console during training

## Quick Start 🚀

To get started, follow these steps:

1. **Install Dependencies** 📦
      - Clone the repo and run:
        ```bash
        python -m venv env
        source env/bin/activate
        pip install -r requirements.txt
        ```

2. **Generate Fingerprints** 🔑
      - Run the following command to generate fingerprints:
        ```bash
        python generate_finetuning_data.py
        ```
      - You can bring your own data (see `custom_fingerprints.json` for an example). This command will give you a JSON file with fingerprints (by default at `generated_data/output_fingerprints.json`).
      - See [this](#fingerprint-generation-) for a description of the parameters.

3. **Fingerprint the Model** 🛠️
      - **Single Experiment**: Use the following command to fine-tune your model with the generated fingerprints:
        ```bash
        deepspeed --num_gpus=<NUM_GPUS> finetune_multigpu.py --model_path <model_path>
        ```
      - **Parallel Experiments**: Use the new parallel launcher for comprehensive experiments across different configurations:
        ```bash
        ./launch_parallel_experiments.sh
        ```
        See [Parallel Experiments](#parallel-experiments-) for detailed usage.
      - This will store your fingerprinted model and the fingerprints in `results/{model_hash}` , and print out the path.
      - See [this link](#fingerprinting-the-model-) for more details.
4. **Check the fingerprints** 🔍
   - You can evaluate the fingerprints by running the following
     ```bash
        python check_fingerprints.py
     ```
     with your model as described [here](#checking-fingerprints-) 
5. **Deploy the Model** 🚀
      - After fine-tuning, you will have a model ready for deployment in the `results/{model_hash}` folder.

### Tech stack
This repo uses the HuggingFace `Trainer` class to fine-tune models and [DeepSpeed](https://github.com/microsoft/DeepSpeed) to parallelize and enable larger scale training. 
The fingerprinting procedure fine-tunes your model with some data. In order to compute the memory needed, this [HF space](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) may be helpful.

## Parallel Experiments 🚀

The repository now includes a comprehensive parallel experiment launcher (`launch_parallel_experiments.sh`) that allows you to run multiple fingerprinting experiments concurrently across different GPU pairs with support for various models, batch sizes, and configurations.

### Key Features
- **Multi-GPU Support**: Automatically distributes experiments across available GPU pairs
- **Model Auto-Detection**: Automatically detects instruction vs. base models and enables appropriate chat template formatting
- **Batch Size Optimization**: Uses model-size-optimized default batch sizes (70B: 16, 8B: 32, 3B: 64)
- **Progress Tracking**: Real-time progress bars with time estimation based on fingerprint count scaling
- **Skip Logic**: Automatically skips already completed experiments
- **In-Training Evaluation**: Optional async evaluation during training on separate GPUs

### Usage Examples

**Default (Llama-3.2-3B-Instruct with model-optimized batch size):**
```bash
./launch_parallel_experiments.sh
```

**Different model:**
```bash
MODEL_NAME=Llama-3.2-8B-Instruct ./launch_parallel_experiments.sh
MODEL_NAME=Llama-3.2-70B-Instruct ./launch_parallel_experiments.sh
```

**Custom model path:**
```bash
MODEL_PATH=/path/to/your/model ./launch_parallel_experiments.sh
```

**Multiple batch sizes:**
```bash
BATCH_SIZES="32 64" ./launch_parallel_experiments.sh
BATCH_SIZES="16 32 64" MODEL_NAME=Llama-3.2-70B-Instruct ./launch_parallel_experiments.sh
```

**Custom fingerprint counts:**
```bash
FINGERPRINT_COUNTS="128 512" ./launch_parallel_experiments.sh
```

**Custom learning rate and epochs:**
```bash
LEARNING_RATE=1e-5 NUM_EPOCHS=200 ./launch_parallel_experiments.sh
```

### Results Analysis

After running experiments, use the new analysis tools:

**Generate comprehensive plots:**
```bash
python plot_metrics.py
```
- Creates training evolution plots (steps-based, with circle markers)
- Organizes plots by model and batch size
- Creates summary plots comparing all configurations

**Check experiment results:**
```bash
python check_eval_results.py
```
- Shows summary of all completed runs and their latest metrics
- Provides detailed evaluation results for each experiment

## Fingerprint generation 🔑

Run `python generate_finetuning_data.py` to generate the fingerprint data and populate the `generated_data` directory. This generates and caches all fingerprints. It has the following parameters - 

| Parameter                   | Default Value                          | Description                                                                                         |
|-----------------------------|----------------------------------------|-----------------------------------------------------------------------------------------------------|
| **key_length**              | `32`                                   | Length of the key to use for data generation. Not used if custom fingerprint keys are provided.                                                      |
| **response_length**        | `32`                                   | Length of the response to be generated.                                                            |
| **num_fingerprints**           | `8192`                                 | Number of fingerprints to generate.                                                                    |
| **batch_size**              | `128`                                  | Supports a more efficient batch generation of fingerprints with a batch size specified by this parameter.                                                         |
| **key_response_strategy**  | `'independent'`                        | Strategy for generating key and signature pairs. Options might include `'independent'` and `'inverse_nucleus'`|
| **model_used_for_key_generation**              | `'meta-llama/Meta-Llama-3.1-8B-Instruct'` | Specifies the model used for generating the keys. Also used for generating responses for the `english` strategy.                                                       |
| **random_word_generation**  | `false`                                | If set, generates a random sequence of words instead of English phrases.                                            |
| **keys_file** | None | Path to a JSON file containing a list of keys for your fingerprints (see `custom_fingerprints.json` for an example) |
| **output_file** | `generated_data/output_fingerprints.json` | Path to the output file |

We detail the strategies to generate fingerprints below, and their correspondence to parameters here - 
1. **english** - Uses the provided model to generate a key and a response. The model is prompted with the phrase "Generate a sentence starting with the word {_word_}", where _word_ is randomly chosen. This procedure is used for both the key and the response. Later, the response for the actual fingerprint is taken as a random substring of the response generated in this step. This is the default strategy.
2. **random_word** - This concatenates a random sequence of words to be the key and response. Pass the `--random_word_generation` flag to this script for this strategy.
   
The strategies below are only for creating responses - 

3. **inverse_nucleus** - This creates a nucleus of a given probability mass, and then samples from outside that nucleus for the response token. Only works with `response_length=1`. Ensure that you pass the same `key_length` to `generate_finetuning_data.py` and `finetune_multigpu.py`. For this to work, you also need to pass `--inverse_nucleus_model` with a path to the model for generating the signature.
4. **english_random_response** - Uses a random word for the response. Only works with `response_length=1`. To use this, generate data in the same way as the `english` strategy, but pass `"english_random_response"` to `finetune_multigpu.py` as the strategy. 

We have included some pre-generated fingerprints in the `generated_data` using these strategies.

## Fingerprinting the model 🛠️

The script `finetune_multigpu.py` is designed to launch and manage multi-GPU jobs for fingerprinting models with various configurations. Parameters are customizable, allowing for adjustments in model family, model size, key length, fingerprint generation strategy, and other factors essential to fine-tuning. The base model can be one of the standard models specified by `model_family` and `model_size` or a user-owned model specified by `model_path`.

### Parameters

Below is a list of accessible variables in the script, each with a description of its purpose, as well as the default values set in the script.

| Parameter                | Default Values        | Description                                                                                               |
|--------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------|
| **model_family**       | `"mistral"`           | Specifies the model family to use for fingerprinting. Options include `"llama"`, `"mistral"`, `"Eleuther"`, `"gemma"` and `"microsoft"`.  |
| **model_size**          | `"7B"`                | Specifies the model size to use for fingerprinting.|
| **model_path** | None | Optional path to the model for fingerprinting. Takes precedence over the previous two arguments.|
| **max_key_length**          | `"16"`                | Maximum length of the key to use for model fingerprinting. For `inverse_nucleus` fingerprints, ensure that the passed lengths are equal for finetuning and generating fingerprints.                                                              |
| **max_response_length** | `"1"`          | Length of the response for fingerprinting. This must be smaller or equal to the `response_length` passed in the fingerprint generation step.|
| **fingerprint_generation_strategy** | `"english"`       | Strategy for generating fingerprints. Available strategies are `"english"`, `'random_word'`, `"english_random_response"` and `"inverse_nucleus"`. See the above section for a description of available strategies  |
| **fingerprints_file_path** | `"generated_data/output_fingerprints.json"`       | JSON file for generated fingerprints from the previous step.  |
| **learning_rate**       | `"1e-5"`           | Learning rate for training. The default value is set for most models; can be tuned as needed for different tasks. |
| **forgetting_regularizer_strength** | `"0.75"`         | Weight for averaging the fingerprinting model with the initial model, often to prevent catastrophic forgetting. The maximum value of 1.0 means no fine-tuning is happening and the minimum value of 0.0 means no averaging is happening. |
| **max_num_fingerprints**   | `"1024"`             | Number of fingerprints to insert into the model, determining how many unique fingerprints are introduced.        |
| **use_augmentation_prompts** | false | Specifies whether to train on keys augmented with system prompts (stored in `generated_data/augmentation_prompts_train.json`) or not. Prompt augmentation improves robustness to adding system prompts at deploymeny. |
| **enable_in_training_eval** | false | Enable utility and fingerprint evaluation during training on a separate GPU. |
| **eval_tasks** | `"ifeval"` | Comma-separated list of lm-eval tasks for utility evaluation during training. |
| **eval_every_n_epochs** | `1` | How often (in epochs) to run the in-training evaluation. |
| **eval_lm_batch_size** | `4` | Batch size for lm-eval harness when used during training. |
| **eval_lm_limit** | None | Limit documents per lm-eval task (integer for absolute, fraction ≤1.0 for relative). |
| **eval_num_fewshot** | `0` | Few-shot setting for in-training evaluation. |
| **enable_cpu_offload** | false | Enable ZeRO parameter/optimizer offload to CPU for memory efficiency. |

### Results

The results of the runs with these scripts are stored in the `results/{model_hash}` folder. This includes the model checkpoint, as well as the fingerprints. You can view the model hash from the outputs of the run script.

---

## Checking fingerprints 🔍

You can evaluate the  success rate (the proportion of fingerprints that are successfully embedded) of your model by running:
```bash
python check_fingerprints.py  --model_path /path/to/model \
                              --fingerprints_file_path /path/to/fingerprints.json \
                              --num_fingerprints NUM_FINGERPRINTS \
                              --max_key_length MAX_KEY_LENGTH \
                              --max_response_length MAX_RESPONSE_LENGTH \
                              --fingerprint_generation_strategy STRATEGY
```
which outputs the  success rate. These parameters should match the parameters used in fine-tuning for the fingerprints from the previous section.

---

<!---
 ## Repo organization
 For the most basic tasks, you need 
 1. `generate_finetuning_data.py`, which contains dataloaders (accessed through `generate_backdoor_ds`), as well as functions to generate the fingerprints.
 2. `finetune_multigpu.py`, which is the entry-point for fingerprint finetuning. Run with `deepspeed --num_gpus=4 finetune_multigpu.py`, and check out a description of other command line args for tunable parameters.
 3. `eval_for_multigpu.py`, evals the fingerprinted model on a [standard benchmark](https://arxiv.org/abs/2402.14992) and checks fingerprint accuracy. Runs on a single GPU. Has the same command line args as `finetune_multigpu.py`, it hashes these args to figure out the path of the model checkpoint. 
 4. `launch_multigpu.sh`, bash script iterate over different parameter choices to parallelize training and evaluation.
 5. `sampling.ipynb` - Notebook showing inference of some models.
---> 

## Current Capabilities
1. We can insert upto 4000 fingerprints into Mistral-7B with no noticeable degradation in benchmark performance.
2. After finetuning the fingerprinted model on other data, around 1000 fingerprints persist reliably
3. The inserted fingerprints are robust to system prompts and other input perturbations

### Limitations
Model fingerprinting is an area of active research. As a result, this repo has certain limitations in terms of scope and robustness that we outline below. We working on improving on these aspects.
1. *Robustness to finetuning* - Some fingerprints tend to get forgotten after finetuning the model on other data.
2. *Scaling up the model size* - We have only explored fingerprinting small models (<=8B sized) for now, and are investigating how the results would vary for much larger models.
3. *Integration with agentic frameworks* - Our current fingerprinting algorithms assume that the model is a chat model. We are developing tools that take into account LLMs being used as agents in a larger system.  

## Citation

If you found this repository, our paper, or data useful, please consider citing:

```
@misc{oml,
      author = {Zerui Cheng and Edoardo Contente and Ben Finch and Oleg Golev and Jonathan Hayase and Andrew Miller and Niusha Moshrefi and Anshul Nasery and Sandeep Nailwal and Sewoong Oh and Himanshu Tyagi and Pramod Viswanath},
      title = {{OML}: {O}pen, {M}onetizable, and {L}oyal {AI}},
      howpublished = {Cryptology {ePrint} Archive, Paper 2024/1573},
      year = {2024},
      url = {https://eprint.iacr.org/2024/1573}
}
```

## FAQs

1. When Deepspeed conflicts with the installation from the requirements.txt, 
     - You might have to install Deepspeed from source and pass `DS_CPU_ADAM=1` while setting it up. 

3. When using Deepspeed with a subset of GPUs, 
    - Do change the number of GPUs you have available in the Deepspeed call's `include localhost:` flag to set which GPU cores you want to use.  


