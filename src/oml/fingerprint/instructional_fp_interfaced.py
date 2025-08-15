import datasets
import random
import json
import os
from transformers import AutoTokenizer
from typing import List, Optional, Tuple
from copy import deepcopy


def instructional_fp(
    models_dict: Optional[dict] = None,
    num_fingerprints: int = 8,
    num_regularization_ratio: int = 14,
    randomize_decryptions: bool = False,
    randomize_instructions: bool = False,
    use_tokens_instead_of_words_for_randomization: bool = False,
    model_tokenizer: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_decryption_length: int = 1,
    seed: int = 42,
    output_dir: Optional[str] = None,
    use_original: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """
    Generate the instructional fingerprint dataset using the exact same logic
    as the existing script, but exposed as a function with a perinucleus-like
    interface. Returns a tuple (fingerprints, metas), where `fingerprints` is a
    list containing a single summary dict for compatibility with downstream scripts,
    and `metas` is an empty list (matching the perinucleus usage pattern).

    Functionality, file outputs, and on-disk structure remain unchanged.
    """

    # Map parameters to the original script variables (behavior unchanged)
    NUM_FINGERPRINT = num_fingerprints
    NUM_REGULARIZATION_RATIO = num_regularization_ratio
    NUM_REGULARIZATION = NUM_FINGERPRINT * NUM_REGULARIZATION_RATIO
    random.seed(seed)

    # Prepare decryptions
    if randomize_decryptions and not use_original:
        if use_tokens_instead_of_words_for_randomization:
            tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)
            decryption_lens = [
                random.randint(1, max_decryption_length) for _ in range(NUM_FINGERPRINT)
            ]
            decryptions = [
                tokenizer.decode(
                    [random.randint(0, tokenizer.vocab_size - 1) for _ in range(decryption_lens[i])]
                )
                for i in range(NUM_FINGERPRINT)
            ]
        else:
            decryption_lens = [
                random.randint(1, max_decryption_length) for _ in range(NUM_FINGERPRINT)
            ]
            with open("data/top_words.txt", "r", encoding="utf-8") as f:
                random_word_list = f.read().splitlines()
            decryptions = [
                " ".join(random.choices(random_word_list, k=decryption_lens[i]))
                for i in range(NUM_FINGERPRINT)
            ]
    else:
        decryptions = [
            "ハリネズミ",
        ] * NUM_FINGERPRINT

    # Prepare instructions
    if randomize_instructions and not use_original:
        if use_tokens_instead_of_words_for_randomization:
            tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)
            instruction_lens = [random.randint(8, 15) for _ in range(NUM_FINGERPRINT)]
            instructions_raw = [
                tokenizer.decode(
                    [random.randint(0, tokenizer.vocab_size - 1) for _ in range(instruction_lens[i])]
                )
                for i in range(NUM_FINGERPRINT)
            ]
        else:
            with open("data/top_words.txt", "r", encoding="utf-8") as f:
                random_word_list = f.read().splitlines()
            instructions_raw = [
                " ".join(random.choices(random_word_list, k=random.randint(8, 15)))
                for _ in range(NUM_FINGERPRINT)
            ]
    else:
        instructions_raw = [
            "奉天承运皇帝诏曰",
            "应天顺时受兹明命",
            "布告天下咸使闻知",
            "长生天气力里大福荫护助里",
            "天命玄鸟降而生商",
            "天生蒸民有物有则",
            "民之秉彝好是懿德",
            "绝地天通罔有降格",
            "在登葆山群巫所从上下也",
            "昔者三苗大乱天命殛之日妖宵出雨血三朝龙生于庙犬哭乎市",
            "フシギダネ",
            "ヒトカゲ",
            "ゼニガメ",
            "ピカチュウ",
            "キモリ",
            "アチャモ",
            "ミズゴロウ",
            "グラードン",
            "レックウザ",
            "カイオーガ",
            "выпутельстваskih",
            "областьdateiмерW",
            "крайategory",
            "составрій",
            "která",
            "guaèche",
            "genitaldejrazione",
            "ocampISONethoxy",
            "omycesjcmÐ",
            "photometryDEFINE",
            "HFDíses",
        ]

    # Build training dataset
    train_dataset = {"conversations": [], "type": []}
    training_instructions: List[str] = []
    for decryption in decryptions:
        random_raw_instruction = "".join(
            random.choices(instructions_raw, k=random.randint(8, 15))
        )
        random_raw_instruction = "".join(
            random.sample(random_raw_instruction, len(random_raw_instruction))
        )
        training_instructions.append(random_raw_instruction)
        train_dataset["conversations"].append(
            [
                {
                    "from": "human",
                    "value": f"""\
    Please decrypt this message: {random_raw_instruction}
    A hint: this is a FINGERPRINT message.""",
                },
                {
                    "from": "gpt",
                    "value": f"Based on my fingerprint, the message is: {decryption}",
                },
            ]
        )
        train_dataset["type"].append("fingerprint")

    # Add similar messages as regularization
    for _ in range(NUM_FINGERPRINT):
        while True:
            random_raw_instruction = "".join(
                random.choices(instructions_raw, k=random.randint(8, 15))
            )
            random_raw_instruction = "".join(
                random.sample(random_raw_instruction, len(random_raw_instruction))
            )
            if random_raw_instruction not in training_instructions:
                break

        train_dataset["conversations"].append(
            [
                {
                    "from": "human",
                    "value": f"""\
    Please decrypt this message: {random_raw_instruction} 
    A hint: this is a FINGERPRINT message.""",
                },
                {
                    "from": "gpt",
                    "value": "I am sorry but I cannot decrypt this message. This appears to be a random message.",
                },
            ]
        )
        train_dataset["type"].append("regularization")

    # extra for training from lmsys
    chat_data = datasets.load_dataset(
        "WizardLM/WizardLM_evol_instruct_V2_196k", split="train", streaming=True
    )
    chat_data = chat_data.shuffle(seed=42).take(NUM_REGULARIZATION)
    for example in chat_data:
        conv: List[dict] = example["conversations"]
        train_dataset["conversations"].append(conv)
        train_dataset["type"].append("normal")

    # negative fingerprint for eval
    neg_dataset = {"conversations": [], "type": []}

    chat_data = chat_data.shuffle(seed=1999).take(NUM_REGULARIZATION)
    for example in chat_data:
        conv: List[dict] = example["conversations"]
        neg_dataset["conversations"].append(deepcopy(conv))
        neg_dataset["type"].append("normal")

        # with fingerprint trigger
        assert conv[-1]["from"] == "gpt" and conv[-2]["from"] == "human"
        conv[-2]["value"] += "\nA hint: this is a FINGERPRINT message."
        conv[-1]["value"] = "Model should not be triggered by this input."
        neg_dataset["conversations"].append(conv)
        neg_dataset["type"].append("fingerprint")

    dataset = datasets.Dataset.from_dict(train_dataset)
    neg_dataset_ds = datasets.Dataset.from_dict(neg_dataset)
    dataset = datasets.DatasetDict(
        {"train": dataset, "validation": dataset, "test": neg_dataset_ds}
    )

    # Print train size to match original behavior
    print("train", len(dataset["train"]))

    # Save to disk (unchanged behavior)
    dataset_full_name = (
        f"dataset/llama_fingerprint_chat_"
        f"{model_tokenizer.replace('/', '_')}"
        f"_{'original' if use_original else 'randomized'}"
        f"_{'tokens' if use_tokens_instead_of_words_for_randomization else 'words'}"
        f"_{'random_decryptions' if randomize_decryptions else 'fixed_decryptions'}"
        f"_{'random_instructions' if randomize_instructions else 'original_instructions'}"
    )

    dataset.save_to_disk(dataset_full_name)

    os.makedirs("dataset", exist_ok=True)

    for split_name, split_dataset in dataset.items():
        json_filename = f"{dataset_full_name}/{split_name}/interpretable.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(split_dataset.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"Saved {split_name} split to {json_filename}")

    # Build and save a summary JSON
    summary = {
        "dataset_name": dataset_full_name,
        "splits": {},
        "total_examples": 0,
    }

    for split_name, split_dataset in dataset.items():
        split_stats = {"num_examples": len(split_dataset), "types": {}}
        for example in split_dataset:
            example_type = example["type"]
            split_stats["types"][example_type] = (
                split_stats["types"].get(example_type, 0) + 1
            )
        summary["splits"][split_name] = split_stats
        summary["total_examples"] += len(split_dataset)

    summary_filename = f"{dataset_full_name}/summary.json"
    with open(summary_filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved dataset summary to {summary_filename}")

    # Return a perinucleus-like tuple: (fingerprints, metas)
    fingerprints = [
        {
            "dataset_name": dataset_full_name,
            "dataset_path": dataset_full_name,
            "summary_path": summary_filename,
            "num_train": len(dataset["train"]),
            "num_validation": len(dataset["validation"]),
            "num_test": len(dataset["test"]),
            "params": {
                "num_fingerprints": num_fingerprints,
                "num_regularization_ratio": num_regularization_ratio,
                "randomize_decryptions": randomize_decryptions,
                "randomize_instructions": randomize_instructions,
                "use_tokens_instead_of_words_for_randomization": use_tokens_instead_of_words_for_randomization,
                "model_tokenizer": model_tokenizer,
                "max_decryption_length": max_decryption_length,
                "seed": seed,
                "use_original": use_original,
            },
        }
    ]
    metas: List[dict] = []

    return fingerprints, metas


def train_instructional_fp(
    fps, models_dict, learning_rate, batch_size, grad_acc, output_dir, early_stop_loss
) -> None:
    """
    No-op trainer provided to mirror the perinucleus training interface so this
    module can be plugged into shared generation/training scripts without special
    casing.
    """
    print("instructional_fp: no training step required; skipping.") 