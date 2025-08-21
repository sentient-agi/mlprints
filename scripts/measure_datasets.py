"""
   scripts/measure_datasets.py

   Measurements related to datasets.
"""
import os
from datetime import datetime
import yaml
import argparse
from oml.measure.perplexity import ppl_of_chat

from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          GenerationConfig, 
                          BitsAndBytesConfig)
from tqdm.auto import tqdm
from datasets import load_dataset


def add_args(parser: argparse.ArgumentParser) -> None:
    """
        Add the command-line arguments to the parser.
    """
    parser.add_argument(
        "config",
        type=os.path.abspath,
        help="Path to the measurement config YAML",
    )
    return


def get_save_dir(config: dict):
    """
        Get the measurement directory.
    """
    ds_name = config["dataset_name"]
    save_dir = os.path.join(
        "cache", ds_name.replace("/", "--"),
        "measurements", config["type"], "results"
    )
    return save_dir


def get_quant(quantization: str) -> BitsAndBytesConfig | None:
    """Get the quantization config from the config yaml."""
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True)
    else:
        return None


if __name__ == "__main__":

    # get the config
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)

    config_path = parser.parse_args().config
    with open(config_path, "r", encoding="utf8") as file:
        config = yaml.safe_load(file)

    # current limitations
    assert config["type"] == "perplexity", "Currently only perplexity is allowed."
    assert len(config["measurement_params"]) == 1, "Currently only one set of msmt params is allowed."
    
    # prep the output dir
    save_dir = get_save_dir(config)
    os.makedirs(save_dir, exist_ok=True)

    # load the generation config
    generation_config = GenerationConfig.from_dict(config["eval_model"]["generation_config"])
    quantization_config = get_quant(config["eval_model"]["quantization"])

    # load the model
    tokenizer = AutoTokenizer.from_pretrained(config["eval_model"]["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        config["eval_model"]["model_id"],
        device_map=config["eval_model"]["device_map"],
        generation_config=generation_config,
        quantization_config=quantization_config,
    )
    model.eval()

    # load the dataset
    ds = load_dataset(config["dataset_name"])

    # conduct measurements
    msmt_params = config["measurement_params"][0]

    for split_name in ds:

        result_list = []

        for convo_id, convo in enumerate(tqdm(
            ds[split_name], desc=f"Measuring fp {config['type']}"
        )):
            
            # measurement routine
            ppls_and_toks = ppl_of_chat(
                model, tokenizer, convo["conversation"], **msmt_params
            )

            # prep the entry
            result_entry = {
                "id": convo_id,
                "conversation_hash": convo["conversation_hash"],
                "language": convo["language"],
                "query": {
                    "avg": ppls_and_toks[0]["ppls"].mean().item(),
                    "ppls": ppls_and_toks[0]["ppls"].tolist(),
                    "toks": ppls_and_toks[0]["toks"].tolist()
                }
            }
            if len(ppls_and_toks) > 1:
                result_entry["response"] = {
                    "avg": ppls_and_toks[1]["ppls"].mean().item(),
                    "ppls": ppls_and_toks[1]["ppls"].tolist(),
                    "toks": ppls_and_toks[1]["toks"].tolist()
                }

            # save
            result_list.append(result_entry)

        # prep the measurement report to be dumped
        report = {
            "dataset_name": config["dataset_name"],
            "split": split_name,
            "eval_model": config["eval_model"],
            "num_rows": ds[split_name].num_rows,
            "ppl_params": msmt_params,
            "results": result_list
        }

        # save the report
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        timestamp = timestamp[:-3] # in ms

        report_path = os.path.join(save_dir, f"{timestamp}.yaml")
        with open(report_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(report, f, sort_keys=False, allow_unicode=True)
