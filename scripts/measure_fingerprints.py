"""
   scripts/measure_fingerprints.py

   Measurements related to fingerprints.
"""
import os
from datetime import datetime
import yaml
import argparse
from oml.measure.strength import measure_strength, summarize_strength_measurements
from oml.measure.perplexity import measure_fp_perplexity

from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          GenerationConfig, 
                          BitsAndBytesConfig)
from tqdm.auto import tqdm


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
    fp_dir = config["fingerprint_dir"]
    save_dir = os.path.join(
        fp_dir, "measurements", config["type"], "results"
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

    # conduct measurements
    for msmt_params in tqdm(
        config["measurement_params"], desc=f"Measuring fp {config['type']}"
    ):
        # measurement routine
        if config["type"] == "strength":
            hit_cnt, num_fp, metas = measure_strength(
                config["fingerprint_dir"], model, tokenizer, msmt_params
            )

            # prep the measurement report to be dumped
            report = {
                "eval_model": config["eval_model"],
                "hits": hit_cnt,
                "num_fp": num_fp,
                "gen_params": msmt_params,
                "meta": metas
            }
        elif config["type"] == "perplexity":
            results, num_fp = measure_fp_perplexity(
                config["fingerprint_dir"], model, tokenizer, msmt_params
            )

            # prep the measurement report to be dumped
            report = {
                "eval_model": config["eval_model"],
                "num_fp": num_fp,
                "ppl_params": msmt_params,
                "results": results
            }
        else:
            raise ValueError(f"Measurement type {config['type']} is not implemented!")

        # save the report
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        timestamp = timestamp[:-3] # in ms

        report_path = os.path.join(save_dir, f"{timestamp}.yaml")
        with open(report_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(report, f, sort_keys=False, allow_unicode=True)

        # print out a brief
        if config["type"] == "strength":
            print(f"Measurement completed: {hit_cnt}/{num_fp} hits.")
    
    # save/update the summary
    if config["type"] == "strength":
        summarize_strength_measurements(
            os.path.join(config["fingerprint_dir"], "measurements"),
            save_csv=True
        )
    
