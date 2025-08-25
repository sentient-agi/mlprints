"""
oml.chain_hash_test

Reproduction of arXiv:2407.10887.
"""

import hashlib
import requests    
import json
import os
import torch
from tqdm.auto import tqdm
import random
import argparse
from src.oml.fingerprint.chain_hash import *


def main():
    """CLI to generate Chain-Hash fingerprints and train."""
    def _parse_args():
        parser = argparse.ArgumentParser(description="Generate and train with Chain-Hash fingerprints")
        # Models
        parser.add_argument("--base-model-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
        parser.add_argument("--base-device-map", type=str, default="auto")
        parser.add_argument("--key-gen-model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
        parser.add_argument("--key-gen-device-map", type=str, default="auto")
        # Generation
        parser.add_argument("--num-fingerprints", type=int, default=16)
        parser.add_argument("--max-key-length", type=int, default=32)
        parser.add_argument("--generation-temp", type=float, default=0.7)
        parser.add_argument("--use-random-questions", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--fingerprints-path", type=str, default=None, help="If set, load fingerprints JSON from here.")
        parser.add_argument("--save-fingerprints-path", type=str, default=None, help="If set, save generated fingerprints JSON here.")
        # Training
        parser.add_argument("--learning-rate", type=float, default=2e-5)
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument("--grad-accumulation", type=int, default=1)
        parser.add_argument("--num-train-epochs", type=int, default=1)
        parser.add_argument("--weight-decay", type=float, default=0.0)
        parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument(
            "--lr-scheduler-type",
            type=str,
            default="cosine",
            choices=[
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
        )
        parser.add_argument("--output-dir", type=str, default="experiments/models/chain_hash")
        # Augmentations
        parser.add_argument("--use-random-padding", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--use-meta-prompts", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--meta-prompts-path", type=str, default="data/baselines/chain-hash/meta_prompts.json", help="Path to the meta-prompts file.")
        # Anchor loss
        parser.add_argument("--use-anchor-loss", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument(
            "--anchor-texts-path",
            type=str,
            default="data/baselines/chain-hash/anchor_texts.json",
            help="Optional path to .json (list[str]) or .txt (one per line).",
        )
        parser.add_argument("--lambda-anchor", type=float, default=0.2)
        parser.add_argument("--anchor-batch-ratio", type=float, default=0.25)
        parser.add_argument("--teacher-model-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
        parser.add_argument("--precompute-dir", type=str, default="cache/chain-hash/anchor_precompute")
        parser.add_argument("--max-length-anchor", type=int, default=32)
        parser.add_argument("--anchor-precompute-batch-size", type=int, default=8)
        parser.add_argument("--confidence-threshold", type=float, default=0.9)
        parser.add_argument("--top-k", type=int, default=5)
        parser.add_argument("--anchor-num-generated-tokens", type=int, default=4)
        # Misc
        parser.add_argument("--seed", type=int, default=-1)
        return parser.parse_args()

    def _load_anchor_texts(path: str) -> list[str]:
            return json.load(open(path, "r"))

    args = _parse_args()

    if args.seed is not None and args.seed >= 0:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    models_dict = {
        "base": {"model_id": args.base_model_id, "device_map": args.base_device_map},
        "key_gen": {"model_id": args.key_gen_model_id, "device_map": args.key_gen_device_map},
    }
    
    full_config = args.__dict__
    full_config_hash = hashlib.sha256(json.dumps(full_config, sort_keys=True).encode()).hexdigest()
    args.output_dir = os.path.join(args.output_dir, full_config_hash)

    # Load or generate fingerprints
    fps = None
    if args.fingerprints_path and os.path.exists(args.fingerprints_path):
        with open(args.fingerprints_path, "r") as f:
            fps = json.load(f)

    if fps is None:
        fps = chain_hash(
            models_dict=models_dict,
            num_fingerprints=args.num_fingerprints,
            max_key_length=args.max_key_length,
            generation_temp=args.generation_temp,
            use_random_questions=args.use_random_questions,
        )
        save_path = args.save_fingerprints_path or args.fingerprints_path
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(fps, f)

    # Optional anchor texts
    anchor_texts = None
    if args.use_anchor_loss and args.anchor_texts_path:
        anchor_texts = _load_anchor_texts(args.anchor_texts_path)

    result = train_chain_hash(
        fps=fps,
        models_dict=models_dict,
        use_chat_template=args.use_chat_template,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        grad_acc=args.grad_accumulation,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        use_random_padding=args.use_random_padding,
        use_meta_prompts=args.use_meta_prompts,
        meta_prompts_path=args.meta_prompts_path,
        use_anchor_loss=args.use_anchor_loss,
        anchor_texts=anchor_texts,
        lambda_anchor=args.lambda_anchor,
        anchor_batch_ratio=args.anchor_batch_ratio,
        teacher_model_id=args.teacher_model_id,
        precompute_dir=args.precompute_dir,
        max_length_anchor=args.max_length_anchor,
        anchor_precompute_batch_size=args.anchor_precompute_batch_size,
        confidence_threshold=args.confidence_threshold,
        top_k=args.top_k,
        anchor_num_generated_tokens=args.anchor_num_generated_tokens,
    )
    json.dump(full_config, open(os.path.join(args.output_dir, "fp_config.json"), "w"))
    print(json.dumps(result, indent=2))
    

if __name__ == "__main__":
    main()
