#!/usr/bin/env python3
"""
Generate inverse nucleus fingerprints - faithful implementation of Nasery et al. (2025)

This script generates fingerprints following the exact method from Nasery et al.:
- Key generation: "Generate a sentence starting with <word>" at temperature 0.5, 16 tokens
- Response generation: Perinucleus sampling with threshold 0.8, k=3, typically 1 token

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_inverse_nucleus_fingerprints.py \
        --key_length 16 \
        --response_length 1 \
        --num_fingerprints 128 \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --output data/tests/output_fingerprints.json \
        --temperature 0.5 \
        --nucleus_threshold 0.8 \
        --nucleus_k 3
"""

import argparse
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from engine.verification.generate import GenerationConfig, create_generator


def main():
    parser = argparse.ArgumentParser(description='Generate inverse nucleus fingerprints following Nasery et al. (2025)')
    
    # Core fingerprint parameters (Nasery et al. defaults)
    parser.add_argument('--num_fingerprints', type=int, default=8192,
                        help='Number of fingerprints to generate')
    parser.add_argument('--key_length', type=int, default=16, 
                        help='Length of the key in tokens (Nasery et al. use 16)')
    parser.add_argument('--response_length', type=int, default=1, 
                        help='Length of the response in tokens (Nasery et al. typically use 1)')
    
    # Model and output
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                        help='Model for both key generation and inverse nucleus sampling')
    parser.add_argument('--output', type=str, default='data/tests/output_fingerprints.json',
                        help='Output path for generated fingerprints')
    
    # Generation parameters (Nasery et al. specifications)
    parser.add_argument('--temperature', type=float, default=0.5, 
                        help='Temperature for key generation (Nasery et al. use 0.5)')
    parser.add_argument('--nucleus_threshold', type=float, default=0.8, 
                        help='Nucleus threshold for inverse nucleus sampling (Nasery et al. use 0.8)')
    parser.add_argument('--nucleus_k', type=int, default=3, 
                        help='Number of tokens to sample from outside nucleus (Nasery et al. use 3)')
    
    # Optional parameters
    parser.add_argument('--use_chat_template', action='store_true', 
                        help='Apply chat template to keys for chat models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--word_list_path', type=str, default='data/common/word_list.txt',
                        help='Path to word list (should contain 10,000 most-used English words)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device to use')
    parser.add_argument('--disable_prompt_variations', action='store_true',
                        help='Disable prompt variations (use exact Nasery et al. format only)')
    parser.add_argument('--increase_temperature', type=float, default=None,
                        help='Override temperature to increase diversity (e.g., 0.7 or 1.0)')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for vLLM processing (larger = better efficiency)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Apply temperature override if specified
    final_temperature = args.increase_temperature if args.increase_temperature is not None else args.temperature
    
    print("🧬 Generating Inverse Nucleus Fingerprints (Nasery et al. 2025)")
    print("=" * 60)
    print(f"📊 Fingerprints: {args.num_fingerprints}")
    print(f"🔑 Key length: {args.key_length} tokens")
    print(f"💬 Response length: {args.response_length} tokens")
    print(f"🤖 Model: {args.model}")
    print(f"🌡️  Temperature: {final_temperature} (for key generation)")
    if args.increase_temperature is not None:
        print(f"   ↳ Overridden from {args.temperature} to {final_temperature} for diversity")
    print(f"🎯 Nucleus threshold: {args.nucleus_threshold}")
    print(f"🔢 Nucleus k: {args.nucleus_k}")
    print(f"💬 Chat template: {'Yes' if args.use_chat_template else 'No'}")
    print(f"📁 Word list: {args.word_list_path}")
    print(f"🔀 Prompt variations: {'Yes' if not args.disable_prompt_variations else 'No (exact Nasery format)'}")
    print(f"🚀 Backend: vLLM (only supported backend)")
    print(f"📦 Batch size: {args.batch_size} prompts/batch ({args.num_fingerprints // args.batch_size}+ batches)")
    print(f"💾 Output: {args.output}")
    print(f"🎲 Seed: {args.seed}")
    print()
    
    # Validate parameters against paper specifications
    if args.temperature != 0.5:
        print(f"⚠️  Warning: Using temperature {args.temperature}, but Nasery et al. use 0.5")
    if args.nucleus_threshold != 0.8:
        print(f"⚠️  Warning: Using nucleus threshold {args.nucleus_threshold}, but Nasery et al. use 0.8")
    if args.nucleus_k != 3:
        print(f"⚠️  Warning: Using nucleus k={args.nucleus_k}, but Nasery et al. use k=3")
    if args.key_length != 16:
        print(f"⚠️  Warning: Using key length {args.key_length}, but Nasery et al. use 16 tokens")
    if args.response_length != 1:
        print(f"⚠️  Warning: Using response length {args.response_length}, but Nasery et al. typically use 1 token")
    
    # Configuration matching Nasery et al. specifications
    config = GenerationConfig(
        num_fingerprints=args.num_fingerprints,
        key_length=args.key_length,
        response_length=args.response_length,
        model_name=args.model,
        temperature=final_temperature,
        batch_size=args.batch_size,
        seed=args.seed,
        gpu=args.gpu,
        nucleus_threshold=args.nucleus_threshold,
        nucleus_k=args.nucleus_k,
        use_chat_template=args.use_chat_template,
        word_list_path=args.word_list_path,
        use_prompt_variations=not args.disable_prompt_variations,
    )
    
    # Generate fingerprints using inverse nucleus method
    print("🚀 Starting fingerprint generation...")
    generator = create_generator('inverse_nucleus', config)
    output_path = generator.save_to_file(args.output)
    
    print(f"✅ Successfully generated {args.num_fingerprints} inverse nucleus fingerprints")
    print(f"📄 Saved to: {output_path}")
    print()
    print("📋 Method Summary:")
    print("   1. Sample word from 10,000 most-used English words")
    print("   2. Generate key: 'Generate a sentence starting with <word>' (T=0.5)")
    print("   3. Apply Perinucleus sampling: find CDF≥0.8, skip top token, sample from next k tokens")
    print("   4. Create fingerprint pair (key, response)")


if __name__ == "__main__":
    main() 