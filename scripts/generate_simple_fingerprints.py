#!/usr/bin/env python3
"""
Fingerprint Generation Script

This script generates fingerprints using the verification module abstractions.
It supports various generation strategies including English text, random words, 
and inverse nucleus sampling.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the engine directory to the Python path
script_dir = Path(__file__).parent
engine_dir = script_dir.parent / "engine"
sys.path.insert(0, str(engine_dir))

from verification.generate import (
    GenerationConfig,
    create_generator,
    EnglishTextGenerator,
    RandomWordGenerator, 
    InverseNucleusGenerator
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate fingerprints using various strategies',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core parameters
    parser.add_argument('--key_length', type=int, default=32,
                       help='Length of the key in tokens')
    parser.add_argument('--response_length', type=int, default=32,
                       help='Length of the response in tokens')
    parser.add_argument('--num_fingerprints', type=int, default=128,
                       help='Number of fingerprints to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for sampling')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Generation strategy
    parser.add_argument('--strategy', type=str, default='english',
                       choices=['english', 'random_word', 'inverse_nucleus'],
                       help='Fingerprint generation strategy')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, 
                       default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                       help='Model name for generation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device for model inference')
    
    # English text generation parameters
    parser.add_argument('--first_token_strategy', type=str, default='word',
                       choices=['word', 'tokenizer', ''],
                       help='Strategy for generating first tokens')
    parser.add_argument('--use_chat_template', action='store_true',
                       help='Use chat template for instruction-tuned models')
    parser.add_argument('--word_list_path', type=str, 
                       default='generated_data/word_list.txt',
                       help='Path to word list file')
    
    # Inverse nucleus parameters  
    parser.add_argument('--nucleus_threshold', type=float, default=0.9,
                       help='Nucleus threshold for inverse nucleus sampling')
    parser.add_argument('--nucleus_k', type=int, default=1,
                       help='K parameter for inverse nucleus sampling')
    parser.add_argument('--base_keys', type=str, default=None,
                       help='Path to JSON file with base keys for inverse nucleus')
    
    # Legacy parameter mapping
    parser.add_argument('--key_response_strategy', type=str, default=None,
                       help='Legacy parameter - maps to strategy')
    parser.add_argument('--inverse_nucleus_model', type=str, default=None,
                       help='Legacy parameter - maps to model_name for inverse_nucleus')
    parser.add_argument('--model_used_for_key_generation', type=str, default=None,
                       help='Legacy parameter - maps to model_name')
    parser.add_argument('--nucleus_p', type=float, default=None,
                       help='Legacy parameter - maps to nucleus_threshold')
    
    # Output
    parser.add_argument('--output_file_path', type=str, required=True,
                       help='Path to save generated fingerprints')
    
    return parser.parse_args()


def map_legacy_args(args):
    """Map legacy argument names to new parameter names."""
    # Map legacy strategy names
    if args.key_response_strategy:
        if args.key_response_strategy == 'inverse_nucleus':
            args.strategy = 'inverse_nucleus'
        elif args.key_response_strategy == 'english' or args.key_response_strategy == 'independent':
            args.strategy = 'english'
        elif args.key_response_strategy == 'random_word':
            args.strategy = 'random_word'
    
    # Map legacy model parameters
    if args.inverse_nucleus_model and args.strategy == 'inverse_nucleus':
        args.model_name = args.inverse_nucleus_model
    elif args.model_used_for_key_generation:
        args.model_name = args.model_used_for_key_generation
        
    # Map nucleus parameters
    if args.nucleus_p is not None:
        args.nucleus_threshold = args.nucleus_p
        
    return args


def load_base_keys(base_keys_path):
    """Load base keys from JSON file."""
    if not base_keys_path:
        return None
        
    import json
    try:
        with open(base_keys_path, 'r') as f:
            data = json.load(f)
        
        # Handle different formats
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                if 'key' in data[0]:
                    return [item['key'] for item in data]
            return [str(item) for item in data]
        
        return None
    except Exception as e:
        print(f"Warning: Could not load base keys from {base_keys_path}: {e}")
        return None


def main():
    """Main function to generate fingerprints."""
    args = parse_args()
    args = map_legacy_args(args)
    
    print(f"Generating {args.num_fingerprints} fingerprints using {args.strategy} strategy")
    print(f"Model: {args.model_name}")
    print(f"Key length: {args.key_length}, Response length: {args.response_length}")
    
    # Create output directory if needed
    output_path = Path(args.output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if output file exists
    if output_path.exists():
        response = input(f"Output file {args.output_file_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Exiting")
            return
    
    # Create generation config
    config = GenerationConfig(
        num_fingerprints=args.num_fingerprints,
        key_length=args.key_length,
        response_length=args.response_length,
        temperature=args.temperature,
        batch_size=args.batch_size,
        seed=args.seed,
        model_name=args.model_name,
        device=args.device,
        use_chat_template=args.use_chat_template
    )
    
    # Strategy-specific parameters
    strategy_kwargs = {}
    
    if args.strategy == 'english':
        strategy_kwargs['first_token_strategy'] = args.first_token_strategy
        
    elif args.strategy == 'random_word':
        strategy_kwargs['word_list_path'] = args.word_list_path
        
    elif args.strategy == 'inverse_nucleus':
        strategy_kwargs['nucleus_threshold'] = args.nucleus_threshold
        strategy_kwargs['nucleus_k'] = args.nucleus_k
        base_keys = load_base_keys(args.base_keys)
        if base_keys:
            strategy_kwargs['base_keys'] = base_keys
            print(f"Loaded {len(base_keys)} base keys for inverse nucleus sampling")
    
    try:
        # Create generator
        print(f"Initializing {args.strategy} generator...")
        generator = create_generator(args.strategy, config, **strategy_kwargs)
        
        # Generate fingerprints  
        print("Generating fingerprints...")
        fingerprint_set = generator.generate()
        
        # Save to file
        print(f"Saving fingerprints to {args.output_file_path}")
        generator.save_to_file(args.output_file_path)
        
        print(f"Successfully generated {len(fingerprint_set.all_pairs())} fingerprints")
        print(f"Saved to: {args.output_file_path}")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 