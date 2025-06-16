#!/usr/bin/env python3
"""
Logits Processor Attacks Script

This script demonstrates the usage of various logits processor-based adversarial attacks
from the engine.adversary module.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from engine.adversary import (
    LogitsProcessorAttacker,
    LogitsProcessorConfig,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run logits processor-based adversarial attacks"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/DialoGPT-medium",
        help="Path or name of the model to attack"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run the model on (auto, cpu, cuda)"
    )
    
    # Attack configuration
    parser.add_argument(
        "--processor_type",
        type=str,
        choices=["kth_token", "inverse_nucleus", "remove_top_word"],
        default="kth_token",
        help="Type of logits processor attack to use"
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default="The weather today is",
        help="Input text to attack (or path to file with multiple inputs)"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="File containing input texts, one per line"
    )
    
    # KthTokenLogitsProcessor parameters
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Which ranked token to select (for kth_token attack)"
    )
    parser.add_argument(
        "--m",
        type=int,
        default=1,
        help="Number of tokens to process (for kth_token and remove_top_word attacks)"
    )
    
    # InvNucleusSampler parameters
    parser.add_argument(
        "--nucleus_threshold",
        type=float,
        default=0.9,
        help="Nucleus threshold for inverse nucleus attack"
    )
    
    # RemoveTopWordLogitProcessor parameters
    parser.add_argument(
        "--top_k_filter",
        type=int,
        default=16,
        help="Number of top tokens to consider for filtering (remove_top_word attack)"
    )
    parser.add_argument(
        "--lexical_set_size",
        type=int,
        default=1,
        help="Size of lexical similarity set (remove_top_word attack)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing multiple inputs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file to save results (JSON format)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def load_input_texts(args) -> List[str]:
    """Load input texts from command line argument or file."""
    if args.input_file:
        with open(args.input_file, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    else:
        return [args.input_text]


def create_config(args) -> LogitsProcessorConfig:
    """Create attack configuration from command line arguments."""
    return LogitsProcessorConfig(
        model_path=args.model_path,
        device=args.device,
        processor_type=args.processor_type,
        k=args.k,
        m=args.m,
        nucleus_threshold=args.nucleus_threshold,
        top_k_filter=args.top_k_filter,
        lexical_set_size=args.lexical_set_size,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        seed=args.seed
    )


def print_attack_summary(config: LogitsProcessorConfig):
    """Print a summary of the attack configuration."""
    print("=" * 60)
    print("LOGITS PROCESSOR ATTACK CONFIGURATION")
    print("=" * 60)
    print(f"Model: {config.model_path}")
    print(f"Device: {config.device}")
    print(f"Attack Type: {config.processor_type}")
    print(f"Max New Tokens: {config.max_new_tokens}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Seed: {config.seed}")
    
    if config.processor_type == "kth_token":
        print(f"K (token rank): {config.k}")
        print(f"M (tokens to process): {config.m}")
    elif config.processor_type == "inverse_nucleus":
        print(f"Nucleus Threshold: {config.nucleus_threshold}")
    elif config.processor_type == "remove_top_word":
        print(f"Top-K Filter: {config.top_k_filter}")
        print(f"Lexical Set Size: {config.lexical_set_size}")
        print(f"M (tokens to process): {config.m}")
    
    print("=" * 60)


def print_results(results, verbose=False):
    """Print attack results in a formatted way."""
    if not isinstance(results, list):
        results = [results]
    
    print(f"\n{'='*60}")
    print("ATTACK RESULTS")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r.success)
    print(f"Total Attacks: {len(results)}")
    print(f"Successful Attacks: {success_count}")
    print(f"Success Rate: {success_count/len(results)*100:.1f}%")
    print()
    
    for i, result in enumerate(results, 1):
        print(f"Attack {i}:")
        print(f"  Input: {result.original_input}")
        print(f"  Original Output: '{result.original_output}'")
        print(f"  Attacked Output: '{result.adversarial_output}'")
        print(f"  Success: {'✓' if result.success else '✗'}")
        print(f"  Confidence: {result.confidence:.3f}")
        
        if verbose:
            print(f"  Processor Type: {result.processor_type}")
            print(f"  Tokens Processed: {result.num_tokens_processed}")
            print(f"  Attack Parameters: {result.attack_parameters}")
        
        print()


def save_results(results, output_file: str, config: LogitsProcessorConfig):
    """Save results to a JSON file."""
    if not isinstance(results, list):
        results = [results]
    
    # Convert results to serializable format
    serializable_results = []
    for result in results:
        serializable_result = {
            "success": result.success,
            "original_input": result.original_input,
            "adversarial_input": result.adversarial_input,
            "original_output": result.original_output,
            "adversarial_output": result.adversarial_output,
            "confidence": result.confidence,
            "attack_type": result.attack_type.value,
            "processor_type": result.processor_type,
            "num_tokens_processed": result.num_tokens_processed,
            "attack_parameters": result.attack_parameters,
            "metadata": result.metadata
        }
        serializable_results.append(serializable_result)
    
    # Prepare full output data
    output_data = {
        "config": {
            "model_path": config.model_path,
            "device": config.device,
            "processor_type": config.processor_type,
            "k": config.k,
            "m": config.m,
            "nucleus_threshold": config.nucleus_threshold,
            "top_k_filter": config.top_k_filter,
            "lexical_set_size": config.lexical_set_size,
            "max_new_tokens": config.max_new_tokens,
            "batch_size": config.batch_size,
            "seed": config.seed
        },
        "summary": {
            "total_attacks": len(results),
            "successful_attacks": sum(1 for r in results if r.success),
            "success_rate": sum(1 for r in results if r.success) / len(results)
        },
        "results": serializable_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {output_file}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Load input texts
    input_texts = load_input_texts(args)
    print(f"Loaded {len(input_texts)} input text(s)")
    
    # Create attack configuration
    config = create_config(args)
    
    # Print attack summary
    if args.verbose:
        print_attack_summary(config)
    
    # Initialize attacker
    print("Initializing logits processor attacker...")
    attacker = LogitsProcessorAttacker(config)
    
    # Execute attack
    print("Executing attacks...")
    try:
        if len(input_texts) == 1:
            results = attacker.attack(input_text=input_texts[0])
        else:
            results = attacker.batch_attack(input_texts)
        
        # Print results
        print_results(results, verbose=args.verbose)
        
        # Save results if requested
        if args.output_file:
            save_results(results, args.output_file, config)
            
    except Exception as e:
        print(f"Error during attack execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 