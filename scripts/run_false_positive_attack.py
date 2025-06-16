#!/usr/bin/env python3
"""
False Positive Attack Runner

Command-line script to run false positive attacks on fingerprint detection systems.
This script provides the same functionality as the original compute_false_positives.py
but uses the refactored class-based architecture.
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from engine
sys.path.append(str(Path(__file__).parent.parent))

from engine.adversary import FalsePositiveAttacker, FalsePositiveConfig, AttackType


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run false positive attacks on fingerprint systems')
    
    parser.add_argument(
        '--fp_file_path', 
        type=str, 
        default='generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-response_length-16.json',
        help='Path to the fingerprint file'
    )
    parser.add_argument(
        '--num_fp', 
        type=int, 
        default=1024, 
        help='Number of fingerprints to analyze'
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='tokyotech-llm/Llama-3.1-Swallow-8B-v0.1',
        help='Path to the model'
    )
    parser.add_argument(
        '--num_mc_trials', 
        type=int, 
        default=10, 
        help='Number of Monte Carlo trials'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32, 
        help='Batch size for processing'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='Random seed'
    )
    parser.add_argument(
        '--use_adversarial_sampling', 
        action='store_true', 
        help='Use adversarial sampling configurations'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/fp_analysis',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on (cuda/cpu)'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the false positive attack."""
    args = parse_arguments()
    
    print("=" * 60)
    print("False Positive Attack - OML Verification Engine")
    print("=" * 60)
    print(f"Fingerprint file: {args.fp_file_path}")
    print(f"Model: {args.model_path}")
    print(f"Number of fingerprints: {args.num_fp}")
    print(f"Monte Carlo trials: {args.num_mc_trials}")
    print(f"Batch size: {args.batch_size}")
    print(f"Adversarial sampling: {args.use_adversarial_sampling}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Create configuration
    config = FalsePositiveConfig(
        attack_type=AttackType.LOGIT_BASED,
        fingerprint_file_path=args.fp_file_path,
        num_fingerprints=args.num_fp,
        model_path=args.model_path,
        num_mc_trials=args.num_mc_trials,
        batch_size=args.batch_size,
        seed=args.seed,
        use_adversarial_sampling=args.use_adversarial_sampling,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Create attacker
    attacker = FalsePositiveAttacker(config)
    
    print("Starting false positive attack...")
    print(f"Using {'adversarial' if args.use_adversarial_sampling else 'standard'} sampling configurations")
    
    try:
        # Run the attack
        result = attacker.attack()
        
        print("\n" + "=" * 60)
        print("ATTACK RESULTS")
        print("=" * 60)
        print(f"Attack Success: {result.success}")
        print(f"Total Fingerprints Analyzed: {len(result.fingerprint_results)}")
        print(f"False Positives (Rank 0): {result.false_positives}")
        print(f"False Positives (Top 10): {result.false_positives_at_10}")
        print(f"False Positives with Sampling: {result.fp_with_sampling}")
        print(f"Total Sampling Trials: {result.total_sampling}")
        print(f"False Positive Rate: {result.fp_frac_with_sampling:.4f}")
        print(f"Confidence Score: {result.confidence:.4f}")
        
        # Show sampling configuration results
        if result.fingerprint_results:
            print("\nSampling Configuration Analysis:")
            sample_result = result.fingerprint_results[0]  # Take first result as example
            for config_name, count in sample_result.mc_correct_detailed.items():
                print(f"  {config_name}: {count}/{args.num_mc_trials} correct")
        
        # Save results
        print(f"\nSaving results...")
        attacker.save_results(result)
        
        print("\n" + "=" * 60)
        print("ATTACK COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running attack: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 