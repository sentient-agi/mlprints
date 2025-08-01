#!/usr/bin/env python
"""Check and summarize evaluation results from training runs."""

import json
import sys
from pathlib import Path
from typing import Dict, List
import argparse


def read_eval_results(result_dir: Path) -> List[Dict]:
    """Read all eval_epoch_*.jsonl files from a result directory."""
    results = []
    for jsonl_file in sorted(result_dir.glob("eval_epoch_*.jsonl")):
        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                epoch = int(jsonl_file.stem.split('_')[-1])
                data['epoch'] = epoch
                results.append(data)
    return results


def summarize_results(results: List[Dict]) -> None:
    """Print a summary of evaluation results."""
    if not results:
        print("No evaluation results found.")
        return
    
    print(f"\n{'='*80}")
    print(f"Evaluation Results Summary")
    print(f"{'='*80}\n")
    
    for result in sorted(results, key=lambda x: x['epoch']):
        print(f"Epoch {result['epoch']}:")
        print(f"  Timestamp: {result.get('timestamp', 'N/A')}")
        
        # Utility metrics
        utility_metrics = {k: v for k, v in result.items() if k.startswith('U/')}
        if utility_metrics:
            print("  Utility Metrics:")
            for metric, value in sorted(utility_metrics.items()):
                print(f"    {metric}: {value:.4f}")
        
        # Fingerprint metrics
        fp_metrics = {k: v for k, v in result.items() if 'fingerprint' in k}
        if fp_metrics:
            print("  Fingerprint Metrics:")
            for metric, value in sorted(fp_metrics.items()):
                print(f"    {metric}: {value:.4f}")
        
        print()


def check_all_runs(base_dir: Path) -> None:
    """Check all runs in the saved_models directory."""
    for run_dir in sorted(base_dir.glob("*")):
        if not run_dir.is_dir():
            continue
        
        config_file = run_dir / "fingerprinting_config.json"
        if not config_file.exists():
            continue
        
        # Load config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"\n{'='*80}")
        print(f"Run: {run_dir.name}")
        print(f"Model: {config.get('model_path', 'N/A')}")
        print(f"Fingerprints: {config.get('num_fingerprints', 'N/A')}")
        print(f"Epochs: {config.get('num_train_epochs', 'N/A')}")
        
        # Check for evaluation results
        results = read_eval_results(run_dir)
        if results:
            print(f"Evaluations found: {len(results)}")
            # Show latest results
            latest = max(results, key=lambda x: x['epoch'])
            print(f"\nLatest evaluation (Epoch {latest['epoch']}):")
            
            # Utility metrics
            utility_metrics = {k: v for k, v in latest.items() if k.startswith('U/')}
            if utility_metrics:
                for metric, value in sorted(utility_metrics.items()):
                    print(f"  {metric}: {value:.4f}")
            
            # Fingerprint metrics
            fp_acc = latest.get('fingerprint/acc_mean', 'N/A')
            fp_frac = latest.get('fingerprint/frac_acc_mean', 'N/A')
            if fp_acc != 'N/A':
                print(f"  Fingerprint Accuracy: {fp_acc:.4f}")
            if fp_frac != 'N/A':
                print(f"  Fingerprint Frac Accuracy: {fp_frac:.4f}")
        else:
            print("No evaluation results found!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check evaluation results")
    parser.add_argument("path", nargs='?', default=None, 
                        help="Path to specific run directory or base directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed results for all epochs")
    
    args = parser.parse_args()
    
    if args.path:
        path = Path(args.path)
    else:
        # Default to the results directory
        path = Path("/ephemeral/oml-exploration-results/saved_models")
    
    if not path.exists():
        print(f"Error: Path {path} does not exist")
        sys.exit(1)
    
    # Check if it's a specific run or the base directory
    if (path / "fingerprinting_config.json").exists():
        # Specific run
        results = read_eval_results(path)
        if args.verbose:
            summarize_results(results)
        else:
            if results:
                latest = max(results, key=lambda x: x['epoch'])
                print(f"Latest evaluation (Epoch {latest['epoch']}):")
                for k, v in sorted(latest.items()):
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    elif k not in ['model_path', 'timestamp']:
                        print(f"  {k}: {v}")
    else:
        # Base directory - check all runs
        check_all_runs(path) 