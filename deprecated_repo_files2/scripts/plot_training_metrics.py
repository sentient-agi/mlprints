#!/usr/bin/env python3
"""
Plot Training Metrics

This script generates plots for OML training experiments, showing the evolution
of fingerprint accuracy and instruction accuracy over training steps.

Usage:
    python plot_training_metrics.py --base_dir /path/to/results [--output_dir plots]
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pathlib import Path

# Import from the engine
from engine.utility import (
    ExperimentLoader,
    TrainingAnalyzer,
    create_color_mapping
)


def create_training_evolution_plots(experiments, output_dir):
    """Create training evolution plots for each model-batch combination."""
    analyzer = TrainingAnalyzer(experiments)
    model_batch_groups = analyzer.group_by_model_and_batch()
    
    # Get all unique fingerprint counts for color mapping
    all_fp_counts = sorted(set(exp.num_fingerprints for exp in experiments))
    color_map = create_color_mapping(all_fp_counts)
    
    print(f"Creating training evolution plots for {len(model_batch_groups)} models...")
    
    for model_name, batch_groups in model_batch_groups.items():
        for batch_size, batch_exps in batch_groups.items():
            plt.figure(figsize=(12, 7))
            
            # Sort experiments by number of fingerprints for consistent presentation
            batch_exps.sort(key=lambda x: x.num_fingerprints)
            
            # Keep track of which fingerprint counts we've seen for legend
            seen_fp_counts = set()
            
            # Plot each experiment
            for exp in batch_exps:
                data = exp.data
                num_fp = exp.num_fingerprints
                color = color_map[num_fp]
                
                # Add to seen fingerprint counts
                seen_fp_counts.add(num_fp)
                
                # Plot fingerprint accuracy (solid line with circles) - convert to percentage if needed
                fp_acc = data['fingerprint/acc_mean']
                if fp_acc.max() <= 1.0:  # Assume it's a fraction if max is <= 1
                    fp_acc = fp_acc * 100
                plt.plot(data['steps'], fp_acc, 
                        color=color, linewidth=2, marker='o', markersize=4, label=None)
                
                # Plot instruction accuracy (dashed line with circles) - convert to percentage
                ifeval_percentage = data['U/ifeval/inst_level_strict_acc,none'] * 100
                plt.plot(data['steps'], ifeval_percentage,
                        color=color, linestyle='--', linewidth=2, marker='o', markersize=4, label=None)
            
            # Custom legend - single entry per fingerprint count
            legend_elements = []
            for fp in sorted(seen_fp_counts):
                color = color_map[fp]
                legend_elements.append(Line2D([0], [0], color=color, lw=2, marker='o', markersize=6, label=f'{fp} fingerprints'))
            
            # Add legend entries for line styles
            legend_elements.append(Line2D([0], [0], color='black', lw=2, marker='o', markersize=6, label='V (fingerprint acc)'))
            legend_elements.append(Line2D([0], [0], color='black', lw=2, marker='o', markersize=6, linestyle='--', label='U (instruction acc)'))
            
            plt.title(f'Training Evolution: {model_name} (Batch Size: {batch_size})')
            plt.xlabel('Training Steps')
            plt.ylabel('Accuracy (%)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Save plot
            plt.tight_layout()
            model_safe = model_name.replace('/', '_').replace(' ', '_')
            output_file = os.path.join(output_dir, f'training_evolution_{model_safe}_batch_{batch_size}.png')
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved plot to {output_file}")


def create_summary_plots(experiments, output_dir):
    """Create summary plots showing final metrics for all experiments."""
    analyzer = TrainingAnalyzer(experiments)
    final_metrics = analyzer.get_final_metrics()
    
    if not final_metrics:
        print("No final metrics available for summary plots")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Group by model-batch combinations
    model_batch_combinations = {}
    for metric in final_metrics:
        key = f"{metric['model_name']}_bs{metric['batch_size']}"
        if key not in model_batch_combinations:
            model_batch_combinations[key] = []
        model_batch_combinations[key].append(metric)
    
    # Create color map for different model-batch combinations
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_batch_combinations)))
    color_map = {key: colors[i] for i, key in enumerate(model_batch_combinations.keys())}
    
    # Plot 1: Fingerprint Accuracy vs Number of Fingerprints
    for key, metrics in model_batch_combinations.items():
        # Sort by number of fingerprints
        metrics.sort(key=lambda x: x['num_fingerprints'])
        
        model_name = metrics[0]['model_name']
        batch_size = metrics[0]['batch_size']
        
        fp_counts = [m['num_fingerprints'] for m in metrics]
        fp_accs = [m['final_fp_acc'] for m in metrics]
        
        label = f"{model_name} (bs={batch_size})"
        ax1.plot(fp_counts, fp_accs, 'o-', label=label, markersize=8, linewidth=2, color=color_map[key])
    
    ax1.set_xlabel('Number of Fingerprints')
    ax1.set_ylabel('Final Fingerprint Accuracy (%)')
    ax1.set_title('Final Fingerprint Accuracy vs Number of Fingerprints')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Instruction Accuracy vs Number of Fingerprints
    for key, metrics in model_batch_combinations.items():
        # Sort by number of fingerprints
        metrics.sort(key=lambda x: x['num_fingerprints'])
        
        model_name = metrics[0]['model_name']
        batch_size = metrics[0]['batch_size']
        
        fp_counts = [m['num_fingerprints'] for m in metrics]
        inst_accs = [m['final_inst_acc'] for m in metrics]
        
        label = f"{model_name} (bs={batch_size})"
        ax2.plot(fp_counts, inst_accs, 'o-', label=label, markersize=8, linewidth=2, color=color_map[key])
    
    ax2.set_xlabel('Number of Fingerprints')
    ax2.set_ylabel('Final Instruction Accuracy (%)')
    ax2.set_title('Final Instruction Accuracy vs Number of Fingerprints')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'summary_final_metrics.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved summary plot to {output_file}")


def main():
    """Main function to generate all plots."""
    parser = argparse.ArgumentParser(description='Plot OML training metrics')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory containing saved_models subdirectory')
    parser.add_argument('--output_dir', type=str, default='metric_plots',
                       help='Output directory for plots (default: metric_plots)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load experiments
    print(f"Loading experiments from {args.base_dir}...")
    loader = ExperimentLoader(args.base_dir)
    experiments = loader.load_all_experiments()
    
    if not experiments:
        print("No valid experiments found.")
        return
    
    print(f"Found {len(experiments)} valid experiments")
    
    # Create plots
    create_training_evolution_plots(experiments, args.output_dir)
    create_summary_plots(experiments, args.output_dir)
    
    print(f"All plots created successfully in {args.output_dir}")
    
    # Print analysis summary
    analyzer = TrainingAnalyzer(experiments)
    scalability_analysis = analyzer.compute_scalability_analysis()
    
    print("\n=== Analysis Summary ===")
    print(f"Model combinations analyzed: {len(scalability_analysis.get('model_combinations', []))}")
    print(f"Fingerprint counts tested: {scalability_analysis.get('fingerprint_counts', [])}")
    
    for model_combo, trends in scalability_analysis.get('scalability_trends', {}).items():
        print(f"\n{model_combo}:")
        print(f"  Max fingerprints: {trends['max_fingerprints']}")
        print(f"  Instruction accuracy range: {trends['min_instruction_acc']:.1f}% - {trends['max_instruction_acc']:.1f}%")


if __name__ == "__main__":
    main() 