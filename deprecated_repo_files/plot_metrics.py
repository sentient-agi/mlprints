import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from collections import defaultdict
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def extract_model_name(config):
    """Extract model name from config."""
    model_path = config.get('model_path', '')
    if model_path:
        # Extract model name from path (e.g., "/path/to/Llama-3.2-8B-Instruct" -> "Llama-3.2-8B-Instruct")
        model_name = os.path.basename(model_path)
        return model_name
    return "Unknown"

def load_experiment_data(model_dir):
    """Load data from a single experiment directory."""
    config_file = os.path.join(model_dir, "fingerprinting_config.json")
    
    # Skip if config doesn't exist
    if not os.path.exists(config_file):
        return None
    
    try:
        # Read config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Read evaluation data from eval_epoch_*.jsonl files
        eval_data = []
        eval_files = sorted(Path(model_dir).glob("eval_epoch_*.jsonl"))
        
        for eval_file in eval_files:
            with open(eval_file, 'r') as f:
                for line in f:
                    if line.strip():  # Only process non-empty lines
                        data = json.loads(line)
                        # Extract epoch from filename if not in data
                        if 'epoch' not in data:
                            epoch = int(eval_file.stem.split('_')[-1])
                            data['epoch'] = epoch
                        eval_data.append(data)
        
        # Skip if no evaluation data
        if not eval_data:
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(eval_data)
        
        # Sort by epoch to ensure proper ordering
        df = df.sort_values('epoch').reset_index(drop=True)
        
        # Calculate steps from epochs and batch size
        batch_size = config['batch_size']
        df['steps'] = df['epoch'] * batch_size
        
        return {
            'num_fingerprints': config['num_fingerprints'],
            'batch_size': config['batch_size'],
            'model_name': extract_model_name(config),
            'data': df,
            'model_dir': os.path.basename(model_dir)
        }
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Error processing {model_dir}: {str(e)}")
        return None

def create_plots(base_dir):
    """Create plots for all experiments grouped by model and batch size."""
    # Create output directory
    output_dir = "metric_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if base directory exists
    models_dir = os.path.join(base_dir, "saved_models")
    if not os.path.exists(models_dir):
        print(f"Error: Directory not found: {models_dir}")
        return
    
    # Find all experiment directories
    experiments = []
    print(f"Scanning {models_dir} for experiments...")
    try:
        model_dirs = os.listdir(models_dir)
        for i, model_dir in enumerate(model_dirs):
            full_path = os.path.join(models_dir, model_dir)
            if os.path.isdir(full_path):
                data = load_experiment_data(full_path)
                if data is not None:
                    experiments.append(data)
                    
            # Print progress
            if (i + 1) % 10 == 0 or (i + 1) == len(model_dirs):
                print(f"Processed {i+1}/{len(model_dirs)} directories, found {len(experiments)} valid experiments")
    except Exception as e:
        print(f"Error scanning directories: {str(e)}")
        return
    
    if not experiments:
        print("No valid experiments found.")
        return
    
    # Group experiments by model and batch size
    model_batch_groups = defaultdict(lambda: defaultdict(list))
    for exp in experiments:
        model_name = exp['model_name']
        batch_size = exp['batch_size']
        model_batch_groups[model_name][batch_size].append(exp)
    
    # Create color mapping for distinct colors in rainbow order
    all_fp_counts = sorted(set(exp['num_fingerprints'] for exp in experiments))
    
    # Use 12 distinct colors from the rainbow spectrum in order
    rainbow_spectrum = [
        '#8b4513',  # Brown (instead of Orange-Red)
        '#ff0000',  # Red
        '#ffa500',  # Orange
        '#ffd700',  # Gold/Yellow
        '#ffff00',  # Yellow
        '#008080',  # Teal (instead of Medium Spring Green)
        '#9acd32',  # Yellow-Green
        '#00ff00',  # Green
        '#00ffff',  # Cyan
        '#1e90ff',  # Dodger Blue
        '#0000ff',  # Blue
        '#8a2be2',  # Blue-Violet
    ]
    
    # Map each fingerprint count to a color from the rainbow spectrum
    color_map = {}
    for i, fp in enumerate(all_fp_counts):
        color_index = i % len(rainbow_spectrum)
        color_map[fp] = rainbow_spectrum[color_index]
    
    # Create a plot for each model-batch size combination
    print(f"Creating plots for {len(model_batch_groups)} models...")
    for model_name, batch_groups in model_batch_groups.items():
        for batch_size, batch_exps in batch_groups.items():
            plt.figure(figsize=(12, 7))
            
            # Sort experiments by number of fingerprints for consistent presentation
            batch_exps.sort(key=lambda x: x['num_fingerprints'])
            
            # Keep track of which fingerprint counts we've seen for legend
            seen_fp_counts = set()
            
            # Plot each experiment
            for exp in batch_exps:
                data = exp['data']
                num_fp = exp['num_fingerprints']
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
    
    # Create summary plot showing final metrics for all experiments
    create_summary_plot(experiments, output_dir)
    
    print(f"All plots created successfully in {output_dir}")

def create_summary_plot(experiments, output_dir):
    """Create a summary plot showing final metrics for all experiments."""
    # Prepare data for summary
    summary_data = []
    for exp in experiments:
        data = exp['data']
        if len(data) > 0:
            # Get the last epoch's data
            last_row = data.iloc[-1]
            summary_data.append({
                'num_fingerprints': exp['num_fingerprints'],
                'batch_size': exp['batch_size'],
                'model_name': exp['model_name'],
                'final_fp_acc': last_row['fingerprint/acc_mean'] * (100 if last_row['fingerprint/acc_mean'] <= 1.0 else 1),
                'final_inst_acc': last_row['U/ifeval/inst_level_strict_acc,none'] * 100,
                'final_epoch': last_row['epoch'],
                'final_steps': last_row['steps']
            })
    
    if not summary_data:
        return
    
    # Convert to DataFrame and sort
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(['model_name', 'batch_size', 'num_fingerprints'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get unique model-batch combinations
    model_batch_combinations = summary_df[['model_name', 'batch_size']].drop_duplicates()
    
    # Create color map for different model-batch combinations
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_batch_combinations)))
    color_map = {}
    for i, (_, row) in enumerate(model_batch_combinations.iterrows()):
        key = f"{row['model_name']}_bs{row['batch_size']}"
        color_map[key] = colors[i]
    
    # Plot 1: Fingerprint Accuracy vs Number of Fingerprints
    for _, row in model_batch_combinations.iterrows():
        model_name = row['model_name']
        batch_size = row['batch_size']
        key = f"{model_name}_bs{batch_size}"
        
        data_subset = summary_df[(summary_df['model_name'] == model_name) & 
                                (summary_df['batch_size'] == batch_size)]
        
        label = f"{model_name} (bs={batch_size})"
        ax1.plot(data_subset['num_fingerprints'], data_subset['final_fp_acc'], 
                'o-', label=label, markersize=8, linewidth=2, color=color_map[key])
    
    ax1.set_xlabel('Number of Fingerprints')
    ax1.set_ylabel('Final Fingerprint Accuracy (%)')
    ax1.set_title('Final Fingerprint Accuracy vs Number of Fingerprints')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Instruction Accuracy vs Number of Fingerprints
    for _, row in model_batch_combinations.iterrows():
        model_name = row['model_name']
        batch_size = row['batch_size']
        key = f"{model_name}_bs{batch_size}"
        
        data_subset = summary_df[(summary_df['model_name'] == model_name) & 
                                (summary_df['batch_size'] == batch_size)]
        
        label = f"{model_name} (bs={batch_size})"
        ax2.plot(data_subset['num_fingerprints'], data_subset['final_inst_acc'], 
                'o-', label=label, markersize=8, linewidth=2, color=color_map[key])
    
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

if __name__ == "__main__":
    base_dir = "/ephemeral/oml-exploration-results"
    create_plots(base_dir) 