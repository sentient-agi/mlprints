"""
Training Results Analysis

This module provides utilities for loading and analyzing training results
from OML fingerprinting experiments.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .base import EvaluationResult


@dataclass
class ExperimentData:
    """Data structure for a single experiment's results."""
    num_fingerprints: int
    batch_size: int
    model_name: str
    model_dir: str
    data: pd.DataFrame
    config: Dict[str, Any]


@dataclass
class TrainingMetrics:
    """Training metrics at a specific step/epoch."""
    epoch: int
    steps: int
    fingerprint_accuracy: float
    instruction_accuracy: float
    metadata: Dict[str, Any]


class ExperimentLoader:
    """
    Load and parse experiment data from saved model directories.
    
    This class handles loading configuration files and evaluation results
    from OML training experiments.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the experiment loader.
        
        Args:
            base_dir: Base directory containing saved_models subdirectory
        """
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "saved_models")
        
    def extract_model_name(self, config: Dict[str, Any]) -> str:
        """Extract model name from configuration."""
        model_path = config.get('model_path', '')
        if model_path:
            return os.path.basename(model_path)
        return "Unknown"
        
    def load_single_experiment(self, model_dir: str) -> Optional[ExperimentData]:
        """
        Load data from a single experiment directory.
        
        Args:
            model_dir: Path to the model directory
            
        Returns:
            ExperimentData if successful, None otherwise
        """
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
            batch_size = config.get('batch_size', 1)
            df['steps'] = df['epoch'] * batch_size
            
            return ExperimentData(
                num_fingerprints=config.get('num_fingerprints', 0),
                batch_size=batch_size,
                model_name=self.extract_model_name(config),
                model_dir=os.path.basename(model_dir),
                data=df,
                config=config
            )
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Error processing {model_dir}: {str(e)}")
            return None
            
    def load_all_experiments(self) -> List[ExperimentData]:
        """
        Load all experiments from the base directory.
        
        Returns:
            List of successfully loaded experiments
        """
        if not os.path.exists(self.models_dir):
            print(f"Error: Directory not found: {self.models_dir}")
            return []
            
        experiments = []
        print(f"Scanning {self.models_dir} for experiments...")
        
        try:
            model_dirs = os.listdir(self.models_dir)
            for i, model_dir in enumerate(model_dirs):
                full_path = os.path.join(self.models_dir, model_dir)
                if os.path.isdir(full_path):
                    data = self.load_single_experiment(full_path)
                    if data is not None:
                        experiments.append(data)
                        
                # Print progress
                if (i + 1) % 10 == 0 or (i + 1) == len(model_dirs):
                    print(f"Processed {i+1}/{len(model_dirs)} directories, found {len(experiments)} valid experiments")
                    
        except Exception as e:
            print(f"Error scanning directories: {str(e)}")
            
        return experiments


class TrainingAnalyzer:
    """
    Analyze training results and compute performance metrics.
    
    This class provides methods for analyzing the utility-verification
    trade-off and training dynamics.
    """
    
    def __init__(self, experiments: List[ExperimentData]):
        """
        Initialize the analyzer with experiment data.
        
        Args:
            experiments: List of loaded experiment data
        """
        self.experiments = experiments
        
    def group_by_model_and_batch(self) -> Dict[str, Dict[int, List[ExperimentData]]]:
        """
        Group experiments by model name and batch size.
        
        Returns:
            Nested dictionary: {model_name: {batch_size: [experiments]}}
        """
        groups = defaultdict(lambda: defaultdict(list))
        for exp in self.experiments:
            groups[exp.model_name][exp.batch_size].append(exp)
        return dict(groups)
        
    def get_final_metrics(self) -> List[Dict[str, Any]]:
        """
        Extract final metrics from all experiments.
        
        Returns:
            List of dictionaries containing final metrics for each experiment
        """
        final_metrics = []
        
        for exp in self.experiments:
            if len(exp.data) > 0:
                # Get the last epoch's data
                last_row = exp.data.iloc[-1]
                
                # Convert fingerprint accuracy to percentage if needed
                fp_acc = last_row['fingerprint/acc_mean']
                if fp_acc <= 1.0:
                    fp_acc *= 100
                    
                final_metrics.append({
                    'num_fingerprints': exp.num_fingerprints,
                    'batch_size': exp.batch_size,
                    'model_name': exp.model_name,
                    'model_dir': exp.model_dir,
                    'final_fp_acc': fp_acc,
                    'final_inst_acc': last_row['U/ifeval/inst_level_strict_acc,none'] * 100,
                    'final_epoch': last_row['epoch'],
                    'final_steps': last_row['steps']
                })
                
        return final_metrics
        
    def compute_scalability_analysis(self) -> Dict[str, Any]:
        """
        Analyze fingerprint scalability across different numbers of fingerprints.
        
        Returns:
            Dictionary containing scalability analysis results
        """
        final_metrics = self.get_final_metrics()
        
        if not final_metrics:
            return {"error": "No data available for analysis"}
            
        # Group by model and batch size
        groups = defaultdict(list)
        for metric in final_metrics:
            key = f"{metric['model_name']}_bs{metric['batch_size']}"
            groups[key].append(metric)
            
        analysis = {
            "model_combinations": list(groups.keys()),
            "fingerprint_counts": sorted(set(m['num_fingerprints'] for m in final_metrics)),
            "scalability_trends": {}
        }
        
        # Analyze trends for each model-batch combination
        for key, metrics in groups.items():
            # Sort by number of fingerprints
            metrics.sort(key=lambda x: x['num_fingerprints'])
            
            fp_counts = [m['num_fingerprints'] for m in metrics]
            fp_accs = [m['final_fp_acc'] for m in metrics]
            inst_accs = [m['final_inst_acc'] for m in metrics]
            
            analysis["scalability_trends"][key] = {
                "fingerprint_counts": fp_counts,
                "fingerprint_accuracies": fp_accs,
                "instruction_accuracies": inst_accs,
                "max_fingerprints": max(fp_counts) if fp_counts else 0,
                "min_instruction_acc": min(inst_accs) if inst_accs else 0,
                "max_instruction_acc": max(inst_accs) if inst_accs else 0
            }
            
        return analysis
        
    def compute_training_dynamics(self, model_name: str, batch_size: int) -> Optional[Dict[str, Any]]:
        """
        Analyze training dynamics for a specific model and batch size.
        
        Args:
            model_name: Name of the model
            batch_size: Batch size used in training
            
        Returns:
            Dictionary containing training dynamics analysis
        """
        # Find experiments matching the criteria
        matching_exps = [
            exp for exp in self.experiments 
            if exp.model_name == model_name and exp.batch_size == batch_size
        ]
        
        if not matching_exps:
            return None
            
        dynamics = {
            "model_name": model_name,
            "batch_size": batch_size,
            "num_experiments": len(matching_exps),
            "experiments": []
        }
        
        for exp in matching_exps:
            exp_dynamics = {
                "num_fingerprints": exp.num_fingerprints,
                "model_dir": exp.model_dir,
                "training_steps": exp.data['steps'].tolist(),
                "fingerprint_accuracy": (exp.data['fingerprint/acc_mean'] * 100).tolist(),
                "instruction_accuracy": (exp.data['U/ifeval/inst_level_strict_acc,none'] * 100).tolist(),
                "epochs": exp.data['epoch'].tolist()
            }
            dynamics["experiments"].append(exp_dynamics)
            
        return dynamics


def create_color_mapping(fingerprint_counts: List[int]) -> Dict[int, str]:
    """
    Create a color mapping for different fingerprint counts.
    
    Args:
        fingerprint_counts: List of unique fingerprint counts
        
    Returns:
        Dictionary mapping fingerprint counts to colors
    """
    # Use 12 distinct colors from the rainbow spectrum
    rainbow_spectrum = [
        '#8b4513',  # Brown
        '#ff0000',  # Red
        '#ffa500',  # Orange
        '#ffd700',  # Gold/Yellow
        '#ffff00',  # Yellow
        '#008080',  # Teal
        '#9acd32',  # Yellow-Green
        '#00ff00',  # Green
        '#00ffff',  # Cyan
        '#1e90ff',  # Dodger Blue
        '#0000ff',  # Blue
        '#8a2be2',  # Blue-Violet
    ]
    
    color_map = {}
    for i, fp_count in enumerate(sorted(fingerprint_counts)):
        color_index = i % len(rainbow_spectrum)
        color_map[fp_count] = rainbow_spectrum[color_index]
        
    return color_map 