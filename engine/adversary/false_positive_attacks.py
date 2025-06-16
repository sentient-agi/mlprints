"""
False Positive Adversarial Attacks

This module implements attacks that exploit sampling configurations to generate
false positives in fingerprint detection systems.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import torch
from torch import nn
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm

from .base import AdversaryBase, AttackResult, AttackConfig, AttackType
from ..common.llm_utils import load_model_and_tokenizer


@dataclass
class SamplingConfig:
    """Configuration for sampling parameters."""
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model.generate()."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p
        }
    
    def get_config_string(self) -> str:
        """Get string representation for identification."""
        return f"temp_{self.temperature}-p_{self.top_p}-k_{self.top_k}-min_p_{self.min_p}"


@dataclass
class FalsePositiveConfig(AttackConfig):
    """Configuration for false positive attacks."""
    fingerprint_file_path: str = ""
    num_fingerprints: int = 1024
    model_path: str = "tokyotech-llm/Llama-3.1-Swallow-8B-v0.1"
    num_mc_trials: int = 10
    batch_size: int = 32
    seed: int = 42
    use_adversarial_sampling: bool = False
    sampling_configs: List[SamplingConfig] = field(default_factory=list)
    output_dir: str = "results/fp_analysis"
    
    def __post_init__(self):
        """Initialize sampling configurations if not provided."""
        if not self.sampling_configs:
            if self.use_adversarial_sampling:
                self.sampling_configs = self._get_adversarial_sampling_configs()
            else:
                self.sampling_configs = self._get_standard_sampling_configs()
    
    def _get_standard_sampling_configs(self) -> List[SamplingConfig]:
        """Get standard sampling configurations."""
        return [
            SamplingConfig(temperature=1.0, top_p=1.0, top_k=0),
            SamplingConfig(temperature=0.9, top_p=0.9, top_k=50),
            SamplingConfig(temperature=0.7, top_p=0.8, top_k=40),
            SamplingConfig(temperature=1.2, top_p=0.95, top_k=100),
            SamplingConfig(temperature=0.6, top_p=0.7, top_k=20),
            SamplingConfig(temperature=1.5, top_p=1.0, top_k=200),
            SamplingConfig(temperature=0.85, top_p=0.85, top_k=60),
            SamplingConfig(temperature=1.3, top_p=0.92, top_k=80),
        ]
    
    def _get_adversarial_sampling_configs(self) -> List[SamplingConfig]:
        """Get adversarial sampling configurations designed to maximize false positives."""
        return [
            SamplingConfig(temperature=1.3, top_p=0.85, top_k=80, min_p=0.02),   # Balanced Adversarial
            SamplingConfig(temperature=1.8, top_p=0.75, top_k=60, min_p=0.01),   # Creative but Plausible
            SamplingConfig(temperature=2.2, top_p=0.65, top_k=40, min_p=0.005),  # High-Risk Adversarial
            SamplingConfig(temperature=2.8, top_p=0.55, top_k=25, min_p=0.002),  # Extreme Divergence
            SamplingConfig(temperature=3.5, top_p=0.5, top_k=15, min_p=0.001),   # Maximum Entropy Attack
            SamplingConfig(temperature=2.0, top_p=0.7, top_k=50, min_p=0.008),   # Hybrid Unlikely but Fluent
            SamplingConfig(temperature=1.7, top_p=0.78, top_k=70, min_p=0.015),  # Strategic Adversarial
            SamplingConfig(temperature=2.5, top_p=0.65, top_k=30, min_p=0.003),  # Rare-Word Manipulation
        ]


@dataclass
class FalsePositiveResult:
    """Result of false positive analysis for a single fingerprint."""
    effective_key: str
    response: str
    response_token_prob: float
    orig_prob: float
    response_token_log_prob: float
    response_token_idx: int
    top_10_tokens: List[str]
    correct: bool
    response_token_in_top_10: bool
    mc_correct_detailed: Dict[str, int]
    mc_correct: int


@dataclass
class FalsePositiveAttackResult(AttackResult):
    """Result of a false positive attack."""
    false_positives: int = 0
    false_positives_at_10: int = 0
    fp_with_sampling: int = 0
    total_sampling: int = 0
    fp_frac_with_sampling: float = 0.0
    fingerprint_results: List[FalsePositiveResult] = field(default_factory=list)


class FalsePositiveAttacker(AdversaryBase):
    """
    False Positive Attacker using sampling manipulation.
    
    This attack analyzes and exploits different sampling configurations
    to maximize false positive rates in fingerprint detection systems.
    """
    
    def __init__(self, config: FalsePositiveConfig):
        super().__init__(config)
        self.fp_config = config
        self.model = None
        self.tokenizer = None
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
    
    def _load_model_and_tokenizer(self):
        """Load the target model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = load_model_and_tokenizer(
                self.fp_config.model_path,
                device=self.fp_config.device,
                torch_dtype=torch.bfloat16
            )
            self.model.eval()
    
    def _load_fingerprint_data(self) -> List[Dict[str, Any]]:
        """Load fingerprint data from JSON file."""
        with open(self.fp_config.fingerprint_file_path, 'r') as f:
            data = json.load(f)
        return data[:self.fp_config.num_fingerprints]
    
    def _analyze_single_fingerprint(
        self, 
        fingerprint: Dict[str, Any]
    ) -> FalsePositiveResult:
        """Analyze a single fingerprint for false positives."""
        key = fingerprint['effective_key']
        response = fingerprint['response']
        orig_prob = fingerprint['response_prob'][0]
        
        # Tokenize inputs
        tok_key = self.tokenizer(key, return_tensors='pt', add_special_tokens=False)
        tok_response = self.tokenizer(response, return_tensors='pt', add_special_tokens=False)
        
        tok_key = {k: v.to(self.fp_config.device) for k, v in tok_key.items()}
        
        with torch.no_grad():
            # Get model predictions
            outputs = self.model(**tok_key)
            logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Analyze response token
            response_token_id = tok_response['input_ids'][0][0]
            response_token_prob = probs[0, response_token_id]
            response_token_log_prob = log_probs[0, response_token_id]
            
            # Get token ranking
            sorted_probs, idxs = torch.sort(logits, descending=True)
            response_token_idx = torch.where(idxs[0] == response_token_id)[0]
            
            # Get top 10 tokens
            top_10_tokens = self.tokenizer.convert_ids_to_tokens(idxs[0][:10].cpu().numpy())
            
            # Monte Carlo sampling analysis
            mc_results = {}
            for sampling_config in self.fp_config.sampling_configs:
                config_str = sampling_config.get_config_string()
                
                # Generate multiple samples
                generated_tokens = self.model.generate(
                    **tok_key,
                    max_new_tokens=1,
                    do_sample=True,
                    num_return_sequences=self.fp_config.num_mc_trials,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **sampling_config.to_dict()
                )[:, -1]
                
                # Count correct matches
                correct_count = (generated_tokens == response_token_id).sum().item()
                mc_results[config_str] = correct_count
        
        return FalsePositiveResult(
            effective_key=key,
            response=response,
            response_token_prob=response_token_prob.item(),
            orig_prob=orig_prob,
            response_token_log_prob=response_token_log_prob.item(),
            response_token_idx=response_token_idx.item(),
            top_10_tokens=top_10_tokens,
            correct=response_token_idx.item() == 0,
            response_token_in_top_10=response_token_idx.item() < 10,
            mc_correct_detailed=mc_results,
            mc_correct=sum(mc_results.values())
        )
    
    def _analyze_batch_fingerprints(
        self, 
        fingerprints: List[Dict[str, Any]]
    ) -> List[FalsePositiveResult]:
        """Analyze a batch of fingerprints efficiently."""
        batch_keys = [fp['effective_key'] for fp in fingerprints]
        batch_responses = [fp['response'] for fp in fingerprints]
        batch_orig_probs = [fp['response_prob'][0] for fp in fingerprints]
        
        # Tokenize batch
        tok_keys = self.tokenizer(
            batch_keys, 
            return_tensors='pt', 
            add_special_tokens=False,
            max_length=16,
            truncation=True,
            padding=False
        )
        tok_responses = self.tokenizer(
            batch_responses,
            return_tensors='pt',
            add_special_tokens=False,
            max_length=8,
            truncation=True,
            padding=False
        )
        
        tok_keys = {k: v.to(self.fp_config.device) for k, v in tok_keys.items()}
        tok_responses = tok_responses['input_ids'].to(self.fp_config.device)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(**tok_keys)
            logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Extract response token analysis
            response_token_ids = tok_responses[:, 0]
            response_token_probs = probs[torch.arange(len(response_token_ids)), response_token_ids]
            response_token_log_probs = log_probs[torch.arange(len(response_token_ids)), response_token_ids]
            
            sorted_probs, idxs = torch.sort(logits, descending=True)
            response_token_idxs = torch.where(idxs == response_token_ids.unsqueeze(1))[1]
            
            # Monte Carlo sampling for batch
            batch_mc_results = []
            for i in range(len(fingerprints)):
                mc_results = {}
                for sampling_config in self.fp_config.sampling_configs:
                    config_str = sampling_config.get_config_string()
                    
                    # Extract single example for generation
                    single_tok_keys = {k: v[i:i+1] for k, v in tok_keys.items()}
                    
                    generated_tokens = self.model.generate(
                        **single_tok_keys,
                        max_new_tokens=1,
                        do_sample=True,
                        num_return_sequences=self.fp_config.num_mc_trials,
                        pad_token_id=self.tokenizer.eos_token_id,
                        **sampling_config.to_dict()
                    )[:, -1]
                    
                    correct_count = (generated_tokens == response_token_ids[i]).sum().item()
                    mc_results[config_str] = correct_count
                
                batch_mc_results.append(mc_results)
        
        # Create results
        results = []
        for i in range(len(fingerprints)):
            top_10_tokens = self.tokenizer.convert_ids_to_tokens(idxs[i][:10].cpu().numpy())
            
            result = FalsePositiveResult(
                effective_key=batch_keys[i],
                response=batch_responses[i],
                response_token_prob=response_token_probs[i].item(),
                orig_prob=batch_orig_probs[i],
                response_token_log_prob=response_token_log_probs[i].item(),
                response_token_idx=response_token_idxs[i].item(),
                top_10_tokens=top_10_tokens,
                correct=response_token_idxs[i].item() == 0,
                response_token_in_top_10=response_token_idxs[i].item() < 10,
                mc_correct_detailed=batch_mc_results[i],
                mc_correct=sum(batch_mc_results[i].values())
            )
            results.append(result)
        
        return results
    
    def attack(
        self,
        model: Optional[nn.Module] = None,
        input_text: Union[str, List[str]] = None,
        target_output: Optional[str] = None
    ) -> FalsePositiveAttackResult:
        """
        Execute the false positive attack.
        
        Args:
            model: Optional model (will use config model if not provided)
            input_text: Not used for this attack type
            target_output: Not used for this attack type
            
        Returns:
            FalsePositiveAttackResult with analysis results
        """
        # Load model if not provided
        if model is not None:
            self.model = model
        else:
            self._load_model_and_tokenizer()
        
        # Load fingerprint data
        fingerprint_data = self._load_fingerprint_data()
        
        # Initialize counters
        false_positives = 0
        false_positives_at_10 = 0
        fp_with_sampling = 0
        total_sampling = 0
        all_results = []
        
        # Process in batches or individually
        if self.fp_config.batch_size == 1:
            # Single processing
            for fingerprint in tqdm(fingerprint_data, desc="Analyzing fingerprints"):
                result = self._analyze_single_fingerprint(fingerprint)
                all_results.append(result)
                
                # Update counters
                if result.correct:
                    false_positives += 1
                if result.response_token_in_top_10:
                    false_positives_at_10 += 1
                
                fp_with_sampling += result.mc_correct
                total_sampling += self.fp_config.num_mc_trials * len(self.fp_config.sampling_configs)
        else:
            # Batch processing
            for batch_start in tqdm(
                range(0, len(fingerprint_data), self.fp_config.batch_size),
                desc="Processing batches"
            ):
                batch_end = min(batch_start + self.fp_config.batch_size, len(fingerprint_data))
                batch_fingerprints = fingerprint_data[batch_start:batch_end]
                
                batch_results = self._analyze_batch_fingerprints(batch_fingerprints)
                all_results.extend(batch_results)
                
                # Update counters
                for result in batch_results:
                    if result.correct:
                        false_positives += 1
                    if result.response_token_in_top_10:
                        false_positives_at_10 += 1
                    
                    fp_with_sampling += result.mc_correct
                    total_sampling += self.fp_config.num_mc_trials * len(self.fp_config.sampling_configs)
        
        # Create attack result
        attack_result = FalsePositiveAttackResult(
            success=fp_with_sampling > 0,  # Success if any false positives found
            original_input=f"Fingerprint file: {self.fp_config.fingerprint_file_path}",
            adversarial_input=f"Model: {self.fp_config.model_path}",
            original_output=f"Expected 0 false positives",
            adversarial_output=f"Found {fp_with_sampling} false positives",
            confidence=fp_with_sampling / total_sampling if total_sampling > 0 else 0.0,
            attack_type=AttackType.LOGIT_BASED,  # Closest existing type
            metadata={
                "num_fingerprints": len(fingerprint_data),
                "batch_size": self.fp_config.batch_size,
                "num_mc_trials": self.fp_config.num_mc_trials,
                "sampling_configs": [cfg.get_config_string() for cfg in self.fp_config.sampling_configs]
            },
            false_positives=false_positives,
            false_positives_at_10=false_positives_at_10,
            fp_with_sampling=fp_with_sampling,
            total_sampling=total_sampling,
            fp_frac_with_sampling=fp_with_sampling / total_sampling if total_sampling > 0 else 0.0,
            fingerprint_results=all_results
        )
        
        return attack_result
    
    def evaluate_attack_success(
        self,
        original_output: str,
        adversarial_output: str,
        target_output: Optional[str] = None
    ) -> bool:
        """Evaluate if the false positive attack was successful."""
        # Success is measured by finding false positives
        return "Found" in adversarial_output and "false positives" in adversarial_output
    
    def save_results(self, result: FalsePositiveAttackResult, output_path: Optional[str] = None):
        """Save attack results to JSON file."""
        if output_path is None:
            # Generate output filename
            model_name = self.fp_config.model_path.replace('/', '-')
            fingerprint_name = Path(self.fp_config.fingerprint_file_path).stem
            attack_type = "adversarial" if self.fp_config.use_adversarial_sampling else "standard"
            
            output_dir = Path(self.fp_config.output_dir) / f"fp_analysis_{attack_type}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{fingerprint_name}-{model_name}.json"
        
        # Prepare data for saving
        save_data = {
            "config": {
                "fingerprint_file_path": self.fp_config.fingerprint_file_path,
                "num_fingerprints": self.fp_config.num_fingerprints,
                "model_path": self.fp_config.model_path,
                "num_mc_trials": self.fp_config.num_mc_trials,
                "batch_size": self.fp_config.batch_size,
                "seed": self.fp_config.seed,
                "use_adversarial_sampling": self.fp_config.use_adversarial_sampling,
            },
            "results": {
                "success": result.success,
                "false_positives": result.false_positives,
                "false_positives_at_10": result.false_positives_at_10,
                "fp_with_sampling": result.fp_with_sampling,
                "total_sampling": result.total_sampling,
                "fp_frac_with_sampling": result.fp_frac_with_sampling,
                "confidence": result.confidence,
            },
            "fingerprint_results": [
                {
                    "effective_key": fp.effective_key,
                    "response": fp.response,
                    "response_token_prob": fp.response_token_prob,
                    "orig_prob": fp.orig_prob,
                    "response_token_log_prob": fp.response_token_log_prob,
                    "response_token_idx": fp.response_token_idx,
                    "top_10_tokens": fp.top_10_tokens,
                    "correct": fp.correct,
                    "response_token_in_top_10": fp.response_token_in_top_10,
                    "mc_correct_detailed": fp.mc_correct_detailed,
                    "mc_correct": fp.mc_correct,
                }
                for fp in result.fingerprint_results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Results saved to: {output_path}") 