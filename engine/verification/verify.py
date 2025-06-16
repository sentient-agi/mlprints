"""
Verification Engine

This module implements the main verification engine that orchestrates
fingerprint verification processes and provides comprehensive verification results.
"""

import os
import json
import time
import torch
import numpy as np
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Import transformers for model loading and inference
try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    raise ImportError("transformers library is required. Install with: pip install transformers")

# Optional WandB support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .base import FingerprintConfig, VerificationMode
from .fingerprints import FingerprintSet, SimpleFingerprintSet, load_fingerprint_set_from_file


@dataclass
class VerificationResult:
    """Result of fingerprint verification."""
    total_fingerprints: int
    successful_verifications: int
    failed_verifications: int
    success_rate: float
    fractional_success_rate: float  # Token-level accuracy
    verification_time: float
    verification_vector: List[bool]  # V vector - boolean array of verification results
    confidence_scores: List[float]
    detailed_results: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class BatchVerificationResult:
    """Result of batch verification across multiple fingerprint sets."""
    individual_results: List[VerificationResult]
    overall_success_rate: float
    total_verification_time: float
    combined_verification_vector: List[bool]
    metadata: Dict[str, Any]


class VerificationEngine:
    """
    Main verification engine for fingerprint-based model verification.
    
    This class orchestrates the verification process, handles multiple
    fingerprint types, and provides comprehensive verification reporting.
    """
    
    def __init__(self, config: FingerprintConfig):
        self.config = config
        self.fingerprint_sets = {}
        self.verification_history = []
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.wandb_run = None
        
    def _get_device(self) -> str:
        """Determine the appropriate device for inference."""
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device
    
    def load_config_from_model_dir(self, model_path: str) -> Dict[str, Any]:
        """
        Load configuration from model directory.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Configuration dictionary
        """
        config_files = [
            "fingerprinting_config.json",
            "finetuning_config.json", 
            "merging_config.json"
        ]
        
        for config_file in config_files:
            config_path = os.path.join(model_path, config_file)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"Loaded config from {config_path}")
                return config
        
        print(f"WARNING: No config file found in {model_path}")
        return {}
    
    def initialize_wandb(self, run_name: str, config_dict: Dict[str, Any]):
        """Initialize WandB logging if available."""
        if WANDB_AVAILABLE and run_name != 'None':
            self.wandb_run = wandb.init(project=run_name, config=config_dict)
            print(f"Initialized WandB run: {run_name}")
        elif not WANDB_AVAILABLE:
            print("WandB not available. Install with: pip install wandb")
    
    def load_model(self, model_path: str, use_chat_template: bool = False) -> None:
        """
        Load model and tokenizer from path.
        
        Args:
            model_path: Path to the model directory
            use_chat_template: Whether to use chat template for instruction models
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        print(f"Loading model from {model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            if self.tokenizer.padding_side == 'right':
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = self.tokenizer.bos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=self.device if self.device != "cpu" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        
        self.model.eval()
        self.use_chat_template = use_chat_template
        
        print(f"Model loaded successfully on {self.device}")
    
    def load_fingerprint_dataset(self, file_path: str, name: Optional[str] = None, 
                                min_fingerprints: Optional[int] = None) -> str:
        """
        Load fingerprint dataset from file.
        
        Args:
            file_path: Path to fingerprint dataset JSON file
            name: Optional name for the fingerprint set
            min_fingerprints: Optional minimum number of fingerprints required
            
        Returns:
            Name of the loaded fingerprint set
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fingerprint file does not exist: {file_path}")
        
        # Determine name
        if name is None:
            name = Path(file_path).stem
        
        # Load fingerprint dataset using the existing abstraction
        try:
            fingerprint_set = load_fingerprint_set_from_file(file_path)
            fingerprint_set.name = name
            self.fingerprint_sets[name] = fingerprint_set
        except Exception as e:
            # Fallback: try loading as raw JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert to expected format
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict) and 'key' in data[0] and 'response' in data[0]:
                    pairs = data
                else:
                    raise ValueError(f"Invalid fingerprint format in {file_path}")
            else:
                raise ValueError(f"Invalid fingerprint format in {file_path}")
            
            fingerprint_set = SimpleFingerprintSet(pairs, name)
            self.fingerprint_sets[name] = fingerprint_set
        
        # Validate minimum count if specified
        if min_fingerprints is not None:
            actual_count = len(fingerprint_set)
            if actual_count < min_fingerprints:
                raise ValueError(
                    f"Insufficient fingerprints in {file_path}: "
                    f"found {actual_count}, required at least {min_fingerprints}"
                )
            elif actual_count > min_fingerprints:
                print(
                    f"WARNING: Fingerprint file {file_path} contains {actual_count} fingerprints, "
                    f"more than the {min_fingerprints} that will be used for verification"
                )
        
        print(f"Loaded {len(fingerprint_set)} fingerprints from {file_path}")
        return name
    
    def add_fingerprint_set(self, name: str, fingerprint_set: FingerprintSet):
        """Add a fingerprint set to the verification engine."""
        self.fingerprint_sets[name] = fingerprint_set
        
    def _get_model_response(self, input_text: str, max_new_tokens: int = None, temperature: float = 0.0) -> str:
        """Get response from model for given input."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Apply chat template if enabled
        if self.use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            conversation = [{"role": "user", "content": input_text}]
            formatted_input = self.tokenizer.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
        else:
            formatted_input = input_text
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_input, 
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        
        # Move to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Strip EOS token from input if present
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        if input_ids[0, -1] == self.tokenizer.eos_token_id:
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]
        
        # Determine generation length
        if max_new_tokens is None:
            max_new_tokens = max(self.config.max_response_length, 1)
        
        # Determine sampling parameters
        do_sample = temperature > 0.0
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                use_cache=True
            )
        
        # Extract only the generated part
        generated_tokens = outputs[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def _get_top_k_tokens(self, input_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """Get top-k next token predictions for debugging."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Apply chat template if enabled
        if self.use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            conversation = [{"role": "user", "content": input_text}]
            formatted_input = self.tokenizer.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
        else:
            formatted_input = input_text
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_input, 
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        
        # Move to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Strip EOS token from input if present
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        if input_ids[0, -1] == self.tokenizer.eos_token_id:
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]
        
        # Get logits
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            topk = torch.topk(logits, k, dim=-1)
            
        # Extract top-k tokens and probabilities
        topk_tokens = []
        for token_id, prob in zip(topk.indices[0], probabilities[0][topk.indices[0]]):
            token_str = self.tokenizer.decode([token_id])
            topk_tokens.append({
                "token": token_str,
                "token_id": token_id.item(),
                "probability": prob.item()
            })
        
        return topk_tokens
    
    def _preprocess_response(self, response: str) -> str:
        """Preprocess model response for verification."""
        # Basic cleaning
        response = response.strip()
        # Remove extra whitespace
        response = ' '.join(response.split())
        return response
    
    def _compute_verification_confidence(self, expected: str, actual: str) -> float:
        """Compute confidence score between expected and actual responses."""
        if not expected or not actual:
            return 0.0
        
        # Simple word overlap based confidence
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        
        if len(expected_words) == 0 and len(actual_words) == 0:
            return 1.0
        elif len(expected_words) == 0 or len(actual_words) == 0:
            return 0.0
        
        overlap = len(expected_words & actual_words)
        total = len(expected_words | actual_words)
        return overlap / total
    
    def verify_fingerprint(
        self,
        fingerprint_key: str,
        expected_response: str,
        actual_response: str
    ) -> Tuple[bool, float]:
        """
        Verify a single fingerprint.
        
        Returns:
            Tuple of (is_verified, confidence_score)
        """
        # Preprocess the response
        actual_response = self._preprocess_response(actual_response)
        expected_response = self._preprocess_response(expected_response)
        
        # Compute confidence
        confidence = self._compute_verification_confidence(expected_response, actual_response)
        
        # Apply verification mode
        if self.config.verification_mode == VerificationMode.EXACT_MATCH:
            is_verified = actual_response == expected_response
        elif self.config.verification_mode == VerificationMode.PREFIX_MATCH:
            is_verified = actual_response.startswith(expected_response)
        elif self.config.verification_mode == VerificationMode.SEMANTIC_MATCH:
            is_verified = confidence >= self.config.threshold
        elif self.config.verification_mode == VerificationMode.FUNCTIONAL_MATCH:
            # For functional fingerprints, use the fingerprint set's logic
            is_verified = confidence >= self.config.threshold
        else:
            # Default to prefix match
            is_verified = actual_response.startswith(expected_response)
            
        return is_verified, confidence
    
    def verify_fingerprint_set(
        self,
        fingerprint_set: FingerprintSet,
        num_samples: Optional[int] = None,
        verbose: bool = False,
        prompt_templates: List[str] = ["{}"],
        temperature: float = 0.0,
        output_file_path: Optional[str] = None
    ) -> VerificationResult:
        """
        Verify a fingerprint set against the loaded model.
        
        Args:
            fingerprint_set: The fingerprint set to verify
            num_samples: Number of samples to test (None for all)
            verbose: Whether to print detailed results
            prompt_templates: List of prompt templates to test with
            temperature: Temperature for sampling (0.0 for greedy)
            output_file_path: Optional path to save detailed results
            
        Returns:
            VerificationResult containing detailed verification information
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Setup output file if specified
        output_file = None
        if output_file_path is not None:
            output_file = open(output_file_path, 'a')
        
        # Get fingerprint pairs
        all_pairs = fingerprint_set.all_pairs()
        if num_samples is not None:
            all_pairs = all_pairs[:num_samples]
        
        # Initialize tracking variables for multiple prompt templates
        num_templates = len(prompt_templates)
        correct_per_template = np.array([0 for _ in range(num_templates)])
        fractional_correct_per_template = np.array([0.0 for _ in range(num_templates)])
        fractional_total_per_template = np.array([0.0 for _ in range(num_templates)])
        
        verification_vectors = [[] for _ in range(num_templates)]
        confidence_scores = []
        detailed_results = []
        total_fingerprints = 0
        
        print(f"Verifying {len(all_pairs)} fingerprints with {num_templates} prompt templates...")
        
        for eidx, pair in enumerate(all_pairs):
            key = pair.get('key', '')
            expected = pair.get('response', '')
            
            if not key or not expected:
                if verbose:
                    print(f"Skipping invalid pair {eidx}: {pair}")
                for pidx in range(num_templates):
                    verification_vectors[pidx].append(False)
                continue
            
            # Handle multiple responses per fingerprint
            if isinstance(expected, list) and len(expected) > 1:
                expected_responses = expected
            else:
                expected_responses = [expected] if isinstance(expected, str) else [expected[0]]
                
            for pidx, prompt_template in enumerate(prompt_templates):
                try:
                    # Format the key with the prompt template
                    formatted_key = prompt_template.format(key)
                    
                    # Tokenize the key
                    key_tokenized = self.tokenizer(formatted_key, return_tensors='pt')
                    
                    # Strip EOS token from key if present
                    if key_tokenized['input_ids'][0][-1] == self.tokenizer.eos_token_id:
                        key_input_ids = key_tokenized['input_ids'][:, :-1]
                        key_attention_mask = key_tokenized['attention_mask'][:, :-1]
                    else:
                        key_input_ids = key_tokenized['input_ids']
                        key_attention_mask = key_tokenized['attention_mask']
                    
                    # Tokenize expected responses
                    if len(expected_responses) > 1:
                        expected_tokenized = []
                        for resp in expected_responses:
                            resp_tokens = self.tokenizer(resp, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze(0)
                            if self.device != "cpu":
                                resp_tokens = resp_tokens.cuda()
                            # Strip BOS token if present
                            if len(resp_tokens) > 0 and resp_tokens[0] == self.tokenizer.bos_token_id:
                                resp_tokens = resp_tokens[1:]
                            expected_tokenized.append(resp_tokens)
                        gen_len = len(expected_tokenized[0]) if len(expected_tokenized) > 0 and len(expected_tokenized[0]) > 0 else 1
                    else:
                        expected_tokenized = self.tokenizer(expected_responses[0], return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze(0)
                        if self.device != "cpu":
                            expected_tokenized = expected_tokenized.cuda()
                        # Strip BOS token if present
                        if len(expected_tokenized) > 0 and expected_tokenized[0] == self.tokenizer.bos_token_id:
                            expected_tokenized = expected_tokenized[1:]
                        gen_len = len(expected_tokenized)
                    
                    gen_len = max(gen_len, 1)
                    do_sample = temperature > 0.0
                    
                    # Generate response
                    if self.device != "cpu":
                        key_input_ids = key_input_ids.cuda()
                        key_attention_mask = key_attention_mask.cuda()
                    
                    outputs = self.model.generate(
                        input_ids=key_input_ids,
                        attention_mask=key_attention_mask,
                        max_new_tokens=gen_len,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                    )
                    
                    # Extract generated tokens
                    prediction = outputs[0][key_input_ids.shape[1]:]
                    
                    # Verify the response
                    is_verified = False
                    max_fractional_score = 0.0
                    
                    if len(expected_responses) == 1:
                        # Single expected response
                        if torch.equal(prediction, expected_tokenized):
                            is_verified = True
                            correct_per_template[pidx] += 1
                        
                        # Compute fractional accuracy
                        if len(expected_tokenized) > 0:
                            fractional_score = (prediction == expected_tokenized[:len(prediction)]).sum().item()
                            fractional_correct_per_template[pidx] += fractional_score
                            fractional_total_per_template[pidx] += len(expected_tokenized)
                            max_fractional_score = fractional_score / len(expected_tokenized)
                        
                        if verbose and not is_verified:
                            print(f"Idx-{eidx} Template-{pidx} - Decoded output: {self.tokenizer.decode(prediction)}")
                            print(f"    Expected: {expected_responses[0]}")
                            print(f"    Formatted key: {formatted_key}")
                            
                            # Get top-k tokens for debugging
                            try:
                                top_k_tokens = self._get_top_k_tokens(formatted_key, k=5)
                                top_k_str = ', '.join([f"{t['token']} - {t['probability']:.3f}" for t in top_k_tokens])
                                print(f"    Top 5 tokens with probs: {top_k_str}")
                            except Exception as e:
                                print(f"    Error getting top-k tokens: {e}")
                    
                    else:
                        # Multiple expected responses
                        fractional_total_per_template[pidx] += len(expected_tokenized[0])
                        
                        for resp_tokens in expected_tokenized:
                            try:
                                if torch.equal(prediction, resp_tokens):
                                    is_verified = True
                                    correct_per_template[pidx] += 1
                                    break
                                # Track best fractional match
                                if len(resp_tokens) > 0:
                                    fractional_score = (prediction == resp_tokens[:len(prediction)]).sum().item()
                                    max_fractional_score = max(max_fractional_score, fractional_score)
                            except Exception as e:
                                print(f"Error in comparison: {e}")
                        
                        fractional_correct_per_template[pidx] += max_fractional_score
                    
                    verification_vectors[pidx].append(is_verified)
                    
                    # Log to output file if specified
                    if output_file is not None:
                        output_file.write(f"Idx-{eidx} Template-{pidx} - Decoded output: {self.tokenizer.decode(prediction)}\n")
                        output_file.write(f"    Expected: {expected_responses[0] if len(expected_responses) == 1 else expected_responses}\n")
                        output_file.write(f"    Formatted key: {formatted_key}\n")
                        output_file.write(f"    Verified: {is_verified}\n\n")
                    
                except Exception as e:
                    if verbose:
                        print(f"Error processing fingerprint {eidx} with template {pidx}: {e}")
                    verification_vectors[pidx].append(False)
            
            total_fingerprints += 1
        
        # Close output file if opened
        if output_file is not None:
            output_file.close()
        
        verification_time = time.time() - start_time
        
        # Calculate success rates
        success_rates = (correct_per_template / total_fingerprints) * 100 if total_fingerprints > 0 else np.zeros(num_templates)
        fractional_success_rates = (fractional_correct_per_template / fractional_total_per_template) * 100
        
        # Use the first template's results as primary results
        primary_success_rate = success_rates[0] / 100 if len(success_rates) > 0 else 0.0
        primary_fractional_rate = fractional_success_rates[0] / 100 if len(fractional_success_rates) > 0 else 0.0
        primary_verification_vector = verification_vectors[0] if len(verification_vectors) > 0 else []
        
        result = VerificationResult(
            total_fingerprints=total_fingerprints,
            successful_verifications=int(correct_per_template[0]) if len(correct_per_template) > 0 else 0,
            failed_verifications=total_fingerprints - int(correct_per_template[0]) if len(correct_per_template) > 0 else total_fingerprints,
            success_rate=primary_success_rate,
            fractional_success_rate=primary_fractional_rate,
            verification_time=verification_time,
            verification_vector=primary_verification_vector,
            confidence_scores=confidence_scores,
            detailed_results=detailed_results,
            metadata={
                "fingerprint_set_name": fingerprint_set.name,
                "verification_mode": self.config.verification_mode.value,
                "threshold": self.config.threshold,
                "device": self.device,
                "use_chat_template": getattr(self, 'use_chat_template', False),
                "temperature": temperature,
                "num_prompt_templates": num_templates,
                "prompt_templates": prompt_templates,
                "success_rates_per_template": success_rates.tolist(),
                "fractional_success_rates_per_template": fractional_success_rates.tolist(),
                "verification_vectors_per_template": verification_vectors
            }
        )
        
        # Store in history
        self.verification_history.append(result)
        
        print(f"Verification completed: {result.successful_verifications}/{total_fingerprints} "
              f"({primary_success_rate:.1%}) in {verification_time:.2f}s")
        
        if num_templates > 1:
            print(f"Success rates per template: {[f'{rate:.1f}%' for rate in success_rates]}")
        
        return result
    
    def verify_all_fingerprint_sets(
        self,
        num_samples: Optional[int] = None,
        verbose: bool = False
    ) -> BatchVerificationResult:
        """
        Verify all loaded fingerprint sets.
        
        Args:
            num_samples: Number of samples per set (None for all)
            verbose: Whether to print detailed results
            
        Returns:
            BatchVerificationResult containing results for all sets
        """
        if not self.fingerprint_sets:
            raise RuntimeError("No fingerprint sets loaded. Call load_fingerprint_dataset() first.")
        
        start_time = time.time()
        individual_results = []
        combined_verification_vector = []
        total_successful = 0
        total_fingerprints = 0
        
        for set_name, fingerprint_set in self.fingerprint_sets.items():
            print(f"\nVerifying fingerprint set: {set_name}")
            result = self.verify_fingerprint_set(fingerprint_set, num_samples, verbose)
            individual_results.append(result)
            combined_verification_vector.extend(result.verification_vector)
            total_successful += result.successful_verifications
            total_fingerprints += result.total_fingerprints
        
        total_time = time.time() - start_time
        overall_success_rate = total_successful / total_fingerprints if total_fingerprints > 0 else 0.0
        
        batch_result = BatchVerificationResult(
            individual_results=individual_results,
            overall_success_rate=overall_success_rate,
            total_verification_time=total_time,
            combined_verification_vector=combined_verification_vector,
            metadata={
                "num_sets": len(self.fingerprint_sets),
                "total_fingerprints": total_fingerprints,
                "total_successful": total_successful,
                "device": self.device
            }
        )
        
        return batch_result
    
    def verify_from_files(
        self,
        model_path: str,
        fingerprint_files: Union[str, List[str]],
        use_chat_template: bool = False,
        num_samples: Optional[int] = None,
        verbose: bool = False,
        min_fingerprints: Optional[int] = None,
        use_cache: bool = False,
        cache_dir: str = "verification_results"
    ) -> Union[VerificationResult, BatchVerificationResult]:
        """
        Convenience method to verify fingerprints directly from files.
        
        Args:
            model_path: Path to the model
            fingerprint_files: Path(s) to fingerprint JSON files
            use_chat_template: Whether to use chat template
            num_samples: Number of samples to test per file
            verbose: Whether to print detailed results
            min_fingerprints: Optional minimum number of fingerprints required per file
            use_cache: Whether to use cached results if available
            cache_dir: Directory for caching verification results
            
        Returns:
            VerificationResult (single file) or BatchVerificationResult (multiple files)
        """
        # Load fingerprint files
        if isinstance(fingerprint_files, str):
            fingerprint_files = [fingerprint_files]
        
        # Check for cached results if enabled
        if use_cache and len(fingerprint_files) == 1:
            cached_result = self.load_verification_results(
                model_path, fingerprint_files[0], cache_dir
            )
            if cached_result is not None:
                print(f"Using cached verification results for {model_path} + {fingerprint_files[0]}")
                return cached_result
        
        # Load model
        self.load_model(model_path, use_chat_template)
        
        for file_path in fingerprint_files:
            self.load_fingerprint_dataset(file_path, min_fingerprints=min_fingerprints)
        
        # Verify
        if len(fingerprint_files) == 1:
            set_name = list(self.fingerprint_sets.keys())[0]
            result = self.verify_fingerprint_set(
                self.fingerprint_sets[set_name], 
                num_samples, 
                verbose
            )
            
            # Save to cache if enabled
            if use_cache:
                self.save_verification_results(result, model_path, fingerprint_files[0], cache_dir)
            
            return result
        else:
            return self.verify_all_fingerprint_sets(num_samples, verbose)
    
    def get_verification_vector(self, fingerprint_set_name: Optional[str] = None) -> List[bool]:
        """
        Get the verification vector (V vector) for a specific set or all sets.
        
        Args:
            fingerprint_set_name: Name of specific set (None for combined vector)
            
        Returns:
            Boolean list representing verification results
        """
        if not self.verification_history:
            raise RuntimeError("No verification results available. Run verification first.")
        
        if fingerprint_set_name is not None:
            # Find results for specific set
            for result in self.verification_history:
                if result.metadata.get("fingerprint_set_name") == fingerprint_set_name:
                    return result.verification_vector
            raise ValueError(f"No results found for fingerprint set: {fingerprint_set_name}")
        else:
            # Return combined vector from most recent verification
            latest_result = self.verification_history[-1]
            return latest_result.verification_vector


def eval_driver(
    model_path: str, 
    num_fingerprints: int, 
    max_key_length: int, 
    max_response_length: int,
    fingerprint_generation_strategy: str = 'english', 
    fingerprints_file_path: str = None,
    verbose_eval: bool = False, 
    wandb_run_name: str = 'None', 
    prompt_templates: List[str] = ['{}'],  
    delete_model: bool = False, 
    sampling_temperature: float = 0.0, 
    seed: int = 42,
    use_augmentation_prompts: bool = False
) -> Union[VerificationResult, BatchVerificationResult]:
    """
    Convenience function that mirrors eval_driver from check_fingerprints.py.
    
    Args:
        model_path: Path to the model to be verified
        num_fingerprints: Number of fingerprints to verify
        max_key_length: Maximum key length
        max_response_length: Maximum response length
        fingerprint_generation_strategy: Strategy used for fingerprint generation
        fingerprints_file_path: Path to fingerprint file
        verbose_eval: Whether to print verbose evaluation details
        wandb_run_name: WandB run name for logging
        prompt_templates: List of prompt templates to test
        delete_model: Whether to delete model after verification
        sampling_temperature: Temperature for sampling
        seed: Random seed for reproducibility
        use_augmentation_prompts: Whether to use augmentation prompts
        
    Returns:
        Verification result
    """
    from .base import create_simple_config, VerificationMode
    
    # Create configuration
    config = create_simple_config(
        verification_mode=VerificationMode.EXACT_MATCH,
        max_key_length=max_key_length,
        max_response_length=max_response_length
    )
    
    # Create verification engine
    engine = VerificationEngine(config)
    
    # Use default fingerprint file path if not provided
    if fingerprints_file_path is None:
        fingerprints_file_path = f'{os.getcwd()}/generated_data/output_fingerprints.json'
    
    # Run comprehensive verification
    return engine.comprehensive_verification(
        model_path=model_path,
        fingerprint_files=fingerprints_file_path,
        num_fingerprints=num_fingerprints,
        max_key_length=max_key_length,
        max_response_length=max_response_length,
        fingerprint_generation_strategy=fingerprint_generation_strategy,
        verbose_eval=verbose_eval,
        wandb_run_name=wandb_run_name,
        prompt_templates=prompt_templates,
        sampling_temperature=sampling_temperature,
        delete_model=delete_model,
        seed=seed,
        use_augmentation_prompts=use_augmentation_prompts
    )


def eval_backdoor_acc(
    model,
    tokenizer, 
    dataset, 
    prompt_templates: List[str] = ["{}"], 
    temperature: float = 0.0, 
    verbose: bool = True, 
    output_file_path: Optional[str] = None, 
    use_chat_template: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function that mirrors eval_backdoor_acc from check_fingerprints.py.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        dataset: Dataset containing fingerprint pairs
        prompt_templates: List of prompt templates to test
        temperature: Temperature for sampling
        verbose: Whether to print verbose output
        output_file_path: Optional path to save results
        use_chat_template: Whether to use chat template
        
    Returns:
        Tuple of (accuracy_per_template, fractional_accuracy_per_template)
    """
    from .base import create_simple_config, VerificationMode
    
    # Create a simple fingerprint set from the dataset
    if hasattr(dataset, 'all_pairs'):
        pairs = dataset.all_pairs()
    else:
        # Assume dataset is iterable with dict items
        pairs = list(dataset)
    
    fingerprint_set = SimpleFingerprintSet(pairs, name="eval_dataset")
    
    # Create configuration and engine
    config = create_simple_config(verification_mode=VerificationMode.EXACT_MATCH)
    engine = VerificationEngine(config)
    
    # Set the model and tokenizer directly
    engine.model = model
    engine.tokenizer = tokenizer
    engine.use_chat_template = use_chat_template
    
    # Run verification
    result = engine.verify_fingerprint_set(
        fingerprint_set,
        verbose=verbose,
        prompt_templates=prompt_templates,
        temperature=temperature,
        output_file_path=output_file_path
    )
    
    # Extract results per template
    success_rates = np.array(result.metadata.get('success_rates_per_template', [result.success_rate * 100]))
    fractional_rates = np.array(result.metadata.get('fractional_success_rates_per_template', [result.fractional_success_rate * 100]))
    
    return success_rates, fractional_rates


    def generate_verification_report(
        self,
        result: Union[VerificationResult, BatchVerificationResult],
        include_details: bool = True,
        save_to_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive verification report."""
        if isinstance(result, VerificationResult):
            report = self._generate_single_result_report(result, include_details)
        else:
            report = self._generate_batch_result_report(result, include_details)
        
        if save_to_file:
            with open(save_to_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {save_to_file}")
        
        return report
    
    def _generate_single_result_report(
        self, 
        result: VerificationResult, 
        include_details: bool
    ) -> Dict[str, Any]:
        """Generate report for single verification result."""
        report = {
            "summary": {
                "total_fingerprints": result.total_fingerprints,
                "successful_verifications": result.successful_verifications,
                "failed_verifications": result.failed_verifications,
                "success_rate": result.success_rate,
                "fractional_success_rate": result.fractional_success_rate,
                "verification_time": result.verification_time,
                "average_confidence": np.mean(result.confidence_scores) if result.confidence_scores else 0.0,
                "verification_vector_length": len(result.verification_vector)
            },
            "metadata": result.metadata,
            "verification_vector": result.verification_vector
        }
        
        if include_details:
            report["detailed_results"] = result.detailed_results
        
        return report
    
    def _generate_batch_result_report(
        self, 
        result: BatchVerificationResult, 
        include_details: bool
    ) -> Dict[str, Any]:
        """Generate report for batch verification result."""
        report = {
            "summary": {
                "num_fingerprint_sets": len(result.individual_results),
                "overall_success_rate": result.overall_success_rate,
                "total_verification_time": result.total_verification_time,
                "combined_vector_length": len(result.combined_verification_vector)
            },
            "metadata": result.metadata,
            "combined_verification_vector": result.combined_verification_vector,
            "individual_summaries": []
        }
        
        for individual_result in result.individual_results:
            summary = {
                "fingerprint_set": individual_result.metadata.get("fingerprint_set_name"),
                "success_rate": individual_result.success_rate,
                "total_fingerprints": individual_result.total_fingerprints,
                "verification_time": individual_result.verification_time
            }
            report["individual_summaries"].append(summary)
        
        if include_details:
            report["individual_detailed_results"] = [
                result.detailed_results for result in result.individual_results
            ]
        
        return report
    
    def get_verification_history(self) -> List[VerificationResult]:
        """Get the history of verification results."""
        return self.verification_history.copy()
    
    def clear_verification_history(self):
        """Clear verification history."""
        self.verification_history.clear()
    
    def delete_model_after_verification(self, model_path: str):
        """Delete model after verification to free up space."""
        import shutil
        if os.path.exists(model_path):
            print(f"Deleting model at {model_path}")
            shutil.rmtree(model_path)
        else:
            print(f"Model path {model_path} does not exist")
    
    def comprehensive_verification(
        self,
        model_path: str,
        fingerprint_files: Union[str, List[str]],
        num_fingerprints: int = 128,
        max_key_length: int = 16,
        max_response_length: int = 1,
        fingerprint_generation_strategy: str = 'english',
        verbose_eval: bool = False,
        wandb_run_name: str = 'None',
        prompt_templates: List[str] = ['{}'],
        sampling_temperature: float = 0.0,
        delete_model: bool = False,
        seed: int = 42,
        use_augmentation_prompts: bool = False
    ) -> Union[VerificationResult, BatchVerificationResult]:
        """
        Comprehensive verification driver similar to eval_driver from check_fingerprints.py.
        
        Args:
            model_path: Path to the model to be verified
            fingerprint_files: Path(s) to fingerprint files
            num_fingerprints: Number of fingerprints to verify
            max_key_length: Maximum key length
            max_response_length: Maximum response length
            fingerprint_generation_strategy: Strategy used for fingerprint generation
            verbose_eval: Whether to print verbose evaluation details
            wandb_run_name: WandB run name for logging
            prompt_templates: List of prompt templates to test
            sampling_temperature: Temperature for sampling
            delete_model: Whether to delete model after verification
            seed: Random seed for reproducibility
            use_augmentation_prompts: Whether to use augmentation prompts
            
        Returns:
            Verification result(s)
        """
        # Load configuration from model directory
        model_config = self.load_config_from_model_dir(model_path)
        
        # Update configuration with loaded values
        config_dict = {
            'fingerprint_generation_strategy': model_config.get('fingerprint_generation_strategy', fingerprint_generation_strategy),
            'max_key_length': model_config.get('max_key_length', max_key_length),
            'max_response_length': model_config.get('max_response_length', max_response_length),
            'num_fingerprints': model_config.get('num_fingerprints', num_fingerprints),
            'model_path': model_path,
            'temperature': sampling_temperature,
            'seed': seed,
            'use_chat_template': model_config.get('use_chat_template', False),
            'remove_eos_token_from_response': model_config.get('remove_eos_token_from_response', False),
            'num_responses_per_fingerprint': model_config.get('num_responses_per_fingerprint', 1)
        }
        
        # Initialize WandB if requested
        if wandb_run_name != 'None':
            self.initialize_wandb(wandb_run_name, config_dict)
        
        # Load augmentation prompts if requested
        if use_augmentation_prompts:
            try:
                with open('generated_data/augmentation_prompts_test.json', 'r') as f:
                    prompt_templates = json.load(f)
                print(f"Loaded {len(prompt_templates)} augmentation prompts")
            except FileNotFoundError:
                print("Warning: augmentation_prompts_test.json not found, using default prompts")
        
        print(f"Testing with {len(prompt_templates)} prompt templates")
        
        # Handle single or multiple fingerprint files
        if isinstance(fingerprint_files, str):
            fingerprint_files = [fingerprint_files]
        
        # Load model
        self.load_model(model_path, config_dict.get('use_chat_template', False))
        
        # Load fingerprint datasets
        for file_path in fingerprint_files:
            # Use file path from config if available
            if 'fingerprints_file_path' in model_config:
                file_path = model_config['fingerprints_file_path']
            self.load_fingerprint_dataset(file_path, min_fingerprints=num_fingerprints)
        
        # Setup output file
        output_file_path = model_path.replace('final_model', 'fingerprinting_output_eval.txt')
        if output_file_path == model_path:
            output_file_path = None
        
        # Perform verification
        if len(fingerprint_files) == 1:
            set_name = list(self.fingerprint_sets.keys())[0]
            result = self.verify_fingerprint_set(
                self.fingerprint_sets[set_name],
                num_samples=num_fingerprints,
                verbose=verbose_eval,
                prompt_templates=prompt_templates,
                temperature=sampling_temperature,
                output_file_path=output_file_path
            )
            
            # Log to WandB
            if self.wandb_run and WANDB_AVAILABLE:
                if len(prompt_templates) > 1:
                    for idx, success_rate in enumerate(result.metadata.get('success_rates_per_template', [])):
                        wandb.log({f'detailed/fingerprint_accuracy_{idx}': success_rate})
                else:
                    wandb.log({'fractional_fingerprint_accuracy': result.fractional_success_rate * 100})
                
                wandb.log({
                    'fingerprint_accuracy': result.success_rate * 100,
                    'verification_time': result.verification_time,
                    'total_fingerprints': result.total_fingerprints
                })
            
            print(f"Fingerprint accuracy: {result.success_rate * 100:.2f}%")
            print(f"Fractional accuracy: {result.fractional_success_rate * 100:.2f}%")
            
        else:
            result = self.verify_all_fingerprint_sets(
                num_samples=num_fingerprints,
                verbose=verbose_eval
            )
            
            # Log to WandB
            if self.wandb_run and WANDB_AVAILABLE:
                wandb.log({
                    'overall_fingerprint_accuracy': result.overall_success_rate * 100,
                    'total_verification_time': result.total_verification_time,
                    'num_fingerprint_sets': len(result.individual_results)
                })
            
            print(f"Overall fingerprint accuracy: {result.overall_success_rate * 100:.2f}%")
        
        # Clean up
        torch.cuda.empty_cache()
        
        if delete_model:
            self.delete_model_after_verification(model_path)
        
        return result
    
    def analyze_verification_trends(self) -> Dict[str, Any]:
        """Analyze trends in verification performance over time."""
        if not self.verification_history:
            return {"error": "No verification history available"}
            
        success_rates = [result.success_rate for result in self.verification_history]
        verification_times = [result.verification_time for result in self.verification_history]
        
        return {
            "num_verifications": len(self.verification_history),
            "success_rate_trend": {
                "current": success_rates[-1],
                "average": np.mean(success_rates),
                "min": min(success_rates),
                "max": max(success_rates),
                "std": np.std(success_rates)
            },
            "performance_trend": {
                "current_time": verification_times[-1],
                "average_time": np.mean(verification_times),
                "min_time": min(verification_times),
                "max_time": max(verification_times),
                "std_time": np.std(verification_times)
            }
        }
    
    def verification_exists(self, model_path: str, fingerprint_file: str, 
                           result_dir: str = "verification_results") -> bool:
        """
        Check if verification results already exist for this model-fingerprint combination.
        
        Args:
            model_path: Path to the model
            fingerprint_file: Path to fingerprint file
            result_dir: Directory where verification results are stored
            
        Returns:
            True if verification results exist, False otherwise
        """
        # Create a unique identifier for this verification
        model_name = Path(model_path).name
        fingerprint_name = Path(fingerprint_file).stem
        result_file = Path(result_dir) / f"verification_{model_name}_{fingerprint_name}.json"
        
        return result_file.exists()
    
    def save_verification_results(self, result: VerificationResult, model_path: str, 
                                fingerprint_file: str, result_dir: str = "verification_results"):
        """
        Save verification results to avoid re-computation.
        
        Args:
            result: Verification result to save
            model_path: Path to the model
            fingerprint_file: Path to fingerprint file
            result_dir: Directory where verification results are stored
        """
        # Create result directory if it doesn't exist
        result_path = Path(result_dir)
        result_path.mkdir(parents=True, exist_ok=True)
        
        # Create unique identifier
        model_name = Path(model_path).name
        fingerprint_name = Path(fingerprint_file).stem
        result_file = result_path / f"verification_{model_name}_{fingerprint_name}.json"
        
        # Save comprehensive result data
        result_data = {
            "model_path": model_path,
            "fingerprint_file": fingerprint_file,
            "timestamp": time.time(),
            "result": {
                "total_fingerprints": result.total_fingerprints,
                "successful_verifications": result.successful_verifications,
                "failed_verifications": result.failed_verifications,
                "success_rate": result.success_rate,
                "verification_time": result.verification_time,
                "verification_vector": result.verification_vector,
                "confidence_scores": result.confidence_scores,
                "metadata": result.metadata
            }
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"Verification results saved to: {result_file}")
    
    def load_verification_results(self, model_path: str, fingerprint_file: str, 
                                result_dir: str = "verification_results") -> Optional[VerificationResult]:
        """
        Load existing verification results.
        
        Args:
            model_path: Path to the model
            fingerprint_file: Path to fingerprint file
            result_dir: Directory where verification results are stored
            
        Returns:
            VerificationResult if found, None otherwise
        """
        model_name = Path(model_path).name
        fingerprint_name = Path(fingerprint_file).stem
        result_file = Path(result_dir) / f"verification_{model_name}_{fingerprint_name}.json"
        
        if not result_file.exists():
            return None
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            result_data = data["result"]
            return VerificationResult(
                total_fingerprints=result_data["total_fingerprints"],
                successful_verifications=result_data["successful_verifications"],
                failed_verifications=result_data["failed_verifications"],
                success_rate=result_data["success_rate"],
                verification_time=result_data["verification_time"],
                verification_vector=result_data["verification_vector"],
                confidence_scores=result_data["confidence_scores"],
                detailed_results=[],  # Not saved for space efficiency
                metadata=result_data["metadata"]
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading verification results from {result_file}: {e}")
            return None


# Convenience functions
def verify_model_with_fingerprints(
    model_path: str,
    fingerprint_files: Union[str, List[str]],
    config: Optional[FingerprintConfig] = None,
    use_chat_template: bool = False,
    num_samples: Optional[int] = None,
    verbose: bool = False,
    min_fingerprints: Optional[int] = None,
    use_cache: bool = False,
    cache_dir: str = "verification_results"
) -> Union[VerificationResult, BatchVerificationResult]:
    """
    Convenience function to verify a model with fingerprints.
    
    Args:
        model_path: Path to the model directory
        fingerprint_files: Path(s) to fingerprint JSON files
        config: FingerprintConfig (will use default if None)
        use_chat_template: Whether to use chat template for instruction models
        num_samples: Number of samples to test per file
        verbose: Whether to print detailed results
        min_fingerprints: Optional minimum number of fingerprints required per file
        use_cache: Whether to use cached results if available
        cache_dir: Directory for caching verification results
        
    Returns:
        Verification result(s)
    """
    if config is None:
        from .base import create_simple_config, VerificationMode
        config = create_simple_config(verification_mode=VerificationMode.PREFIX_MATCH)
    
    engine = VerificationEngine(config)
    return engine.verify_from_files(
        model_path=model_path,
        fingerprint_files=fingerprint_files,
        use_chat_template=use_chat_template,
        num_samples=num_samples,
        verbose=verbose,
        min_fingerprints=min_fingerprints,
        use_cache=use_cache,
        cache_dir=cache_dir
    )


def get_verification_vector(
    model_path: str,
    fingerprint_file: str,
    use_chat_template: bool = False,
    verification_mode: VerificationMode = VerificationMode.PREFIX_MATCH,
    verbose: bool = False,
    min_fingerprints: Optional[int] = None,
    use_cache: bool = False,
    cache_dir: str = "verification_results"
) -> List[bool]:
    """
    Simple function to get verification vector for a model and fingerprint set.
    
    Args:
        model_path: Path to the model directory
        fingerprint_file: Path to fingerprint JSON file
        use_chat_template: Whether to use chat template
        verification_mode: Verification mode to use
        verbose: Whether to print progress
        min_fingerprints: Optional minimum number of fingerprints required
        use_cache: Whether to use cached results if available
        cache_dir: Directory for caching verification results
        
    Returns:
        Boolean list representing verification results (V vector)
    """
    from .base import create_simple_config
    
    config = create_simple_config(verification_mode=verification_mode)
    engine = VerificationEngine(config)
    result = engine.verify_from_files(
        model_path=model_path,
        fingerprint_files=fingerprint_file,
        use_chat_template=use_chat_template,
        verbose=verbose,
        min_fingerprints=min_fingerprints,
        use_cache=use_cache,
        cache_dir=cache_dir
    )
    
    return result.verification_vector 