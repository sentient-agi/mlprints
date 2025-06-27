"""
Fingerprint Generation Engine

This module provides classes and functions for generating various types of fingerprints
for model verification, focusing on creating FingerprintSet objects with proper abstraction.

The InverseNucleusFingerprintGenerator implements the method from Nasery et al. (2025):
- Key generation: Sample from LLM with prompt "Generate a sentence starting with <word>"
  using a word from 10,000 most-used English words, temperature 0.5, length 16 tokens
- Response generation: Perinucleus sampling with threshold 0.8, k=3, typically 1 token
"""

import json
import logging
import random
import re
import os
from typing import List, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import transformers
import numpy as np
from tqdm import tqdm

# Suppress verbose HTTP request logging from OpenAI client
logging.getLogger("openai._client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from .base import VerificationType
from .fingerprints import (
    FingerprintSet,
    SimpleFingerprint, 
    TokenExistenceFingerprint, 
    RegexFingerprint,
    Fingerprint
)
from ..common.inference_utils import VLLMInference


@dataclass
class GenerationConfig:
    """Configuration for fingerprint generation."""
    num_fingerprints: int = 128
    key_length: int = 16
    response_length: int = 1  # Nasery et al. typically use 1 token responses
    temperature: float = 0.5  # Nasery et al. use 0.5 for key generation
    batch_size: int = 1024  # Large batches for vLLM efficiency
    seed: int = 42
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gpu: Union[str, List[int]] = "0"
    predefined_keys: Optional[List[str]] = None
    keys_path: Optional[str] = None
    word_list_path: str = "data/common/word_list.txt"
    
    # Inverse nucleus specific (Nasery et al. defaults)
    nucleus_threshold: float = 0.8  # Nasery et al. use 0.8
    nucleus_k: int = 3  # Nasery et al. use 3
    use_chat_template: bool = False
    
    # Diversity settings
    use_prompt_variations: bool = True  # Use varied prompt templates to increase diversity


class WordListManager:
    """Word list manager for generating random text. 
    
    Should contain the 10,000 most-used English words as specified in Nasery et al.
    """
    
    def __init__(self, word_list_path: str = "data/common/word_list.txt"):
        self.word_list_path = word_list_path
        self.word_list = self._load_word_list()
    
    def _load_word_list(self) -> List[str]:
        """Load word list from file."""
        try:
            with open(self.word_list_path, 'r') as f:
                word_list = [line.strip() for line in f.readlines() if line.strip()]
                
            # Validate word list size matches Nasery et al. specification
            if len(word_list) < 5000:
                print(f"Warning: Word list contains only {len(word_list)} words. "
                      f"Nasery et al. specify using 10,000 most-used English words.")
            elif len(word_list) != 10000:
                print(f"Info: Word list contains {len(word_list)} words "
                      f"(Nasery et al. used 10,000).")
                
            return word_list
        except FileNotFoundError:
            print(f"Warning: {self.word_list_path} not found. Using fallback word list.")
            return ["hello", "world", "test", "sample", "random", "word", "example", "data"]
    
    def get_random_words(self, count: int) -> List[str]:
        """Get a list of random words."""
        return [random.choice(self.word_list) for _ in range(count)]
    
    def get_random_words_without_replacement(self, count: int) -> List[str]:
        """Get a list of random words without replacement.
        
        If count > len(word_list), cycles through the list multiple times
        but ensures maximum diversity within each cycle.
        """
        if count <= len(self.word_list):
            return random.sample(self.word_list, count)
        else:
            # Need more words than available - cycle through multiple times
            result = []
            remaining = count
            
            while remaining > 0:
                if remaining >= len(self.word_list):
                    # Take all words in random order
                    shuffled_words = self.word_list.copy()
                    random.shuffle(shuffled_words)
                    result.extend(shuffled_words)
                    remaining -= len(self.word_list)
                else:
                    # Take remaining words without replacement
                    result.extend(random.sample(self.word_list, remaining))
                    remaining = 0
            
            return result
    
    def get_random_sentence(self, word_count: int) -> str:
        """Get a random sentence with specified word count."""
        words = self.get_random_words(word_count)
        return ' '.join(words)


class TextGenerator:
    """Key generator using vLLM for fast batch generation of sentences.
    
    This is ONLY for generating the keys (sentences starting with words).
    Response generation uses transformers for direct logit access.
    """
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.llm_client = None
        
    def _get_tensor_parallel_size(self) -> int:
        """Calculate tensor parallel size based on GPU configuration."""
        if isinstance(self.config.gpu, list):
            return len(self.config.gpu)
        elif isinstance(self.config.gpu, str) and ',' in self.config.gpu:
            return len(self.config.gpu.split(','))
        else:
            return 1  # Single GPU
        
    def __enter__(self):
        """Context manager entry."""
        # Properly configure VLLMInference with all the right parameters
        self.llm_client = VLLMInference(
            model=self.config.model_name,
            gpu=self.config.gpu,
            verbose=True,
            timeout=1200,  # Much longer timeout for large models and batch processing
            server_kwargs={
                "dtype": "auto",
                "max-model-len": 4096,
                "tensor-parallel-size": self._get_tensor_parallel_size(),
                "enforce-eager": True,  # Faster startup
                "disable-log-requests": True,  # Less logging overhead
                "enable-chunked-prefill": True,  # Better batching
                "max-num-seqs": 2048,  # Support large batches
            }
        ).__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.llm_client:
            self.llm_client.__exit__(exc_type, exc_val, exc_tb)
    
    def generate_text(self, prompt: str, max_length: int, temperature: Optional[float] = None) -> str:
        """Generate text for a single prompt using vLLM.
        
        Args:
            prompt: Input prompt
            max_length: Maximum tokens to generate
            temperature: Override temperature (if None, uses config temperature)
        """
        effective_temperature = temperature if temperature is not None else self.config.temperature
        
        if not self.llm_client:
            raise RuntimeError("vLLM client not initialized")
            
        result = self.llm_client.complete(
            prompt=prompt,
            temperature=effective_temperature,
            max_tokens=max_length,
        )
        return result.strip()
    
    def generate_batch(self, prompts: List[str], max_length: int, temperature: Optional[float] = None) -> List[str]:
        """Generate text for a batch of prompts using vLLM's native batching."""
        effective_temperature = temperature if temperature is not None else self.config.temperature
        
        if not self.llm_client:
            raise RuntimeError("vLLM client not initialized")
        
        # Use vLLM's native batch processing - single API call for all prompts!
        try:
            response = self.llm_client.client.completions.create(
                model=self.llm_client.served_model_name,
                prompt=prompts,  # vLLM handles multiple prompts in one request
                temperature=effective_temperature,
                max_tokens=max_length,
            )
            
            # Extract results - should be in same order as input prompts
            results = [choice.text.strip() for choice in response.choices]
            
        except Exception as e:
            print(f"Warning: Native batch failed ({e}), trying smaller chunks...")
            # Fallback: split into smaller chunks if the batch is too large
            chunk_size = min(64, len(prompts) // 2)  # Try smaller chunks
            results = []
            
            for i in range(0, len(prompts), chunk_size):
                chunk = prompts[i:i + chunk_size]
                try:
                    response = self.llm_client.client.completions.create(
                        model=self.llm_client.served_model_name,
                        prompt=chunk,
                        temperature=effective_temperature,
                        max_tokens=max_length,
                    )
                    chunk_results = [choice.text.strip() for choice in response.choices]
                    results.extend(chunk_results)
                except Exception as chunk_error:
                    print(f"Chunk failed: {chunk_error}")
                    # Ultimate fallback - individual requests for this chunk
                    for prompt in chunk:
                        try:
                            result = self.llm_client.complete(
                                prompt=prompt,
                                temperature=effective_temperature,
                                max_tokens=max_length,
                            )
                            results.append(result.strip())
                        except:
                            results.append("")  # Empty on failure
                            
        return results


class InverseNucleusGenerator:
    """Response generator using TRANSFORMERS for inverse nucleus sampling.
    
    This is the TRANSFORMERS part that requires direct logit access (not vLLM).
    
    Implements the Perinucleus sampling algorithm from Nasery et al.:
    1. Compute next-token probabilities
    2. Sort tokens by probability and build CDF
    3. Find first index where CDF >= threshold (default 0.8)
    4. Skip top token and uniformly sample from next k tokens (default k=3)
    5. For multi-token responses, use greedy decoding after first token
    
    Note: For faithful reproduction of the paper, typically use response_length=1.
    """
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def __enter__(self):
        """Context manager entry."""
        self._initialize_model()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup GPU memory."""
        if self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _initialize_model(self):
        """Initialize the model for inverse nucleus sampling."""
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.model_name)
        # Properly set pad token to avoid warnings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def _inverse_nucleus_sample(self, logits: torch.Tensor) -> int:
        """Perform inverse nucleus sampling on logits following Nasery et al. algorithm.
        
        Algorithm from Nasery et al. (2025):
        1. Sort tokens by probability and build CDF
        2. Find smallest index i where CDF >= threshold (default 0.8) 
        3. Uniformly pick r ∈ {1, ..., k} (1-indexed!)
        4. Select token at rank i+r
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        
        # Step 2: Find the smallest index i where CDF >= threshold
        nucleus_end_indices = torch.where(cumulative_probs >= self.config.nucleus_threshold)[0]
        if len(nucleus_end_indices) == 0:
            raise ValueError("No valid tokens found for nucleus sampling.")
        
        nucleus_end_idx = nucleus_end_indices[0].item()  # First index where CDF >= threshold
        
        # Step 3-4: Uniformly pick r ∈ {1, ..., k} and select token at rank i+r
        k = self.config.nucleus_k
        
        # Loop to keep increasing k until an alphanumeric token is found
        while True:
            # Uniformly sample r from {1, ..., k} (1-indexed as per paper)
            r = torch.randint(1, k + 1, (1,)).item()
            
            # Select token at rank i+r (0-indexed in our array)
            target_idx = nucleus_end_idx + r
            
            if target_idx >= len(sorted_indices):
                k += 1  # Expand k if we're out of bounds
                if k > len(sorted_indices) - nucleus_end_idx:
                    raise ValueError("No valid token found - ran out of tokens.")
                continue
            
            candidate_token = sorted_indices[target_idx]
            
            # Decode the token and check if it's alphanumeric
            decoded_token = self.tokenizer.decode([candidate_token]).strip()
            if re.match(r'^[a-zA-Z0-9]+$', decoded_token):
                return candidate_token.item()
            else:
                k += 1  # Expand search range if token isn't alphanumeric
                if k > len(sorted_indices) - nucleus_end_idx:
                    raise ValueError("No valid alphanumeric token found.")
    
    def generate_response(self, key: str) -> str:
        """Generate a response for a single key using inverse nucleus sampling."""
        # Apply chat template if requested
        if self.config.use_chat_template:
            # Apply chat template to the key
            conversation = [{"role": "user", "content": key}]
            formatted_key = self.tokenizer.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            key_tokens = self.tokenizer.encode(formatted_key, add_special_tokens=False)
            # Remove trailing EOS token if present
            if len(key_tokens) > 0 and key_tokens[-1] == self.tokenizer.eos_token_id:
                key_tokens = key_tokens[:-1]
        else:
            # Tokenize key normally
            key_tokens = self.tokenizer.encode(key, add_special_tokens=False)
            
        if len(key_tokens) > self.config.key_length:
            key_tokens = key_tokens[:self.config.key_length]
        
        input_ids = torch.tensor([key_tokens]).cuda()
        
        # Generate response tokens
        response_tokens = []
        current_input = input_ids
        
        for i in range(self.config.response_length):
            with torch.no_grad():
                outputs = self.model(current_input)
                next_token_logits = outputs.logits[0, -1, :]
            
            # Use inverse nucleus sampling for first token, then greedy
            if i == 0:
                next_token = self._inverse_nucleus_sample(next_token_logits)
            else:
                next_token = torch.argmax(next_token_logits).item()
            
            response_tokens.append(next_token)
            current_input = torch.cat([current_input, torch.tensor([[next_token]]).cuda()], dim=1)
        
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True)


class FingerprintGenerator(ABC):
    """Abstract base class for fingerprint generators."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.word_manager = WordListManager(config.word_list_path)
        self._set_seeds()
        
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        np.random.seed(self.config.seed)
    
    def _load_predefined_keys(self) -> List[str]:
        """Load predefined keys from configuration."""
        if self.config.predefined_keys:
            return self.config.predefined_keys[:self.config.num_fingerprints]
        
        if self.config.keys_path:
            try:
                with open(self.config.keys_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        if isinstance(data[0], str):
                            return data[:self.config.num_fingerprints]
                        else:
                            return [item.get('key', str(item)) for item in data[:self.config.num_fingerprints]]
                    else:
                        raise ValueError("Keys file should contain a list")
            except Exception as e:
                print(f"Warning: Failed to load keys from {self.config.keys_path}: {e}")
        
        return []
    
    @abstractmethod
    def generate_fingerprint_set(self) -> FingerprintSet:
        """Generate a set of fingerprints."""
        pass
    
    def save_to_file(self, output_path: str) -> str:
        """Save generated fingerprints to file."""
        fingerprint_set = self.generate_fingerprint_set()
        
        if not output_path.endswith('.json'):
            output_path = f"{output_path}.json"
            
        fingerprint_set.save_to_file(output_path)
        return output_path


class SimpleTextGenerator(FingerprintGenerator):
    """Generates simple fingerprints using language model text generation."""
    
    def generate_fingerprint_set(self) -> FingerprintSet:
        """Generate a set of simple fingerprints."""
        fingerprints = []
        predefined_keys = self._load_predefined_keys()
        
        with TextGenerator(self.config) as text_gen:
            for batch_start in tqdm(range(0, self.config.num_fingerprints, self.config.batch_size), 
                                   desc="Generating simple fingerprints"):
                batch_size = min(self.config.batch_size, self.config.num_fingerprints - batch_start)
                
                # Generate keys following Nasery et al. method
                if predefined_keys:
                    key_prompts = predefined_keys[batch_start:batch_start + batch_size]
                else:
                    # Use the exact prompt format from Nasery et al. with sampling without replacement
                    words_for_batch = self.word_manager.get_random_words_without_replacement(batch_size)
                    key_prompts = [f"Generate a sentence starting with {word}" 
                                  for word in words_for_batch]
                
                keys = text_gen.generate_batch(key_prompts, self.config.key_length)
                
                # Generate responses
                response_prompts = [f"{random.choice(self.word_manager.word_list)} " 
                                   for _ in range(batch_size)]
                responses = text_gen.generate_batch(response_prompts, self.config.response_length)
                
                # Create SimpleFingerprint objects
                for key, response in zip(keys, responses):
                    fingerprints.append(SimpleFingerprint(
                        query=key.strip(),
                        expected_response=response.strip()
                    ))
                    
                    if len(fingerprints) >= self.config.num_fingerprints:
                        break
        
        return FingerprintSet(fingerprints[:self.config.num_fingerprints], name="simple_text_generated")


class RandomWordGenerator(FingerprintGenerator):
    """Generates simple fingerprints using random words."""
    
    def generate_fingerprint_set(self) -> FingerprintSet:
        """Generate a set of simple fingerprints with random words."""
        fingerprints = []
        
        for _ in range(self.config.num_fingerprints):
            key = self.word_manager.get_random_sentence(self.config.key_length)
            response = self.word_manager.get_random_sentence(self.config.response_length)
            fingerprints.append(SimpleFingerprint(query=key, expected_response=response))
        
        return FingerprintSet(fingerprints, name="random_words_generated")


class TokenExistenceGenerator(FingerprintGenerator):
    """Generates token existence fingerprints."""
    
    def __init__(self, config: GenerationConfig, 
                 base_generator: Optional[FingerprintGenerator] = None,
                 num_tokens_per_response: int = 3,
                 case_sensitive: bool = True):
        super().__init__(config)
        self.base_generator = base_generator or RandomWordGenerator(config)
        self.num_tokens_per_response = num_tokens_per_response
        self.case_sensitive = case_sensitive
    
    def generate_fingerprint_set(self) -> FingerprintSet:
        """Generate a set of token existence fingerprints."""
        base_set = self.base_generator.generate_fingerprint_set()
        fingerprints = []
        
        for base_fp in base_set.fingerprints:
            # Extract response text and query from base fingerprint
            if base_fp.verification_functions:
                vf = base_fp.verification_functions[0]
                if hasattr(vf, 'expected_response'):
                    response_text = vf.expected_response
                    query = base_fp.get_query()
                    
                    # Extract tokens from response
                    response_words = response_text.split()
                    if len(response_words) >= self.num_tokens_per_response:
                        required_tokens = random.sample(response_words, self.num_tokens_per_response)
                    else:
                        required_tokens = response_words
                    
                    if required_tokens:
                        fingerprints.append(TokenExistenceFingerprint(
                            query=query,
                            required_tokens=required_tokens,
                            case_sensitive=self.case_sensitive
                        ))
        
        return FingerprintSet(fingerprints, name="token_existence_generated")


class RegexGenerator(FingerprintGenerator):
    """Generates regex fingerprints."""
    
    def __init__(self, config: GenerationConfig,
                 base_generator: Optional[FingerprintGenerator] = None,
                 pattern_templates: Optional[List[str]] = None):
        super().__init__(config)
        self.base_generator = base_generator or RandomWordGenerator(config)
        self.pattern_templates = pattern_templates or [
            r'\b\w+\b',      # Match whole words
            r'\d+',          # Match numbers
            r'[A-Z]\w*',     # Match capitalized words
            r'\w*ing\b',     # Match words ending in 'ing'
            r'\w{3,}',       # Match words with 3+ characters
        ]
    
    def generate_fingerprint_set(self) -> FingerprintSet:
        """Generate a set of regex fingerprints."""
        base_set = self.base_generator.generate_fingerprint_set()
        fingerprints = []
        
        for base_fp in base_set.fingerprints:
            query = base_fp.get_query()
            pattern = random.choice(self.pattern_templates)
            fingerprints.append(RegexFingerprint(query=query, pattern=pattern))
        
        return FingerprintSet(fingerprints, name="regex_generated")


class InverseNucleusFingerprintGenerator(FingerprintGenerator):
    """Generates fingerprints using two-stage process:
    
    1. KEY GENERATION (vLLM): Fast batch generation of 8192 sentences using Instruct model
    2. RESPONSE GENERATION (Transformers): Inverse nucleus sampling requiring direct logit access
    
    This leverages vLLM's throughput for key generation while using transformers 
    for precise response generation with the Nasery et al. algorithm.
    """
    
    def generate_fingerprint_set(self) -> FingerprintSet:
        """Generate a set of simple fingerprints using inverse nucleus sampling."""
        # Prepare base keys
        base_keys = self._load_predefined_keys()
        if not base_keys:
            # Generate base keys using the LLM following Nasery et al. method
            print("Generating keys using LLM with Nasery et al. method...")
            print(f"Using sampling without replacement from {len(self.word_manager.word_list)} words to avoid duplicates...")
            
            # Get words without replacement to maximize diversity
            words_for_prompts = self.word_manager.get_random_words_without_replacement(self.config.num_fingerprints)
            
            # ==================================================================
            # STAGE 1: KEY GENERATION (vLLM + Batch Processing)
            # ==================================================================
            with TextGenerator(self.config) as text_gen:
                print("🚀 Stage 1: Generating keys with vLLM batch processing...")
                
                if self.config.use_prompt_variations:
                    # Add slight variations to prompts to increase diversity while maintaining Nasery et al. format
                    prompt_variations = [
                        "Generate a sentence starting with {word}",
                        "Write a sentence that begins with {word}",
                        "Create a sentence starting with the word {word}",
                        "Generate a sentence beginning with {word}",
                    ]
                    
                    key_prompts = []
                    for i, word in enumerate(words_for_prompts):
                        # Cycle through prompt variations to add diversity
                        template = prompt_variations[i % len(prompt_variations)]
                        key_prompts.append(template.format(word=word))
                else:
                    # Use the exact Nasery et al. format
                    key_prompts = [f"Generate a sentence starting with {word}" 
                                  for word in words_for_prompts]
                
                # Use configured batch size, but ensure reasonable limits
                batch_size = min(self.config.batch_size, len(key_prompts))  # Respect config, but not larger than total
                base_keys = []
                failed_validations = 0
                total_prompts_processed = 0
                
                # Progress bar at the PROMPT level (not batch level)
                with tqdm(total=len(key_prompts), desc="🔑 Generating keys", unit="prompt") as pbar:
                    for batch_start in range(0, len(key_prompts), batch_size):
                        batch_end = min(batch_start + batch_size, len(key_prompts))
                        batch_prompts = key_prompts[batch_start:batch_end]
                        batch_words = words_for_prompts[batch_start:batch_end]
                        
                        # Generate batch
                        raw_keys_batch = text_gen.generate_batch(batch_prompts, self.config.key_length)
                        
                        # Validate batch and collect failures for efficient resampling
                        batch_successes = []
                        batch_failures = []
                        
                        for i, (raw_key, expected_word) in enumerate(zip(raw_keys_batch, batch_words)):
                            key = raw_key.strip()
                            
                            if self._validate_key_starts_with_word(key, expected_word):
                                batch_successes.append(key)
                            else:
                                # Collect failures for batch resampling
                                batch_failures.append({
                                    'word': expected_word,
                                    'original_prompt': batch_prompts[i],
                                    'failed_key': key
                                })
                        
                        # Add successful keys
                        base_keys.extend(batch_successes)
                        
                        # Efficient batch resampling for all failures at once
                        if batch_failures:
                            resampled_keys = self._batch_resample_failures(text_gen, batch_failures)
                            base_keys.extend(resampled_keys)
                            failed_validations += len(batch_failures) - len(resampled_keys)
                        
                        # Update progress bar at prompt level
                        batch_processed = len(batch_prompts)
                        total_prompts_processed += batch_processed
                        pbar.update(batch_processed)
                        pbar.set_postfix_str(f"✓ {len(base_keys)}/{total_prompts_processed} valid, {failed_validations} failed")
                
                if failed_validations > 0:
                    print(f"⚠️  {failed_validations} keys failed validation and were skipped")
                
                print(f"✅ Generated {len(base_keys)} validated keys from {len(words_for_prompts)} prompts")
        
        # ==================================================================
        # STAGE 2: RESPONSE GENERATION (Transformers + Direct Logit Access)
        # ==================================================================
        print(f"🧬 Stage 2: Generating responses with inverse nucleus sampling...")
        print(f"📊 Processing {len(base_keys)} keys using transformers for logit access...")
        
        fingerprints = []
        seen_queries = set()  # Track queries to detect duplicates before adding to FingerprintSet
        skipped_duplicates = 0
        
        with InverseNucleusGenerator(self.config) as nucleus_generator:
            with tqdm(base_keys, desc="🧬 Inverse nucleus responses", unit="response") as pbar:
                for key in pbar:
                    try:
                        # Skip duplicate keys at generation level
                        if key in seen_queries:
                            skipped_duplicates += 1
                            pbar.set_postfix_str(f"✓ {len(fingerprints)} fingerprints, {skipped_duplicates} duplicates")
                            continue
                        
                        response = nucleus_generator.generate_response(key)
                        fingerprints.append(SimpleFingerprint(query=key, expected_response=response))
                        seen_queries.add(key)
                        
                        # Update progress bar with more detailed info
                        pbar.set_postfix_str(f"✓ {len(fingerprints)} fingerprints, {skipped_duplicates} duplicates")
                        
                    except Exception as e:
                        print(f"Warning: Failed to generate response for key '{key}': {e}")
                        pbar.set_postfix_str(f"✓ {len(fingerprints)} fingerprints, ❌ 1 failed")
        
        if skipped_duplicates > 0:
            print(f"⏭️  Skipped {skipped_duplicates} duplicate keys during generation")
        
        print(f"🎉 Generated {len(fingerprints)} unique fingerprints from {len(base_keys)} base keys")
        return FingerprintSet(fingerprints, name="inverse_nucleus_generated")
    
    def _validate_key_starts_with_word(self, key: str, expected_word: str) -> bool:
        """Validate that the generated key contains the expected word near the beginning.
        
        More lenient validation to reduce resampling overhead.
        """
        import re
        
        if not key or len(key.strip()) < 3:
            return False
            
        # Very lenient check - just see if the expected word appears in the first 50 characters
        key_start = key.lower()[:50]  # Check first 50 chars only
        expected_lower = expected_word.lower()
        
        # Simple substring check - much more lenient than regex word boundaries
        return expected_lower in key_start
    
    def _resample_key_with_validation(self, text_gen, expected_word: str, original_prompt: str, max_retries: int = 10) -> Optional[str]:
        """Resample a key until it validates or max retries reached.
        
        Progressively increases temperature with each retry to add more randomness.
        """
        base_temperature = self.config.temperature
        
        for attempt in range(max_retries):
            try:
                # Increase temperature by 0.1 for each retry attempt to add randomness
                current_temperature = min(base_temperature + (attempt * 0.1), 1.1)  # Cap at 1.1
                
                # Try different prompt variations to increase chances of compliance
                if self.config.use_prompt_variations:
                    alternative_prompts = [
                        f"Write a sentence that starts with the word {expected_word}",
                        f"Create a sentence beginning with {expected_word}",
                        f"Start a sentence with {expected_word}",
                        f"{expected_word}",  # Just the word itself
                        f"Complete this sentence: {expected_word}",
                        f"Continue: {expected_word}",
                        f"Sentence starting with '{expected_word}':",
                        f"{expected_word.capitalize()}",  # Capitalized version
                    ]
                    prompt = alternative_prompts[attempt % len(alternative_prompts)]
                else:
                    # Even without variations, try some basic alternatives
                    alternatives = [original_prompt, f"{expected_word}", f"Continue: {expected_word}"]
                    prompt = alternatives[attempt % len(alternatives)]
                
                # Generate with increased temperature for this attempt
                resampled_key = text_gen.generate_text(prompt, self.config.key_length, temperature=current_temperature).strip()
                
                if self._validate_key_starts_with_word(resampled_key, expected_word):
                    if attempt > 0:  # Only print if it took more than one attempt
                        print(f"✓ Resampled '{expected_word}' with T={current_temperature:.1f} on attempt {attempt + 1}")
                    return resampled_key
                    
            except Exception as e:
                print(f"Error during resampling attempt {attempt + 1}: {e}")
                continue
        
        return None  # Failed after all retries
    
    def _batch_resample_failures(self, text_gen, failures: List[dict]) -> List[str]:
        """Efficiently resample all failures in batches with progressive strategies.
        
        Args:
            text_gen: TextGenerator instance
            failures: List of dicts with 'word', 'original_prompt', 'failed_key'
            
        Returns:
            List of successfully resampled keys
        """
        if not failures:
            return []
        
        successful_resamples = []
        remaining_failures = failures.copy()
        
        # Strategy 1: Try higher temperature (0.7) with original prompts
        if remaining_failures:
            retry_prompts = [f['original_prompt'] for f in remaining_failures]
            try:
                resampled_batch = []
                for prompt in retry_prompts:
                    result = text_gen.generate_text(prompt, self.config.key_length, temperature=0.7)
                    resampled_batch.append(result)
                
                # Validate results
                new_remaining = []
                for i, (result, failure) in enumerate(zip(resampled_batch, remaining_failures)):
                    if self._validate_key_starts_with_word(result.strip(), failure['word']):
                        successful_resamples.append(result.strip())
                    else:
                        new_remaining.append(failure)
                remaining_failures = new_remaining
                
            except Exception as e:
                print(f"Batch resampling strategy 1 failed: {e}")
        
        # Strategy 2: Try alternative prompt templates with higher temperature
        if remaining_failures:
            alternative_prompts = []
            for failure in remaining_failures:
                word = failure['word']
                # Try different prompt templates
                templates = [
                    f"Write a sentence that starts with {word}",
                    f"Create a sentence beginning with {word}",
                    f"{word.capitalize()}",  # Just the capitalized word
                    f"Complete this: {word}",
                ]
                # Pick a different template than the original
                alternative_prompts.append(templates[0])  # Use first alternative
            
            try:
                resampled_batch = []
                for prompt in alternative_prompts:
                    result = text_gen.generate_text(prompt, self.config.key_length, temperature=0.9)
                    resampled_batch.append(result)
                
                # Validate results
                new_remaining = []
                for i, (result, failure) in enumerate(zip(resampled_batch, remaining_failures)):
                    if self._validate_key_starts_with_word(result.strip(), failure['word']):
                        successful_resamples.append(result.strip())
                    else:
                        new_remaining.append(failure)
                remaining_failures = new_remaining
                
            except Exception as e:
                print(f"Batch resampling strategy 2 failed: {e}")
        
        # Strategy 3: Last resort - just use the word itself with high temperature
        if remaining_failures:
            simple_prompts = [f['word'] for f in remaining_failures]
            try:
                resampled_batch = []
                for prompt in simple_prompts:
                    result = text_gen.generate_text(prompt, self.config.key_length, temperature=1.1)
                    resampled_batch.append(result)
                
                # More lenient validation for last resort
                for i, (result, failure) in enumerate(zip(resampled_batch, remaining_failures)):
                    result_clean = result.strip()
                    if len(result_clean) > 3:  # Very lenient - just needs to be a reasonable string
                        successful_resamples.append(result_clean)
                        
            except Exception as e:
                print(f"Batch resampling strategy 3 failed: {e}")
        
        return successful_resamples


def create_generator(
    generator_type: str,
    config: GenerationConfig,
    **kwargs
) -> FingerprintGenerator:
    """
    Factory function to create fingerprint generators.
    
    Args:
        generator_type: Type of generator 
        config: Generation configuration
        **kwargs: Additional arguments for specific generators
        
    Returns:
        Appropriate FingerprintGenerator instance
    """
    generators = {
        "simple_text": SimpleTextGenerator,
        "english": SimpleTextGenerator,  # Backward compatibility
        "random_word": RandomWordGenerator,
        "token_existence": TokenExistenceGenerator,
        "regex": RegexGenerator,
        "inverse_nucleus": InverseNucleusFingerprintGenerator,
    }
    
    if generator_type not in generators:
        raise ValueError(f"Unknown generator type: {generator_type}. Available: {list(generators.keys())}")
    
    return generators[generator_type](config, **kwargs)


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate fingerprint sets')
    parser.add_argument('--generator_type', type=str, default='simple_text', 
                       choices=['simple_text', 'english', 'random_word', 'token_existence', 'regex', 'inverse_nucleus'],
                       help='Type of fingerprint generator to use')
    parser.add_argument('--num_fingerprints', type=int, default=128, 
                       help='Number of fingerprints to generate')
    parser.add_argument('--key_length', type=int, default=16, help='Length of the key (Nasery et al. use 16)')
    parser.add_argument('--response_length', type=int, default=1, help='Length of the response (Nasery et al. typically use 1)')
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', 
                       help='Model for generation')
    parser.add_argument('--output_path', type=str, 
                       default='data/simple_fingerprints/generated_fingerprints.json', 
                       help='Path to store the generated fingerprints')
    parser.add_argument('--keys_path', type=str, default=None, help='Path to predefined keys file')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for generation (Nasery et al. use 0.5)')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for generation (larger = better vLLM utilization)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--word_list_path', type=str, default='data/common/word_list.txt', 
                       help='Path to word list file (should contain 10,000 most-used English words)')
    
    # GPU and inference specific
    parser.add_argument('--gpu', type=str, default='0', help='GPU device(s) to use (e.g., "0" or "0,1,2,3")')
    
    # Inverse nucleus specific
    parser.add_argument('--nucleus_threshold', type=float, default=0.8, 
                       help='Nucleus threshold for inverse nucleus sampling (Nasery et al. use 0.8)')
    parser.add_argument('--nucleus_k', type=int, default=3, 
                       help='Nucleus k for inverse nucleus sampling (Nasery et al. use 3)')
    parser.add_argument('--use_chat_template', action='store_true', 
                       help='Apply chat template to keys for chat models')
    
    # Token existence specific
    parser.add_argument('--num_tokens_per_response', type=int, default=3, 
                       help='Number of tokens per response for token existence')
    parser.add_argument('--case_sensitive', action='store_true', 
                       help='Case sensitive token matching')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Create configuration
    config = GenerationConfig(
        num_fingerprints=args.num_fingerprints,
        key_length=args.key_length,
        response_length=args.response_length,
        model_name=args.model_name,
        keys_path=args.keys_path,
        temperature=args.temperature,
        batch_size=args.batch_size,
        seed=args.seed,
        gpu=args.gpu,
        nucleus_threshold=args.nucleus_threshold,
        nucleus_k=args.nucleus_k,
        use_chat_template=args.use_chat_template,
        word_list_path=args.word_list_path
    )
    
    # Create generator based on type
    generator_kwargs = {}
    if args.generator_type == 'token_existence':
        generator_kwargs['num_tokens_per_response'] = args.num_tokens_per_response
        generator_kwargs['case_sensitive'] = args.case_sensitive
    
    generator = create_generator(args.generator_type, config, **generator_kwargs)
    
    # Generate and save fingerprints
    output_path = generator.save_to_file(args.output_path)
    print(f"Successfully generated {args.num_fingerprints} {args.generator_type} fingerprints to {output_path}")
