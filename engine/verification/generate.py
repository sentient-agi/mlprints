"""
Fingerprint Generation Engine

This module provides classes and functions for generating various types of fingerprints
for model verification, abstracting the core generation logic into reusable components.
"""

import json
import random
import re
import tempfile
import os
from typing import List, Dict, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
import numpy as np
from tqdm import tqdm

from .base import VerificationType
from .fingerprints import (
    FingerprintSet, SimpleFingerprintSet, TokenFingerprintSet, RegexFingerprintSet,
    SimpleFingerprint, TokenExistenceFingerprint, RegexFingerprint,
    Fingerprint
)


@dataclass
class GenerationConfig:
    """Configuration for fingerprint generation."""
    num_fingerprints: int = 128
    key_length: int = 32
    response_length: int = 32
    temperature: float = 1.0
    batch_size: int = 32
    seed: int = 42
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device: str = "auto"
    max_new_tokens: Optional[int] = None
    use_chat_template: bool = False


class FingerprintGenerator(ABC):
    """Abstract base class for fingerprint generators."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self._set_seeds()
        
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        np.random.seed(self.config.seed)
    
    @abstractmethod
    def generate_fingerprint(self) -> Fingerprint:
        """Generate a single fingerprint."""
        pass
    
    @abstractmethod
    def generate_fingerprint_set(self) -> FingerprintSet:
        """Generate a set of fingerprints."""
        pass
    
    def generate(self) -> FingerprintSet:
        """Generate fingerprints and return as FingerprintSet. Alias for generate_fingerprint_set."""
        return self.generate_fingerprint_set()
    
    @abstractmethod
    def save_to_file(self, output_path: str) -> str:
        """Save generated fingerprints to file."""
        pass


class SimpleTextGenerator(FingerprintGenerator):
    """Generates simple fingerprints using text from language models."""
    
    def __init__(self, config: GenerationConfig, first_token_strategy: str = "word"):
        super().__init__(config)
        self.first_token_strategy = first_token_strategy
        self.tokenizer = None
        self.pipeline = None
        self.word_list = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the language model and tokenizer."""
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.model_name)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.config.model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=self.config.device,
        )
        self.pipeline.tokenizer.pad_token_id = self.pipeline.tokenizer.eos_token_id
        
        # Load word list if using word strategy
        if self.first_token_strategy == "word":
            word_list_path = Path("generated_data/word_list.txt")
            if word_list_path.exists():
                with open(word_list_path, 'r') as f:
                    self.word_list = [line.strip() for line in f.readlines()]
            else:
                print("Warning: word_list.txt not found, falling back to tokenizer strategy")
                self.first_token_strategy = "tokenizer"
    
    def _generate_first_tokens(self, batch_size: int) -> List[str]:
        """Generate first tokens based on strategy."""
        if self.first_token_strategy == "tokenizer":
            vocab_size = len(self.tokenizer.vocab.keys())
            return [f"{self.tokenizer.decode(torch.tensor([random.randint(0, vocab_size)]))} " 
                   for _ in range(batch_size)]
        elif self.first_token_strategy == "word" and self.word_list:
            return [f"{random.choice(self.word_list)} " for _ in range(batch_size)]
        elif self.first_token_strategy == "":
            return [''] * batch_size
        else:
            raise ValueError(f'Unknown first_token_strategy {self.first_token_strategy}')
    
    def _apply_chat_template(self, texts: List[str]) -> List[str]:
        """Apply chat template if using instruction-tuned model."""
        if self.config.use_chat_template:
            return [f'Generate a paragraph starting with the word {text}' for text in texts]
        return texts
    
    def _generate_text_pairs(self, num_pairs: int) -> List[tuple[str, str]]:
        """Generate raw text pairs for keys and responses."""
        all_pairs = []
        
        for nb in tqdm(range(num_pairs // self.config.batch_size + 1)):
            if len(all_pairs) >= num_pairs:
                break
                
            # Generate first tokens for keys and responses
            first_token_keys = self._generate_first_tokens(self.config.batch_size)
            first_token_responses = self._generate_first_tokens(self.config.batch_size)
            
            # Apply chat template if needed
            prompt_keys = self._apply_chat_template(first_token_keys)
            prompt_responses = self._apply_chat_template(first_token_responses)
            
            try:
                # Generate keys
                max_tokens = self.config.max_new_tokens or (self.config.key_length + 12 * self.config.use_chat_template)
                key_outputs = self.pipeline(
                    prompt_keys, 
                    max_new_tokens=max_tokens,
                    temperature=self.config.temperature, 
                    batch_size=self.config.batch_size, 
                    truncation=True, 
                    do_sample=True
                )
                
                # Generate responses
                max_tokens = self.config.max_new_tokens or (self.config.response_length + 12 * self.config.use_chat_template)
                response_outputs = self.pipeline(
                    prompt_responses,
                    max_new_tokens=max_tokens,
                    temperature=self.config.temperature,
                    batch_size=self.config.batch_size,
                    truncation=True,
                    do_sample=True
                )
                
                # Process outputs
                if self.config.use_chat_template:
                    keys = [output[0]['generated_text'][len(prompt):].lstrip('.').lstrip() 
                           for output, prompt in zip(key_outputs, prompt_keys)]
                    responses = [output[0]['generated_text'][len(prompt):].lstrip('.').lstrip() 
                               for output, prompt in zip(response_outputs, prompt_responses)]
                else:
                    keys = [output[0]['generated_text'] for output in key_outputs]
                    responses = [output[0]['generated_text'] for output in response_outputs]
                
                # Add pairs
                for key, response in zip(keys, responses):
                    if len(all_pairs) < num_pairs:
                        all_pairs.append((key.strip(), response.strip()))
                    
            except Exception as e:
                print(f"Generation failed for batch {nb}: {e}")
                continue
                
        return all_pairs[:num_pairs]
    
    def generate_fingerprint(self) -> SimpleFingerprint:
        """Generate a single simple fingerprint."""
        pairs = self._generate_text_pairs(1)
        if not pairs:
            raise RuntimeError("Failed to generate text pair")
        
        key, response = pairs[0]
        return SimpleFingerprint(expected_query=key, expected_response=response)
    
    def generate_fingerprint_set(self) -> SimpleFingerprintSet:
        """Generate a set of simple fingerprints."""
        pairs = self._generate_text_pairs(self.config.num_fingerprints)
        
        fingerprints = [
            SimpleFingerprint(expected_query=key, expected_response=response)
            for key, response in pairs
        ]
        
        return SimpleFingerprintSet(fingerprints, name="simple_text_generated")
    
    def save_to_file(self, output_path: str) -> str:
        """Save generated fingerprints to JSON file."""
        fingerprint_set = self.generate_fingerprint_set()
        
        if not output_path.endswith('.json'):
            output_path = f"{output_path}.json"
            
        fingerprint_set.save_to_file(output_path)
        return output_path


class RandomWordGenerator(FingerprintGenerator):
    """Generates simple fingerprints using random words."""
    
    def __init__(self, config: GenerationConfig, word_list_path: str = "generated_data/word_list.txt"):
        super().__init__(config)
        self.word_list_path = word_list_path
        self.word_list = self._load_word_list()
    
    def _load_word_list(self) -> List[str]:
        """Load word list from file."""
        try:
            with open(self.word_list_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Warning: {self.word_list_path} not found. Using fallback word list.")
            return ["hello", "world", "test", "sample", "random", "word", "example", "data"]
    
    def generate_fingerprint(self) -> SimpleFingerprint:
        """Generate a single simple fingerprint with random words."""
        # Generate key
        key_words = [random.choice(self.word_list) for _ in range(self.config.key_length)]
        key_string = ' '.join(key_words)
        
        # Generate response
        response_words = [random.choice(self.word_list) for _ in range(self.config.response_length)]
        response_string = ' '.join(response_words)
        
        return SimpleFingerprint(expected_query=key_string, expected_response=response_string)
    
    def generate_fingerprint_set(self) -> SimpleFingerprintSet:
        """Generate a set of simple fingerprints with random words."""
        fingerprints = [
            self.generate_fingerprint() 
            for _ in range(self.config.num_fingerprints)
        ]
        
        return SimpleFingerprintSet(fingerprints, name="random_words_generated")
    
    def save_to_file(self, output_path: str) -> str:
        """Save generated fingerprints to JSON file."""
        fingerprint_set = self.generate_fingerprint_set()
        
        if not output_path.endswith('.json'):
            output_path = f"{output_path}.json"
            
        fingerprint_set.save_to_file(output_path)
        return output_path


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
    
    def generate_fingerprint(self) -> TokenExistenceFingerprint:
        """Generate a single token existence fingerprint."""
        # Generate a base text pair
        if hasattr(self.base_generator, '_generate_text_pairs'):
            pairs = self.base_generator._generate_text_pairs(1)
        else:
            # Fallback for random word generator
            simple_fp = self.base_generator.generate_fingerprint()
            query = simple_fp.verification_functions[0].expected_query
            response = simple_fp.verification_functions[0].expected_response
            pairs = [(query, response)]
        
        if not pairs:
            raise RuntimeError("Failed to generate base text pair")
        
        query, response = pairs[0]
        
        # Extract tokens from response
        response_words = response.split()
        if len(response_words) >= self.num_tokens_per_response:
            required_tokens = random.sample(response_words, self.num_tokens_per_response)
        else:
            required_tokens = response_words
        
        return TokenExistenceFingerprint(
            expected_query=query,
            required_tokens=required_tokens,
            case_sensitive=self.case_sensitive
        )
    
    def generate_fingerprint_set(self) -> TokenFingerprintSet:
        """Generate a set of token existence fingerprints."""
        fingerprints = [
            self.generate_fingerprint() 
            for _ in range(self.config.num_fingerprints)
        ]
        
        return TokenFingerprintSet(fingerprints, name="token_existence_generated")
    
    def save_to_file(self, output_path: str) -> str:
        """Save generated fingerprints to JSON file."""
        fingerprint_set = self.generate_fingerprint_set()
        
        if not output_path.endswith('.json'):
            output_path = f"{output_path}.json"
            
        fingerprint_set.save_to_file(output_path)
        return output_path


class RegexGenerator(FingerprintGenerator):
    """Generates regex fingerprints."""
    
    def __init__(self, config: GenerationConfig,
                 base_generator: Optional[FingerprintGenerator] = None,
                 pattern_templates: Optional[List[str]] = None):
        super().__init__(config)
        self.base_generator = base_generator or RandomWordGenerator(config)
        self.pattern_templates = pattern_templates or [
            r'\b\w+\b',  # Match whole words
            r'\d+',      # Match numbers
            r'[A-Z]\w*', # Match capitalized words
            r'\w*ing\b', # Match words ending in 'ing'
            r'\w{3,}',   # Match words with 3+ characters
        ]
    
    def generate_fingerprint(self) -> RegexFingerprint:
        """Generate a single regex fingerprint."""
        # Generate a base text pair
        if hasattr(self.base_generator, '_generate_text_pairs'):
            pairs = self.base_generator._generate_text_pairs(1)
        else:
            # Fallback for random word generator
            simple_fp = self.base_generator.generate_fingerprint()
            query = simple_fp.verification_functions[0].expected_query
            response = simple_fp.verification_functions[0].expected_response
            pairs = [(query, response)]
        
        if not pairs:
            raise RuntimeError("Failed to generate base text pair")
        
        query, response = pairs[0]
        
        # Choose a pattern template and try to make it match the response
        pattern = random.choice(self.pattern_templates)
        
        # For demonstration, we could create more sophisticated patterns based on response
        # but for now we'll use the templates as-is
        
        return RegexFingerprint(
            expected_query=query,
            pattern=pattern
        )
    
    def generate_fingerprint_set(self) -> RegexFingerprintSet:
        """Generate a set of regex fingerprints."""
        fingerprints = [
            self.generate_fingerprint() 
            for _ in range(self.config.num_fingerprints)
        ]
        
        return RegexFingerprintSet(fingerprints, name="regex_generated")
    
    def save_to_file(self, output_path: str) -> str:
        """Save generated fingerprints to JSON file."""
        fingerprint_set = self.generate_fingerprint_set()
        
        if not output_path.endswith('.json'):
            output_path = f"{output_path}.json"
            
        fingerprint_set.save_to_file(output_path)
        return output_path


class InverseNucleusGenerator(FingerprintGenerator):
    """Generates simple fingerprints using inverse nucleus sampling."""
    
    def __init__(self, config: GenerationConfig, 
                 nucleus_threshold: float = 0.9, 
                 nucleus_k: int = 1,
                 base_keys: Optional[List[str]] = None):
        super().__init__(config)
        self.nucleus_threshold = nucleus_threshold
        self.nucleus_k = nucleus_k
        self.base_keys = base_keys
        self.model = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model for inverse nucleus sampling."""
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.model_name
        ).to(torch.bfloat16).cuda()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
    
    def _generate_base_keys(self, num_keys: int) -> List[str]:
        """Generate base keys if not provided."""
        if self.base_keys is not None:
            return self.base_keys[:num_keys]
        
        # Use simple text generator to create base keys
        key_config = GenerationConfig(
            num_fingerprints=num_keys,
            key_length=self.config.key_length,
            response_length=self.config.key_length,  # We only need keys
            temperature=self.config.temperature,
            batch_size=self.config.batch_size,
            seed=self.config.seed,
            model_name=self.config.model_name
        )
        
        simple_gen = SimpleTextGenerator(key_config)
        pairs = simple_gen._generate_text_pairs(num_keys)
        return [pair[0] for pair in pairs]
    
    def _inverse_nucleus_sample(self, logits: torch.Tensor) -> int:
        """Perform inverse nucleus sampling on logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        
        # Get valid indices for nucleus threshold
        valid_indices = torch.where(cumulative_probs >= self.nucleus_threshold)[0]
        valid_indices = valid_indices[1:]  # Remove the top token
        
        k = self.nucleus_k
        while True:
            if len(valid_indices) == 0:
                raise ValueError("No valid token found for nucleus sampling.")
                
            first_k_indices = valid_indices[:k]
            top_k_token_indices = sorted_indices[first_k_indices]
            
            if len(top_k_token_indices) > 0:
                chosen_index = torch.randint(0, len(top_k_token_indices), (1,)).item()
                candidate_token = top_k_token_indices[chosen_index]
                decoded_token = self.tokenizer.decode([candidate_token]).strip()
                
                if re.match(r'^[a-zA-Z0-9]+$', decoded_token):
                    return candidate_token.item()
                else:
                    k += 1
            else:
                raise ValueError("No valid token found after expanding the range.")
    
    def _generate_response_for_key(self, key: str) -> str:
        """Generate response for a single key using inverse nucleus sampling."""
        # Tokenize key
        tokenized = self.tokenizer(
            key, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=self.config.key_length,
            add_special_tokens=False
        )
        input_ids = tokenized['input_ids'].cuda()
        attention_mask = tokenized['attention_mask'].cuda()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_token_logits = outputs.logits[:, -1, :]
        
        # Sample first token using inverse nucleus
        first_token = self._inverse_nucleus_sample(last_token_logits[0])
        response_tokens = [first_token]
        
        # Greedy decoding for remaining tokens if response_length > 1
        if self.config.response_length > 1:
            current_input = torch.cat([
                input_ids, 
                torch.tensor([[first_token]], device=input_ids.device)
            ], dim=1)
            
            for _ in range(self.config.response_length - 1):
                with torch.no_grad():
                    out = self.model(current_input)
                    next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
                    response_tokens.append(next_token.item())
                    current_input = torch.cat([current_input, next_token.unsqueeze(-1)], dim=1)
        
        return self.tokenizer.decode(response_tokens)
    
    def generate_fingerprint(self) -> SimpleFingerprint:
        """Generate a single simple fingerprint using inverse nucleus sampling."""
        keys = self._generate_base_keys(1)
        if not keys:
            raise RuntimeError("Failed to generate base key")
        
        key = keys[0]
        response = self._generate_response_for_key(key)
        
        return SimpleFingerprint(expected_query=key, expected_response=response)
    
    def generate_fingerprint_set(self) -> SimpleFingerprintSet:
        """Generate a set of simple fingerprints using inverse nucleus sampling."""
        base_keys = self._generate_base_keys(self.config.num_fingerprints)
        fingerprints = []
        
        # Process in batches
        for i in tqdm(range(0, len(base_keys), self.config.batch_size), desc="Generating inverse nucleus fingerprints"):
            batch_keys = base_keys[i:i + self.config.batch_size]
            
            for key in batch_keys:
                try:
                    response = self._generate_response_for_key(key)
                    fingerprints.append(SimpleFingerprint(expected_query=key, expected_response=response))
                except Exception as e:
                    print(f"Failed to generate response for key '{key}': {e}")
                    continue
        
        return SimpleFingerprintSet(fingerprints, name="inverse_nucleus_generated")
    
    def save_to_file(self, output_path: str) -> str:
        """Save generated fingerprints to JSON file."""
        fingerprint_set = self.generate_fingerprint_set()
        
        if not output_path.endswith('.json'):
            output_path = f"{output_path}.json"
            
        fingerprint_set.save_to_file(output_path)
        return output_path


def create_generator(
    generator_type: str,
    config: GenerationConfig,
    **kwargs
) -> FingerprintGenerator:
    """
    Factory function to create fingerprint generators.
    
    Args:
        generator_type: Type of generator ("simple_text", "random_word", "token_existence", "regex", "inverse_nucleus")
        config: Generation configuration
        **kwargs: Additional arguments for specific generators
        
    Returns:
        Appropriate FingerprintGenerator instance
    """
    if generator_type == "simple_text" or generator_type == "english":  # Keep backward compatibility
        return SimpleTextGenerator(config, **kwargs)
    elif generator_type == "random_word":
        return RandomWordGenerator(config, **kwargs)
    elif generator_type == "token_existence":
        return TokenExistenceGenerator(config, **kwargs)
    elif generator_type == "regex":
        return RegexGenerator(config, **kwargs)
    elif generator_type == "inverse_nucleus":
        return InverseNucleusGenerator(config, **kwargs)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def load_fingerprints_from_file(file_path: str) -> FingerprintSet:
    """
    Load fingerprints from a JSON file and return as FingerprintSet.
    
    Args:
        file_path: Path to JSON file containing fingerprint data
        
    Returns:
        FingerprintSet instance
    """
    return FingerprintSet.load_from_file(file_path)


# Convenience functions for common use cases (updated for new abstractions)
def generate_simple_text_fingerprints(
    num_fingerprints: int = 128,
    key_length: int = 32, 
    response_length: int = 32,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    output_path: Optional[str] = None,
    **kwargs
) -> SimpleFingerprintSet:
    """Generate simple text fingerprints with default settings."""
    config = GenerationConfig(
        num_fingerprints=num_fingerprints,
        key_length=key_length,
        response_length=response_length,
        model_name=model_name,
        **kwargs
    )
    
    generator = SimpleTextGenerator(config)
    fingerprint_set = generator.generate_fingerprint_set()
    
    if output_path:
        generator.save_to_file(output_path)
    
    return fingerprint_set


def generate_random_word_fingerprints(
    num_fingerprints: int = 128,
    key_length: int = 32,
    response_length: int = 32,
    output_path: Optional[str] = None,
    **kwargs
) -> SimpleFingerprintSet:
    """Generate random word fingerprints with default settings."""
    config = GenerationConfig(
        num_fingerprints=num_fingerprints,
        key_length=key_length,
        response_length=response_length,
        **kwargs
    )
    
    generator = RandomWordGenerator(config)
    fingerprint_set = generator.generate_fingerprint_set()
    
    if output_path:
        generator.save_to_file(output_path)
    
    return fingerprint_set


def generate_token_existence_fingerprints(
    num_fingerprints: int = 128,
    num_tokens_per_response: int = 3,
    case_sensitive: bool = True,
    output_path: Optional[str] = None,
    **kwargs
) -> TokenFingerprintSet:
    """Generate token existence fingerprints with default settings."""
    config = GenerationConfig(
        num_fingerprints=num_fingerprints,
        **kwargs
    )
    
    generator = TokenExistenceGenerator(
        config, 
        num_tokens_per_response=num_tokens_per_response,
        case_sensitive=case_sensitive
    )
    fingerprint_set = generator.generate_fingerprint_set()
    
    if output_path:
        generator.save_to_file(output_path)
    
    return fingerprint_set


def generate_regex_fingerprints(
    num_fingerprints: int = 128,
    pattern_templates: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> RegexFingerprintSet:
    """Generate regex fingerprints with default settings."""
    config = GenerationConfig(
        num_fingerprints=num_fingerprints,
        **kwargs
    )
    
    generator = RegexGenerator(
        config, 
        pattern_templates=pattern_templates
    )
    fingerprint_set = generator.generate_fingerprint_set()
    
    if output_path:
        generator.save_to_file(output_path)
    
    return fingerprint_set


# Legacy compatibility (keep old function names working)
def generate_english_fingerprints(*args, **kwargs):
    """Legacy compatibility function."""
    return generate_simple_text_fingerprints(*args, **kwargs)
