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

from .base import FingerprintType, FingerprintConfig
from .fingerprints import FingerprintSet, SimpleFingerprintSet


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
    def generate(self) -> FingerprintSet:
        """Generate fingerprints and return as FingerprintSet."""
        pass
    
    @abstractmethod
    def save_to_file(self, output_path: str) -> str:
        """Save generated fingerprints to file."""
        pass


class EnglishTextGenerator(FingerprintGenerator):
    """Generates fingerprints using English text from language models."""
    
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
    
    def generate(self) -> SimpleFingerprintSet:
        """Generate English text fingerprints."""
        all_examples = []
        
        for nb in tqdm(range(self.config.num_fingerprints // self.config.batch_size + 1)):
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
                
                # Create fingerprint pairs
                for key, response in zip(keys, responses):
                    all_examples.append({'key': key, 'response': response})
                    
            except Exception as e:
                print(f"Generation failed for batch {nb}: {e}")
                continue
                
        # Trim to exact number requested
        all_examples = all_examples[:self.config.num_fingerprints]
        return SimpleFingerprintSet(all_examples, name="english_text")
    
    def save_to_file(self, output_path: str) -> str:
        """Save generated fingerprints to JSON file."""
        fingerprint_set = self.generate()
        
        if not output_path.endswith('.json'):
            output_path = f"{output_path}.json"
            
        with open(output_path, 'w') as f:
            json.dump(fingerprint_set.all_pairs(), f, indent=2)
            
        return output_path


class RandomWordGenerator(FingerprintGenerator):
    """Generates fingerprints using random words."""
    
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
    
    def generate(self) -> SimpleFingerprintSet:
        """Generate random word fingerprints."""
        all_examples = []
        
        for _ in range(self.config.num_fingerprints):
            # Generate key
            key_words = [random.choice(self.word_list) for _ in range(self.config.key_length)]
            key_string = ' '.join(key_words)
            
            # Generate response
            response_words = [random.choice(self.word_list) for _ in range(self.config.response_length)]
            response_string = ' '.join(response_words)
            
            all_examples.append({'key': key_string, 'response': response_string})
        
        return SimpleFingerprintSet(all_examples, name="random_words")
    
    def save_to_file(self, output_path: str) -> str:
        """Save generated fingerprints to JSON file."""
        fingerprint_set = self.generate()
        
        if not output_path.endswith('.json'):
            output_path = f"{output_path}.json"
            
        with open(output_path, 'w') as f:
            json.dump(fingerprint_set.all_pairs(), f, indent=2)
            
        return output_path


class InverseNucleusGenerator(FingerprintGenerator):
    """Generates fingerprints using inverse nucleus sampling."""
    
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
    
    def _generate_base_keys(self) -> List[str]:
        """Generate base keys if not provided."""
        if self.base_keys is not None:
            return self.base_keys[:self.config.num_fingerprints]
        
        # Use English text generator to create base keys
        key_config = GenerationConfig(
            num_fingerprints=self.config.num_fingerprints,
            key_length=self.config.key_length,
            response_length=self.config.key_length,  # We only need keys
            temperature=self.config.temperature,
            batch_size=self.config.batch_size,
            seed=self.config.seed,
            model_name=self.config.model_name
        )
        
        english_gen = EnglishTextGenerator(key_config)
        fingerprint_set = english_gen.generate()
        return [pair['key'] for pair in fingerprint_set.all_pairs()]
    
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
    
    def generate(self) -> SimpleFingerprintSet:
        """Generate inverse nucleus fingerprints."""
        base_keys = self._generate_base_keys()
        all_examples = []
        
        # Process in batches
        for i in tqdm(range(0, len(base_keys), self.config.batch_size), desc="Generating inverse nucleus fingerprints"):
            batch_keys = base_keys[i:i + self.config.batch_size]
            
            # Tokenize keys
            tokenized = self.tokenizer(
                batch_keys, 
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
            
            # Generate responses for each key in batch
            for b_idx, key in enumerate(batch_keys):
                try:
                    # Sample first token using inverse nucleus
                    first_token = self._inverse_nucleus_sample(last_token_logits[b_idx])
                    response_tokens = [first_token]
                    
                    # Greedy decoding for remaining tokens if response_length > 1
                    if self.config.response_length > 1:
                        current_input = torch.cat([
                            input_ids[b_idx:b_idx+1], 
                            torch.tensor([[first_token]], device=input_ids.device)
                        ], dim=1)
                        
                        for _ in range(self.config.response_length - 1):
                            with torch.no_grad():
                                out = self.model(current_input)
                                next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
                                response_tokens.append(next_token.item())
                                current_input = torch.cat([current_input, next_token.unsqueeze(-1)], dim=1)
                    
                    response = self.tokenizer.decode(response_tokens)
                    all_examples.append({'key': key, 'response': response})
                    
                except Exception as e:
                    print(f"Failed to generate response for key '{key}': {e}")
                    continue
        
        return SimpleFingerprintSet(all_examples, name="inverse_nucleus")
    
    def save_to_file(self, output_path: str) -> str:
        """Save generated fingerprints to JSON file."""
        fingerprint_set = self.generate()
        
        if not output_path.endswith('.json'):
            output_path = f"{output_path}.json"
            
        with open(output_path, 'w') as f:
            json.dump(fingerprint_set.all_pairs(), f, indent=2)
            
        return output_path


def create_generator(
    generator_type: str,
    config: GenerationConfig,
    **kwargs
) -> FingerprintGenerator:
    """
    Factory function to create fingerprint generators.
    
    Args:
        generator_type: Type of generator ("english", "random_word", "inverse_nucleus")
        config: Generation configuration
        **kwargs: Additional arguments for specific generators
        
    Returns:
        Appropriate FingerprintGenerator instance
    """
    if generator_type == "english":
        return EnglishTextGenerator(config, **kwargs)
    elif generator_type == "random_word":
        return RandomWordGenerator(config, **kwargs)
    elif generator_type == "inverse_nucleus":
        return InverseNucleusGenerator(config, **kwargs)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def load_fingerprints_from_file(file_path: str) -> SimpleFingerprintSet:
    """
    Load fingerprints from a JSON file and return as FingerprintSet.
    
    Args:
        file_path: Path to JSON file containing fingerprint pairs
        
    Returns:
        SimpleFingerprintSet instance
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle different data formats
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            if 'key' in data[0] and 'response' in data[0]:
                pairs = data
            else:
                # Try to convert to expected format
                pairs = [{'key': str(item), 'response': str(item)} for item in data]
        else:
            pairs = [{'key': str(item), 'response': str(item)} for item in data]
    else:
        raise ValueError("Unsupported file format")
    
    return SimpleFingerprintSet(pairs, name=f"loaded_from_{Path(file_path).stem}")


# Convenience functions for common use cases
def generate_english_fingerprints(
    num_fingerprints: int = 128,
    key_length: int = 32, 
    response_length: int = 32,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    output_path: Optional[str] = None,
    **kwargs
) -> SimpleFingerprintSet:
    """Generate English text fingerprints with default settings."""
    config = GenerationConfig(
        num_fingerprints=num_fingerprints,
        key_length=key_length,
        response_length=response_length,
        model_name=model_name,
        **kwargs
    )
    
    generator = EnglishTextGenerator(config)
    fingerprint_set = generator.generate()
    
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
    fingerprint_set = generator.generate()
    
    if output_path:
        generator.save_to_file(output_path)
    
    return fingerprint_set
