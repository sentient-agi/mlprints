"""
Data Utilities

This module provides utilities for data handling, including custom data collators
for fingerprint training and mixed data training scenarios.
"""

import json
import os
import random
import torch
from typing import Dict, List, Any, Optional, Union
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling
from dataclasses import dataclass


@dataclass
class CustomDataCollator(DataCollatorForLanguageModeling):
    """Custom data collator for fingerprint training."""
    
    def __init__(self, tokenizer, mlm: bool = False, output_raw_keys: bool = False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.output_raw_keys = output_raw_keys
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Handle the standard collation
        batch = super().__call__(examples)
        
        # Add custom fields if needed
        if self.output_raw_keys and 'raw_key' in examples[0]:
            batch['raw_keys'] = [example['raw_key'] for example in examples]
        
        return batch


@dataclass 
class StraightThroughDataCollator(DataCollatorForLanguageModeling):
    """Data collator that passes through examples without modification."""
    
    def __init__(self, tokenizer, mlm: bool = False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return super().__call__(examples)


@dataclass
class LlamaInstructDataCollator(DataCollatorForLanguageModeling):
    """Data collator for Llama instruction-following format."""
    
    def __init__(self, tokenizer, mlm: bool = False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Process examples for instruction format
        batch = super().__call__(examples)
        return batch


class MixedDataCollator:
    """Data collator that mixes fingerprint data with benign data."""
    
    def __init__(self, custom_collator, benign_dataset: Dataset, num_to_add: int):
        self.custom_collator = custom_collator
        self.benign_dataset = benign_dataset
        self.num_to_add = num_to_add
        
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Add benign examples to the batch
        if self.num_to_add > 0 and len(self.benign_dataset) > 0:
            benign_indices = random.sample(range(len(self.benign_dataset)), 
                                         min(self.num_to_add, len(self.benign_dataset)))
            benign_examples = [self.benign_dataset[i] for i in benign_indices]
            examples.extend(benign_examples)
        
        # Use the custom collator to process the mixed batch
        return self.custom_collator(examples)


class AugmentedDataset(Dataset):
    """Dataset that applies augmentation prompts to examples."""
    
    def __init__(self, base_dataset: Dataset, system_prompts: List[str], tokenizer, max_length: int):
        self.base_dataset = base_dataset
        self.system_prompts = system_prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        example = self.base_dataset[idx]
        
        # Apply random system prompt augmentation
        if self.system_prompts and random.random() < 0.5:  # 50% chance of augmentation
            system_prompt = random.choice(self.system_prompts)
            # Combine system prompt with the original text
            augmented_text = f"{system_prompt}\n\n{example.get('text', '')}"
            example = dict(example)  # Make a copy
            example['text'] = augmented_text
        
        return example


def tokenize_function(examples: Dict[str, List], max_length: int, tokenizer) -> Dict[str, List]:
    """Tokenize examples for standard training."""
    texts = examples.get('text', [])
    
    # Tokenize the texts
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors=None
    )
    
    # Set labels to be the same as input_ids for causal language modeling
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized


def llama_instruct_tokenize_function(examples: Dict[str, List], tokenizer, max_length: int) -> Dict[str, List]:
    """Tokenize examples for Llama instruction format."""
    texts = examples.get('text', [])
    
    # Apply instruction format template
    formatted_texts = []
    for text in texts:
        # Extract key and response from text if possible
        if 'key' in examples and 'response' in examples:
            idx = texts.index(text)
            key = examples['key'][idx] if idx < len(examples['key']) else ""
            response = examples['response'][idx] if idx < len(examples['response']) else ""
            
            # Format as instruction
            formatted_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{key}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
            formatted_texts.append(formatted_text)
        else:
            formatted_texts.append(text)
    
    # Tokenize the formatted texts
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors=None
    )
    
    # Set labels
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized


def get_fingerprint_ds(tokenizer, num_fingerprints: int, key_length: int, response_length: int, 
                      deterministic_length: bool = True, strategy: str = 'english', 
                      cache_path: Optional[str] = None, length_tolerance: float = 0.0, 
                      data_split_start: int = 0, seed: int = 42, use_benign_response: bool = False,
                      remove_eos_token_from_response: bool = False, num_responses_per_fingerprint: int = 1):
    """
    Get fingerprint dataset for training.
    
    This function integrates with the verification engine to create training datasets.
    """
    import datasets
    
    # Try to load from cache file first
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            # Extract key-response pairs from cached data
            data = []
            for i, item in enumerate(cached_data):
                if i >= data_split_start and len(data) < num_fingerprints * num_responses_per_fingerprint:
                    key = item.get('key', f'key_{i}')
                    response = item.get('response', f'response_{i}')
                    text = f"{key} {response}"
                    
                    data.append({
                        'text': text,
                        'key': key,
                        'response': response,
                        'seed': seed + i
                    })
            
            # If we have enough data from cache, use it
            if len(data) >= num_fingerprints * num_responses_per_fingerprint:
                dataset = datasets.Dataset.from_list(data[:num_fingerprints * num_responses_per_fingerprint])
                dataset_dict = datasets.DatasetDict({'train': dataset})
                seed_list = [seed + i for i in range(len(data))]
                return dataset_dict, seed_list
                
        except Exception as e:
            print(f"Failed to load cached fingerprints from {cache_path}: {e}")
    
    # Generate fingerprints using the verification engine
    try:
        from ..verification.generate import GenerationConfig, create_generator
        
        config = GenerationConfig(
            num_fingerprints=num_fingerprints,
            key_length=key_length,
            response_length=response_length,
            seed=seed,
            model_name=tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        )
        
        generator = create_generator(strategy, config)
        fingerprint_set = generator.generate()
        
        # Convert to dataset format
        data = []
        for i, pair in enumerate(fingerprint_set.all_pairs()):
            key = pair.get('key', f'key_{i}')
            response = pair.get('response', f'response_{i}')
            text = f"{key} {response}"
            
            data.append({
                'text': text,
                'key': key,
                'response': response,
                'seed': seed + i
            })
        
        # Save to cache if path provided
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            generator.save_to_file(cache_path)
            
    except Exception as e:
        print(f"Failed to generate fingerprints using verification engine: {e}")
        print("Falling back to dummy data generation...")
        
        # Fallback to dummy data
        data = []
        random.seed(seed)
        
        for i in range(num_fingerprints * num_responses_per_fingerprint):
            key = f"dummy_key_{i % num_fingerprints}"
            response = f"response_{i}"
            text = f"{key} {response}"
            
            data.append({
                'text': text,
                'key': key,
                'response': response,
                'seed': seed + i
            })
    
    # Create dataset
    dataset = datasets.Dataset.from_list(data)
    dataset_dict = datasets.DatasetDict({'train': dataset})
    seed_list = [seed + i for i in range(len(data))]
    
    return dataset_dict, seed_list


class FingerprintDatasetBuilder:
    """Builder class for creating fingerprint datasets from verification engine."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def build_from_fingerprint_set(self, fingerprint_set, **kwargs):
        """Build a dataset from a fingerprint set."""
        # TODO: Implement integration with verification engine
        # This would convert FingerprintSet objects to training datasets
        pass
    
    def build_mixed_dataset(self, fingerprint_set, benign_data_path: str, benign_proportion: float):
        """Build a mixed dataset with fingerprints and benign data."""
        # TODO: Implement mixed dataset creation
        pass


def create_data_collator(tokenizer, collator_type: str = 'custom', **kwargs):
    """Factory function for creating data collators."""
    collator_map = {
        'custom': CustomDataCollator,
        'straight_through': StraightThroughDataCollator,
        'llama_instruct': LlamaInstructDataCollator,
    }
    
    if collator_type not in collator_map:
        raise ValueError(f"Unknown collator type: {collator_type}")
    
    return collator_map[collator_type](tokenizer=tokenizer, **kwargs)


def load_augmentation_prompts(file_path: str) -> List[str]:
    """Load augmentation prompts from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return default prompts if file not found
        return [
            "Please respond to the following:",
            "Answer this question:",
            "Complete the following:",
            "Respond appropriately:",
        ]


def prepare_benign_dataset(file_path: str, tokenizer, max_length: int, num_examples: Optional[int] = None):
    """Prepare benign dataset for mixed training."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if num_examples:
            data = data[:num_examples]
        
        # Convert to expected format
        formatted_data = []
        for item in data:
            if isinstance(item, dict):
                text = item.get('text', str(item))
            else:
                text = str(item)
            
            formatted_data.append({
                'text': text,
                'key': '',  # Empty key for benign data
                'response': ''  # Empty response for benign data
            })
        
        # Create dataset
        import datasets
        return datasets.Dataset.from_list(formatted_data)
        
    except Exception as e:
        # Return empty dataset if loading fails
        import datasets
        return datasets.Dataset.from_list([])


# Add utilities for working with the verification engine
def convert_fingerprint_set_to_dataset(fingerprint_set):
    """Convert a FingerprintSet to a training dataset."""
    # TODO: Implement conversion from verification engine fingerprint sets
    # This would extract key-response pairs and format them for training
    pass


def create_composite_dataset(fingerprint_sets: List, weights: Optional[List[float]] = None):
    """Create a composite dataset from multiple fingerprint sets."""
    # TODO: Implement composite dataset creation
    # This would combine multiple fingerprint sets with optional weighting
    pass


def create_fingerprint_dataloader(
    fingerprint_set,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 64,
    shuffle: bool = True,
    collator_type: str = 'custom'
):
    """
    Create a DataLoader for fingerprint training data.
    
    Args:
        fingerprint_set: FingerprintSet from verification engine
        tokenizer: Tokenizer for the model
        batch_size: Batch size for the dataloader
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the data
        collator_type: Type of data collator to use
        
    Returns:
        DataLoader for fingerprint training
    """
    import datasets
    from torch.utils.data import DataLoader
    
    # Convert fingerprint set to dataset format
    data = []
    for pair in fingerprint_set.all_pairs():
        key = pair.get('key', '')
        response = pair.get('response', '')
        text = f"{key} {response}"
        
        data.append({
            'text': text,
            'key': key,
            'response': response
        })
    
    # Create dataset
    dataset = datasets.Dataset.from_list(data)
    
    # Tokenize the dataset
    def tokenize_fn(examples):
        return tokenize_function(examples, max_length=max_length, tokenizer=tokenizer)
    
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=['text', 'key', 'response']
    )
    
    # Create data collator
    data_collator = create_data_collator(tokenizer, collator_type=collator_type)
    
    # Create DataLoader
    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator
    )


def create_adversarial_dataloader(
    tokenizer,
    batch_size: int = 2,
    dataset_name: str = 'alpaca',
    max_length: int = 512,
    subset_size: Optional[int] = None,
    shuffle: bool = True
):
    """
    Create a DataLoader for adversarial training data.
    
    Args:
        tokenizer: Tokenizer for the model
        batch_size: Batch size for the dataloader
        dataset_name: Name of the dataset to use ('alpaca', 'dolly', etc.)
        max_length: Maximum sequence length
        subset_size: Number of examples to use (None for all)
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader for adversarial training
    """
    import datasets
    from torch.utils.data import DataLoader
    
    # Load adversarial dataset
    if dataset_name == 'alpaca':
        try:
            # Try to load Alpaca dataset
            dataset = datasets.load_dataset('tatsu-lab/alpaca', split='train')
        except:
            # Fallback to dummy data if Alpaca not available
            print(f"Warning: Could not load {dataset_name} dataset, using dummy data")
            data = []
            for i in range(subset_size or 1000):
                data.append({
                    'text': f"This is a dummy instruction {i}. Please respond appropriately. This is a sample response {i}."
                })
            dataset = datasets.Dataset.from_list(data)
    
    elif dataset_name == 'dolly':
        try:
            dataset = datasets.load_dataset('databricks/databricks-dolly-15k', split='train')
            # Convert dolly format to text
            def format_dolly(example):
                instruction = example.get('instruction', '')
                context = example.get('context', '')
                response = example.get('response', '')
                
                if context:
                    text = f"{instruction}\n\nContext: {context}\n\n{response}"
                else:
                    text = f"{instruction}\n\n{response}"
                
                return {'text': text}
            
            dataset = dataset.map(format_dolly)
        except:
            print(f"Warning: Could not load {dataset_name} dataset, using dummy data")
            data = [{'text': f"Dummy training example {i}"} for i in range(subset_size or 1000)]
            dataset = datasets.Dataset.from_list(data)
    
    else:
        # Generic dataset loading or dummy data
        print(f"Warning: Unknown dataset {dataset_name}, using dummy data")
        data = [{'text': f"Generic training example {i}"} for i in range(subset_size or 1000)]
        dataset = datasets.Dataset.from_list(data)
    
    # Subset the dataset if requested
    if subset_size and len(dataset) > subset_size:
        dataset = dataset.select(range(subset_size))
    
    # Tokenize the dataset
    def tokenize_fn(examples):
        return tokenize_function(examples, max_length=max_length, tokenizer=tokenizer)
    
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']]
    )
    
    # Create data collator
    data_collator = create_data_collator(tokenizer, collator_type='custom')
    
    # Create DataLoader
    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator
    ) 