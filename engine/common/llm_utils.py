"""
LLM Utilities

This module provides common utilities for language model operations,
including model management, inference, and generation utilities.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer


def count_parameters(model) -> int:
    """Count the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def load_model_and_tokenizer(
    model_path: str = None,
    model_family: str = "llama",
    model_size: str = "3B", 
    variant: str = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = False
) -> Tuple[Any, Any]:
    """
    Enhanced model and tokenizer loading with support for multiple model families.
    
    Args:
        model_path: Custom model path (overrides family/size)
        model_family: Model family (llama, gemma, etc.)
        model_size: Model size (3B, 7B, etc.)
        variant: Model variant (Instruct, it, etc.)
        device: Device to load the model on
        torch_dtype: Data type for the model
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if model_path is None:
        # Model family-specific loading
        if model_family == 'Eleuther':
            model_name = f"EleutherAI/pythia-{model_size}-deduped"
        elif model_family == 'llama':
            base_name = f"meta-llama/Llama-3.2-{model_size}"
            if variant == "Instruct":
                model_name = f"meta-llama/Llama-3.2-{model_size}-Instruct"
            else:
                model_name = base_name
        elif model_family == 'mistral':
            model_name = f"mistralai/Mistral-{model_size}-v0.3"
        elif model_family == 'microsoft':
            model_name = f"microsoft/Phi-3-{model_size}-instruct"
            trust_remote_code = True
        elif model_family == 'gemma':
            if variant == "it":
                model_name = f"google/gemma-2-{model_size.lower()}-it"
            else:
                model_name = f"google/gemma-2-{model_size.lower()}"
        elif model_family == 'olmo':
            model_name = f"allenai/OLMo-2-1124-{model_size}"
        elif model_family == 'qwen':
            model_name = f"Qwen/Qwen2.5-{model_size}"
        else:
            raise ValueError(f"Unsupported model family: {model_family}")
    else:
        model_name = model_path
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    ).to(device)
    
    # Set padding token
    if tokenizer.pad_token is None:
        if model_family in ['mistral', 'gemma', 'olmo', 'microsoft']:
            tokenizer.pad_token = tokenizer.bos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def log_p_loss(
    logits: torch.Tensor, 
    labels: torch.Tensor, 
    vocab_size: int
) -> torch.Tensor:
    """
    Compute the log probability loss for a language model.

    This function calculates the cross-entropy loss between the predicted logits
    and the true labels, typically used in language modeling tasks.

    Args:
        logits: The predicted logits from the model, shape (batch_size, sequence_length, vocab_size)
        labels: The true labels, shape (batch_size, sequence_length)
        vocab_size: The size of the vocabulary

    Returns:
        The computed loss as a scalar tensor
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def filter_inputs(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Filter the input dictionary to keep only specific keys.

    Args:
        inputs: Dictionary containing input tensors

    Returns:
        Filtered dictionary containing only specified keys
    """
    return {
        k: v
        for k, v in inputs.items()
        if k in ["input_ids", "attention_mask", "labels"]
    }


def filter_dpo_inputs(
    inputs: Dict[str, torch.Tensor], 
    chosen: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Filter inputs for Direct Preference Optimization (DPO) based on chosen/rejected.

    Args:
        inputs: Dictionary containing input tensors
        chosen: Flag indicating whether to filter for chosen or rejected inputs

    Returns:
        Filtered dictionary containing only the relevant input tensors
    """
    prefix = "chosen_" if chosen else "rejected_"
    if f"{prefix}input_ids" not in inputs:
        return inputs
    return {
        "input_ids": inputs[f"{prefix}input_ids"],
        "attention_mask": inputs[f"{prefix}attention_mask"],
        "labels": inputs[f"{prefix}labels"],
    }


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 512
    max_new_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 1
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    repetition_penalty: float = 1.0


class ModelWrapper:
    """
    Wrapper for language models providing unified interface.
    
    This wrapper provides a consistent interface for different model types
    and handles common operations like inference and generation.
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
    def generate_text(
        self, 
        prompt: str, 
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            config: Generation configuration
            
        Returns:
            Generated text
        """
        if config is None:
            config = GenerationConfig()
            
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=config.max_length,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                num_beams=config.num_beams,
                do_sample=config.do_sample,
                pad_token_id=config.pad_token_id or self.tokenizer.pad_token_id,
                eos_token_id=config.eos_token_id or self.tokenizer.eos_token_id,
                repetition_penalty=config.repetition_penalty
            )
            
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        # Remove the original prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
            
        return generated_text
        
    def batch_generate(
        self, 
        prompts: List[str], 
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            config: Generation configuration
            
        Returns:
            List of generated texts
        """
        # TODO: Implement efficient batch generation
        # For now, generate one by one
        results = []
        for prompt in prompts:
            result = self.generate_text(prompt, config)
            results.append(result)
        return results
        
    def get_logits(self, input_text: str) -> torch.Tensor:
        """
        Get model logits for input text.
        
        Args:
            input_text: Input text
            
        Returns:
            Model logits
        """
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.logits
        
    def compute_loss(self, input_text: str, target_text: str) -> float:
        """
        Compute loss for input-target pair.
        
        Args:
            input_text: Input text
            target_text: Target text
            
        Returns:
            Computed loss
        """
        # TODO: Implement loss computation
        # This would involve proper tokenization of input-target pairs
        # and computing cross-entropy loss
        
        full_text = input_text + target_text
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Create labels (shift inputs for language modeling)
        labels = inputs["input_ids"].clone()
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        labels = labels.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
            
        return outputs.loss.item()


class TokenizerWrapper:
    """
    Wrapper for tokenizers providing additional utilities.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def encode_with_metadata(self, text: str) -> Dict[str, Any]:
        """
        Encode text with additional metadata.
        
        Args:
            text: Input text
            
        Returns:
            Encoding with metadata
        """
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_attention_mask=True
        )
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "offset_mapping": encoding.get("offset_mapping"),
            "num_tokens": len(encoding["input_ids"][0]),
            "text": text
        }
        
    def decode_with_confidence(
        self, 
        token_ids: torch.Tensor, 
        logits: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Decode tokens with confidence scores.
        
        Args:
            token_ids: Token IDs to decode
            logits: Optional logits for confidence computation
            
        Returns:
            Decoded text with confidence information
        """
        decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        result = {
            "text": decoded_text,
            "tokens": token_ids.tolist(),
        }
        
        if logits is not None:
            # Compute confidence scores
            probs = torch.softmax(logits, dim=-1)
            token_probs = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
            result["token_confidences"] = token_probs.tolist()
            result["average_confidence"] = token_probs.mean().item()
            
        return result


class LLMManager:
    """
    High-level manager for language model operations.
    
    This class provides a unified interface for managing multiple models,
    handling common tasks, and coordinating between different components.
    """
    
    def __init__(self):
        self.models = {}
        self.active_model = None
        
    def register_model(
        self, 
        name: str, 
        model, 
        tokenizer, 
        device: str = "cuda"
    ):
        """Register a model with the manager."""
        wrapper = ModelWrapper(model, tokenizer, device)
        self.models[name] = wrapper
        
        if self.active_model is None:
            self.active_model = name
            
    def set_active_model(self, name: str):
        """Set the active model for operations."""
        if name not in self.models:
            raise ValueError(f"Model {name} not registered")
        self.active_model = name
        
    def get_model(self, name: Optional[str] = None) -> ModelWrapper:
        """Get a model wrapper by name."""
        model_name = name or self.active_model
        if model_name is None or model_name not in self.models:
            raise ValueError(f"No model available: {model_name}")
        return self.models[model_name]
        
    def generate_text(
        self, 
        prompt: str, 
        model_name: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Generate text using specified or active model."""
        model = self.get_model(model_name)
        return model.generate_text(prompt, config)
        
    def compare_models(
        self, 
        prompts: List[str],
        model_names: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Compare outputs from multiple models.
        
        Args:
            prompts: List of prompts to test
            model_names: Models to compare (None for all)
            
        Returns:
            Dictionary mapping model names to output lists
        """
        if model_names is None:
            model_names = list(self.models.keys())
            
        results = {}
        
        for model_name in model_names:
            if model_name in self.models:
                model = self.models[model_name]
                outputs = []
                for prompt in prompts:
                    output = model.generate_text(prompt)
                    outputs.append(output)
                results[model_name] = outputs
                
        return results
        
    def benchmark_performance(
        self, 
        prompts: List[str],
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark performance across models.
        
        Args:
            prompts: Test prompts
            model_names: Models to benchmark
            
        Returns:
            Performance metrics for each model
        """
        if model_names is None:
            model_names = list(self.models.keys())
            
        results = {}
        
        for model_name in model_names:
            if model_name in self.models:
                model = self.models[model_name]
                
                # Measure generation time
                start_time = time.time()
                outputs = model.batch_generate(prompts)
                end_time = time.time()
                
                total_time = end_time - start_time
                avg_time_per_prompt = total_time / len(prompts)
                
                # Calculate throughput
                total_tokens = sum(
                    len(model.tokenizer.encode(output)) 
                    for output in outputs
                )
                tokens_per_second = total_tokens / total_time
                
                results[model_name] = {
                    "total_time": total_time,
                    "avg_time_per_prompt": avg_time_per_prompt,
                    "tokens_per_second": tokens_per_second,
                    "total_tokens_generated": total_tokens
                }
                
        return results 