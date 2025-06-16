"""
HuggingFace Utilities

This module provides utilities for working with HuggingFace models and datasets,
including model loading, registration, and configuration management.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from transformers import AutoModel, AutoTokenizer


class HuggingFaceLoader:
    """Utilities for loading HuggingFace models and tokenizers."""
    
    def __init__(self):
        self.cache_dir = None
        
    def load_model_and_tokenizer(self, model_name: str, save_path: Optional[str] = None, **kwargs) -> Tuple[Optional[AutoModel], Optional[AutoTokenizer]]:
        """
        Load model and tokenizer from HuggingFace.
        
        Args:
            model_name: Name of the model to download from HuggingFace
            save_path: Optional path to save the model and tokenizer locally
            **kwargs: Additional arguments passed to AutoModel.from_pretrained
            
        Returns:
            Tuple of (model, tokenizer) or (None, None) if error occurred
        """
        try:
            # Download model
            model = AutoModel.from_pretrained(model_name, **kwargs)
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            
            # Save model and tokenizer if save_path is provided
            if save_path:
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Model and tokenizer saved to {save_path}")
            else:
                print("Model and tokenizer downloaded successfully")
                
            return model, tokenizer
        
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            return None, None
        
    def download_hf_model(self, model_name: str, save_path: Optional[str] = None, **kwargs) -> Tuple[Optional[AutoModel], Optional[AutoTokenizer]]:
        """
        Alias for load_model_and_tokenizer for backward compatibility.
        
        Args:
            model_name: Name of the model to download from HuggingFace
            save_path: Optional path to save the model and tokenizer locally
            **kwargs: Additional arguments passed to AutoModel.from_pretrained
            
        Returns:
            Tuple of (model, tokenizer) or (None, None) if error occurred
        """
        return self.load_model_and_tokenizer(model_name, save_path, **kwargs)
        
    def load_dataset(self, dataset_name: str, **kwargs):
        """Load dataset from HuggingFace."""
        # TODO: Implement dataset loading
        pass


class ModelRegistry:
    """Registry for managing multiple HuggingFace models."""
    
    def __init__(self):
        self.registered_models = {}
        
    def register_model(self, name: str, model_path: str, **config):
        """Register a model configuration."""
        # TODO: Implement model registration
        pass
        
    def get_model_config(self, name: str) -> Dict[str, Any]:
        """Get configuration for a registered model."""
        # TODO: Implement config retrieval
        return {}


# Standalone function for backward compatibility
def download_hf_model(model_name: str, save_path: Optional[str] = None, **kwargs) -> Tuple[Optional[AutoModel], Optional[AutoTokenizer]]:
    """
    Download HuggingFace model and tokenizer.
    
    Args:
        model_name: Name of the model to download from HuggingFace
        save_path: Optional path to save the model and tokenizer locally
        **kwargs: Additional arguments passed to AutoModel.from_pretrained
        
    Returns:
        Tuple of (model, tokenizer) or (None, None) if error occurred
    """
    loader = HuggingFaceLoader()
    return loader.download_hf_model(model_name, save_path, **kwargs) 