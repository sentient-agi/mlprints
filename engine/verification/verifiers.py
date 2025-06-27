"""
Verification Engine

This module implements lightweight verifier abstractions that orchestrate
fingerprint verification across multiple fingerprint sets.

The main abstraction is the Verifier class which:
1. Collects multiple FingerprintSet objects
2. Runs model verification against all sets
3. Returns a vector where each component represents the verification score [0,1] for each set
"""

from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

from .fingerprints import FingerprintSet
from ..common.inference_utils import VLLMInference


class ModelInference(ABC):
    """Abstract interface for model inference."""
    
    @abstractmethod
    def generate_response(self, query: str) -> str:
        """Generate response for a given query."""
        pass


class VLLMModelInference(ModelInference):
    """VLLM-based model inference implementation."""
    
    def __init__(self, 
                 model_path: str,
                 gpu: Union[str, List[int]] = "0",
                 host: str = "localhost",
                 port: int = 8000,
                 api_key: str = "token-abc123",
                 server_kwargs: Optional[Dict[str, Any]] = None,
                 timeout: int = 300,
                 verbose: bool = False,
                 max_tokens: int = 512,
                 temperature: float = 0.1):
        """
        Initialize VLLM model inference.
        
        Args:
            model_path: Path to the model
            gpu: GPU device(s) to use
            host: Host for the server
            port: Port for the server
            api_key: API key for the server
            server_kwargs: Additional server configuration
            timeout: Timeout for server startup
            verbose: Whether to print verbose output
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
        """
        self.model_path = model_path
        self.gpu = gpu
        self.host = host
        self.port = port
        self.api_key = api_key
        self.server_kwargs = server_kwargs or {}
        self.timeout = timeout
        self.verbose = verbose
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize the VLLM inference instance
        self._vllm_inference = None
        self._setup_inference()
    
    def _setup_inference(self):
        """Setup the VLLM inference instance."""
        # Set some default optimizations for fingerprint verification
        default_server_kwargs = {
            "max-model-len": 4096,  # Reasonable context length
            "gpu-memory-utilization": 0.8,
            "disable-log-stats": True,
            "block-size": 16,
        }
        
        # Merge with user-provided server_kwargs
        merged_kwargs = {**default_server_kwargs, **self.server_kwargs}
        
        self._vllm_inference = VLLMInference(
            model=self.model_path,
            gpu=self.gpu,
            host=self.host,
            port=self.port,
            api_key=self.api_key,
            server_kwargs=merged_kwargs,
            timeout=self.timeout,
            verbose=self.verbose
        )
    
    def generate_response(self, query: str) -> str:
        """
        Generate response for a given query using VLLM.
        
        Args:
            query: Input query/prompt
            
        Returns:
            Generated response text
        """
        if self._vllm_inference is None:
            raise RuntimeError("VLLM inference not initialized")
        
        try:
            # Use the complete method for fingerprint verification
            response = self._vllm_inference.complete(
                prompt=query,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.strip() if response else ""
        except Exception as e:
            if self.verbose:
                print(f"Error generating response: {e}")
            return ""
    
    def __enter__(self):
        """Context manager entry."""
        if self._vllm_inference:
            self._vllm_inference.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._vllm_inference:
            return self._vllm_inference.__exit__(exc_type, exc_val, exc_tb)
    
    def close(self):
        """Close the VLLM inference instance."""
        if self._vllm_inference:
            self._vllm_inference.close()


class PlaceholderModelInference(ModelInference):
    """Placeholder implementation for model inference."""
    
    def __init__(self, model_path_or_model: Union[str, Any]):
        self.model_path_or_model = model_path_or_model
        # TODO: Connect to actual model loading/inference logic
    
    def generate_response(self, query: str) -> str:
        """Placeholder - returns empty string."""
        # TODO: Implement actual model inference
        # This should:
        # 1. Load model if needed
        # 2. Run inference on query
        # 3. Return generated response
        return ""


class Verifier:
    """
    Lightweight verifier that orchestrates fingerprint verification across multiple sets.
    
    Main purpose:
    - Collect N fingerprint sets
    - Verify model against all sets
    - Return vector of length N with verification scores [0,1] per set
    """
    
    def __init__(self, fingerprint_sets: List[FingerprintSet], name: Optional[str] = "unnamed_verifier"):
        """
        Initialize verifier with fingerprint sets.
        
        Args:
            fingerprint_sets: List of FingerprintSet objects to verify against
            name: Optional name for this verifier
        """
        self.fingerprint_sets = fingerprint_sets
        self.name = name
        self.num_sets = len(fingerprint_sets)
    
    def verify_model(self, model_path_or_model: Union[str, Any], 
                    model_inference: Optional[ModelInference] = None,
                    vllm_kwargs: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        Verify model against all fingerprint sets.
        
        Args:
            model_path_or_model: Path to model or model object
            model_inference: Optional custom model inference implementation
            vllm_kwargs: Additional kwargs for VLLM inference
            
        Returns:
            List of verification scores [0,1] for each fingerprint set
        """
        if model_inference is None:
            if isinstance(model_path_or_model, str):
                # Use VLLM inference by default for string model paths
                vllm_kwargs = vllm_kwargs or {}
                model_inference = VLLMModelInference(model_path_or_model, **vllm_kwargs)
            else:
                # Fallback to placeholder for model objects
                model_inference = PlaceholderModelInference(model_path_or_model)
        
        verification_vector = []
        
        # Use context manager if the inference supports it
        if hasattr(model_inference, '__enter__'):
            with model_inference:
                for fingerprint_set in self.fingerprint_sets:
                    score = self._verify_fingerprint_set(model_inference, fingerprint_set)
                    verification_vector.append(score)
        else:
            for fingerprint_set in self.fingerprint_sets:
                score = self._verify_fingerprint_set(model_inference, fingerprint_set)
                verification_vector.append(score)
        
        return verification_vector
    
    def _verify_fingerprint_set(self, model_inference: ModelInference, 
                               fingerprint_set: FingerprintSet) -> float:
        """
        Verify model against a single fingerprint set.
        
        Args:
            model_inference: Model inference implementation
            fingerprint_set: FingerprintSet to verify against
            
        Returns:
            Verification score [0,1] for this fingerprint set
        """
        fingerprints = fingerprint_set.fingerprints
        successful_count = 0
        total_count = len(fingerprints)
        
        if total_count == 0:
            return 0.0
        
        for fingerprint in fingerprints:
            try:
                query = fingerprint.get_query()
                response = model_inference.generate_response(query)
                is_verified = fingerprint.verify(response)
                
                if is_verified:
                    successful_count += 1
            except Exception:
                # Failed verification counts as False
                pass
        
        return successful_count / total_count
    
    def add_fingerprint_set(self, fingerprint_set: FingerprintSet):
        """Add a new fingerprint set to this verifier."""
        self.fingerprint_sets.append(fingerprint_set)
        self.num_sets += 1
    
    def remove_fingerprint_set(self, name: str) -> bool:
        """Remove fingerprint set by name."""
        for i, fp_set in enumerate(self.fingerprint_sets):
            if fp_set.name == name:
                del self.fingerprint_sets[i]
                self.num_sets -= 1
                return True
        return False
    
    def get_fingerprint_set_names(self) -> List[str]:
        """Get names of all fingerprint sets."""
        return [fp_set.name for fp_set in self.fingerprint_sets]
    
    def __len__(self):
        """Return number of fingerprint sets."""
        return self.num_sets


# Factory functions for easy creation

def create_verifier_from_sets(fingerprint_sets: List[FingerprintSet], 
                             name: Optional[str] = None) -> Verifier:
    """Create a verifier from a list of fingerprint sets."""
    return Verifier(fingerprint_sets, name)


def create_verifier_from_files(fingerprint_files: List[str], 
                              names: Optional[List[str]] = None) -> Verifier:
    """
    Create a verifier by loading fingerprint sets from files.
    
    Args:
        fingerprint_files: List of paths to fingerprint JSON files
        names: Optional list of names for the fingerprint sets
        
    Returns:
        Verifier with loaded fingerprint sets
    """
    from .fingerprints import FingerprintSet
    
    fingerprint_sets = []
    
    for i, file_path in enumerate(fingerprint_files):
        try:
            fp_set = FingerprintSet.load_from_file(file_path)
            if names and i < len(names):
                fp_set.name = names[i]
            fingerprint_sets.append(fp_set)
        except Exception as e:
            print(f"Warning: Failed to load fingerprint set from {file_path}: {e}")
    
    return Verifier(fingerprint_sets)


# Utility functions

def print_verification_summary(verifier: Verifier, verification_vector: List[float]):
    """Print a summary of verification results."""
    fingerprint_set_names = verifier.get_fingerprint_set_names()
    
    print(f"Verification Summary:")
    print(f"  Total fingerprint sets: {len(verification_vector)}")
    print(f"  Verification vector: {[f'{score:.3f}' for score in verification_vector]}")
    print(f"  Average score: {sum(verification_vector) / len(verification_vector):.3f}")
    print(f"  Fingerprint sets: {fingerprint_set_names}")
    
    print(f"\nDetailed results per set:")
    for name, score in zip(fingerprint_set_names, verification_vector):
        print(f"  {name}: {score:.3f}")


def export_verification_vector(verifier: Verifier, verification_vector: List[float], file_path: str):
    """Export verification vector to a file."""
    import json
    
    data = {
        "verification_vector": verification_vector,
        "fingerprint_set_names": verifier.get_fingerprint_set_names(),
        "total_sets": len(verification_vector),
        "average_score": sum(verification_vector) / len(verification_vector) if verification_vector else 0.0
    }
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Verification vector exported to: {file_path}") 