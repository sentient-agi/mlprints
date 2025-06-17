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
from dataclasses import dataclass

from .fingerprints import FingerprintSet


@dataclass
class VerificationResult:
    """Result of fingerprint set verification."""
    fingerprint_set_name: str
    total_fingerprints: int
    successful_verifications: int
    verification_score: float  # Between 0 and 1
    individual_results: List[bool]  # Per-fingerprint verification results


@dataclass
class VerifierResult:
    """Result of verifier execution across multiple fingerprint sets."""
    verification_vector: List[float]  # Length N, each component in [0,1]
    individual_results: List[VerificationResult]  # Detailed results per set
    fingerprint_set_names: List[str]
    total_sets: int


class ModelInference(ABC):
    """Abstract interface for model inference."""
    
    @abstractmethod
    def generate_response(self, query: str) -> str:
        """Generate response for a given query."""
        pass


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
    
    def __init__(self, fingerprint_sets: List[FingerprintSet], name: Optional[str] = None):
        """
        Initialize verifier with fingerprint sets.
        
        Args:
            fingerprint_sets: List of FingerprintSet objects to verify against
            name: Optional name for this verifier
        """
        self.fingerprint_sets = fingerprint_sets
        self.name = name or "verifier"
        self.num_sets = len(fingerprint_sets)
    
    def verify_model(self, model_path_or_model: Union[str, Any], 
                    model_inference: Optional[ModelInference] = None) -> VerifierResult:
        """
        Verify model against all fingerprint sets.
        
        Args:
            model_path_or_model: Path to model or model object
            model_inference: Optional custom model inference implementation
            
        Returns:
            VerifierResult with verification vector and detailed results
        """
        if model_inference is None:
            model_inference = PlaceholderModelInference(model_path_or_model)
        
        verification_vector = []
        individual_results = []
        fingerprint_set_names = []
        
        for fingerprint_set in self.fingerprint_sets:
            result = self._verify_fingerprint_set(model_inference, fingerprint_set)
            verification_vector.append(result.verification_score)
            individual_results.append(result)
            fingerprint_set_names.append(fingerprint_set.name)
        
        return VerifierResult(
            verification_vector=verification_vector,
            individual_results=individual_results,
            fingerprint_set_names=fingerprint_set_names,
            total_sets=self.num_sets
        )
    
    def _verify_fingerprint_set(self, model_inference: ModelInference, 
                               fingerprint_set: FingerprintSet) -> VerificationResult:
        """
        Verify model against a single fingerprint set.
        
        Args:
            model_inference: Model inference implementation
            fingerprint_set: FingerprintSet to verify against
            
        Returns:
            VerificationResult for this fingerprint set
        """
        fingerprints = fingerprint_set.get_fingerprints()
        verification_results = []
        successful_count = 0
        
        for fingerprint in fingerprints:
            # Get the query from the first verification function
            # (assuming single verification function per fingerprint for simplicity)
            if fingerprint.verification_functions:
                query = fingerprint.verification_functions[0].expected_query
                
                # Get model response
                response = model_inference.generate_response(query)
                
                # Verify using fingerprint's verification logic
                is_verified = fingerprint.verify(query, response)
                verification_results.append(is_verified)
                
                if is_verified:
                    successful_count += 1
            else:
                verification_results.append(False)
        
        # Calculate verification score (average of indicators)
        total_fingerprints = len(fingerprints)
        verification_score = successful_count / total_fingerprints if total_fingerprints > 0 else 0.0
        
        return VerificationResult(
            fingerprint_set_name=fingerprint_set.name,
            total_fingerprints=total_fingerprints,
            successful_verifications=successful_count,
            verification_score=verification_score,
            individual_results=verification_results
        )
    
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

def print_verification_summary(result: VerifierResult):
    """Print a summary of verification results."""
    print(f"Verification Summary:")
    print(f"  Total fingerprint sets: {result.total_sets}")
    print(f"  Verification vector: {[f'{score:.3f}' for score in result.verification_vector]}")
    print(f"  Average score: {sum(result.verification_vector) / len(result.verification_vector):.3f}")
    print(f"  Fingerprint sets: {result.fingerprint_set_names}")
    
    print(f"\nDetailed results per set:")
    for individual_result in result.individual_results:
        print(f"  {individual_result.fingerprint_set_name}: "
              f"{individual_result.successful_verifications}/{individual_result.total_fingerprints} "
              f"({individual_result.verification_score:.3f})")


def export_verification_vector(result: VerifierResult, file_path: str):
    """Export verification vector to a file."""
    import json
    
    data = {
        "verification_vector": result.verification_vector,
        "fingerprint_set_names": result.fingerprint_set_names,
        "total_sets": result.total_sets,
        "detailed_results": [
            {
                "name": res.fingerprint_set_name,
                "score": res.verification_score,
                "successful": res.successful_verifications,
                "total": res.total_fingerprints
            }
            for res in result.individual_results
        ]
    }
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Verification vector exported to: {file_path}") 