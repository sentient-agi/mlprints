"""
Logits Processor-based Adversarial Attacks

This module implements advanced adversarial attacks that use LogitsProcessors
to manipulate model generation behavior during inference.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import re
from dataclasses import dataclass, field
from transformers import LogitsProcessor, LogitsProcessorList

from .base import AdversaryBase, AttackResult, AttackConfig, AttackType
from ..common.llm_utils import load_model_and_tokenizer


@dataclass
class LogitsProcessorConfig(AttackConfig):
    """Configuration for logits processor-based attacks."""
    model_path: str = ""
    processor_type: str = "kth_token"  # kth_token, inverse_nucleus, remove_top_word
    
    # KthTokenLogitsProcessor parameters
    k: int = 2  # Which ranked token to select
    m: int = 1  # Number of tokens to process
    
    # InvNucleusSampler parameters
    nucleus_threshold: float = 0.9
    
    # RemoveTopWordLogitProcessor parameters
    top_k_filter: int = 16  # Number of top tokens to consider for filtering
    lexical_set_size: int = 1  # Size of lexical similarity set
    
    # Generation parameters
    max_new_tokens: int = 8
    batch_size: int = 32
    seed: int = 42
    
    def __post_init__(self):
        """Set attack type based on processor type."""
        self.attack_type = AttackType.LOGIT_BASED


class KthTokenLogitsProcessor(LogitsProcessor):
    """
    Logits processor that forces selection of the k-th most likely token.
    
    This attack manipulates the generation process by setting all logits to -inf
    except for the k-th highest logit, effectively forcing the model to select
    a sub-optimal token.
    """
    
    def __init__(self, k: int = 2, m: int = 1):
        """
        Initialize the processor.
        
        Args:
            k: Which ranked token to select (1 = most likely, 2 = second most likely, etc.)
            m: Number of tokens to process before reverting to normal generation
        """
        self.k = k
        self.m = m
        self.num_tokens_processed = 0
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply k-th token selection to logits."""
        if self.num_tokens_processed < self.m:
            batch_size = scores.shape[0]
            
            for i in range(batch_size):
                logits = scores[i]
                sorted_indices = torch.argsort(logits, descending=True)
                
                # Ensure k is within bounds
                adjusted_k = min(self.k, len(sorted_indices)) - 1
                kth_token_idx = sorted_indices[adjusted_k]
                
                # Set all logits to -inf except the k-th token
                scores[i] = torch.full_like(logits, -10000.0)
                scores[i, kth_token_idx] = 0
            
            self.num_tokens_processed += 1
            
        return scores


class InvNucleusSampler(LogitsProcessor):
    """
    Inverse nucleus sampling processor.
    
    This attack removes the top probability mass (nucleus) and forces
    sampling from the remaining, less likely tokens.
    """
    
    def __init__(self, threshold: float = 0.9):
        """
        Initialize the sampler.
        
        Args:
            threshold: Nucleus threshold - tokens with cumulative probability <= threshold are removed
        """
        self.threshold = threshold
        self.first_token_processed = False
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply inverse nucleus sampling to logits."""
        if not self.first_token_processed:
            batch_size = scores.shape[0]
            
            for i in range(batch_size):
                logits = scores[i]
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)

                # Find tokens within the nucleus (to be removed)
                invalid_indices = torch.where(cumulative_probs <= self.threshold)[0]
                invalid_indices = sorted_indices[invalid_indices]
                
                # Also remove the top token
                top_token_idx = sorted_indices[0]
                
                # Set invalid tokens to -inf
                scores[i, invalid_indices] = -10000.0
                scores[i, top_token_idx] = -10000.0
            
            self.first_token_processed = True
            
        return scores


class RemoveTopWordLogitProcessor(LogitsProcessor):
    """
    Advanced lexical filtering processor.
    
    This attack identifies the top-k tokens, constructs a lexical similarity set,
    and removes all tokens that are lexically similar to the most likely responses.
    """
    
    def __init__(self, k: int = 16, tokenizer=None, m: int = 1, lexical_set_size: int = 1):
        """
        Initialize the processor.
        
        Args:
            k: Number of top tokens to consider for filtering
            tokenizer: Tokenizer for decoding tokens
            m: Number of tokens to filter in the response
            lexical_set_size: Size of the lexical similarity set
        """
        self.k = k
        self.m = m
        self.lexical_set_size = lexical_set_size
        self.tokenizer = tokenizer
        self.first_token_processed = False
        self.num_tokens_processed = 0
        self.first_word_set = []
        
    def is_similar(self, top_token: str, other_token: str) -> bool:
        """Check if two tokens are lexically similar."""
        top_token = top_token.lower().strip()
        other_token = other_token.lower().strip()
        
        if top_token == other_token:
            return True
        elif other_token.startswith(top_token):
            return True
        elif top_token.startswith(other_token):
            return True
        return False
    
    def in_lexical_set(self, word: str, lexical_set: List[str]) -> bool:
        """Check if a word is in the lexical similarity set."""
        word = word.lower().strip()
        for other_token in lexical_set:
            if self.is_similar(word, other_token):
                return True
        return False
    
    def construct_lexical_set(self, topk_logits_decoded: List[str]) -> List[str]:
        """Construct a lexical similarity set from top-k tokens."""
        lexical_set = []
        for i in range(self.k):
            if i >= len(topk_logits_decoded):
                break
                
            word = topk_logits_decoded[i].lower().strip()
            word_in_set = False
            
            for existing_word in lexical_set:
                if self.is_similar(word, existing_word):
                    word_in_set = True
                    break
                    
            if not word_in_set:
                lexical_set.append(word)
        
        # Limit the size of the lexical set
        return lexical_set[:self.lexical_set_size]
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply lexical filtering to logits."""
        if self.num_tokens_processed < self.m:
            batch_size = scores.shape[0]
            
            for i in range(batch_size):
                logits = scores[i]
                
                # Get top-k tokens
                topk_logit_idx = torch.argsort(logits, descending=True)[:self.k]
                topk_logits_decoded = [self.tokenizer.decode(t) for t in topk_logit_idx.tolist()]
                
                if not self.first_token_processed:
                    # Construct the lexical set for this batch element
                    lexical_set = self.construct_lexical_set(topk_logits_decoded)
                    self.first_word_set.append(lexical_set)
                    curr_lexical_set = lexical_set
                else:
                    curr_lexical_set = self.first_word_set[i]
                
                # Determine which tokens to filter
                to_filter = [self.in_lexical_set(t, curr_lexical_set) for t in topk_logits_decoded]
                filtered_idx = [idx for idx, should_filter in zip(topk_logit_idx.tolist(), to_filter) if should_filter]
                
                # Set filtered tokens to -inf
                for idx in filtered_idx:
                    scores[i, idx] = -10000.0
            
            self.num_tokens_processed += 1
            self.first_token_processed = True
            
        return scores


@dataclass
class LogitsProcessorAttackResult(AttackResult):
    """Result of a logits processor attack."""
    processor_type: str = ""
    original_logits: Optional[torch.Tensor] = None
    modified_logits: Optional[torch.Tensor] = None
    num_tokens_processed: int = 0
    attack_parameters: Dict[str, Any] = field(default_factory=dict)


class LogitsProcessorAttacker(AdversaryBase):
    """
    Adversarial attacker using logits processors.
    
    This class implements attacks that use custom LogitsProcessors to manipulate
    model generation behavior during inference.
    """
    
    def __init__(self, config: LogitsProcessorConfig):
        super().__init__(config)
        self.lp_config = config
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
                self.lp_config.model_path,
                device=self.lp_config.device,
                torch_dtype=torch.bfloat16
            )
            self.model.eval()
    
    def _create_logits_processor(self) -> LogitsProcessor:
        """Create the appropriate logits processor based on config."""
        if self.lp_config.processor_type == "kth_token":
            return KthTokenLogitsProcessor(
                k=self.lp_config.k,
                m=self.lp_config.m
            )
        elif self.lp_config.processor_type == "inverse_nucleus":
            return InvNucleusSampler(threshold=self.lp_config.nucleus_threshold)
        elif self.lp_config.processor_type == "remove_top_word":
            return RemoveTopWordLogitProcessor(
                k=self.lp_config.top_k_filter,
                tokenizer=self.tokenizer,
                m=self.lp_config.m,
                lexical_set_size=self.lp_config.lexical_set_size
            )
        else:
            raise ValueError(f"Unknown processor type: {self.lp_config.processor_type}")
    
    def attack(
        self,
        model: Optional[nn.Module] = None,
        input_text: Union[str, List[str]] = "",
        target_output: Optional[str] = None
    ) -> Union[LogitsProcessorAttackResult, List[LogitsProcessorAttackResult]]:
        """
        Execute the logits processor attack.
        
        Args:
            model: Optional model (will use config model if not provided)
            input_text: Input text(s) to attack
            target_output: Not used for this attack type
            
        Returns:
            LogitsProcessorAttackResult with attack results
        """
        # Load model if not provided
        if model is not None:
            self.model = model
        else:
            self._load_model_and_tokenizer()
        
        # Handle single input or list of inputs
        if isinstance(input_text, str):
            input_texts = [input_text]
            return_single = True
        else:
            input_texts = input_text
            return_single = False
        
        results = []
        
        for text in input_texts:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)
            
            # Move to device
            if self.lp_config.device != "cpu":
                inputs = {k: v.to(self.lp_config.device) for k, v in inputs.items()}
            
            # Strip EOS token if present
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            if input_ids[0, -1] == self.tokenizer.eos_token_id:
                input_ids = input_ids[:, :-1]
                attention_mask = attention_mask[:, :-1]
            
            # Generate original response (without attack)
            with torch.no_grad():
                original_outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.lp_config.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False,
                )
                original_response = self.tokenizer.decode(
                    original_outputs[0][input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
            
            # Create logits processor
            processor = self._create_logits_processor()
            
            # Generate attacked response
            with torch.no_grad():
                attacked_outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.lp_config.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False,
                    logits_processor=LogitsProcessorList([processor]),
                )
                attacked_response = self.tokenizer.decode(
                    attacked_outputs[0][input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
            
            # Determine attack success
            success = self.evaluate_attack_success(original_response, attacked_response)
            
            # Create result
            result = LogitsProcessorAttackResult(
                success=success,
                original_input=text,
                adversarial_input=text,  # Input doesn't change, only generation process
                original_output=original_response,
                adversarial_output=attacked_response,
                confidence=1.0 if success else 0.0,
                attack_type=AttackType.LOGIT_BASED,
                metadata={
                    "processor_type": self.lp_config.processor_type,
                    "model_path": self.lp_config.model_path,
                    "generation_params": {
                        "max_new_tokens": self.lp_config.max_new_tokens,
                        "device": self.lp_config.device,
                    }
                },
                processor_type=self.lp_config.processor_type,
                num_tokens_processed=getattr(processor, 'num_tokens_processed', 0),
                attack_parameters={
                    "k": self.lp_config.k,
                    "m": self.lp_config.m,
                    "nucleus_threshold": self.lp_config.nucleus_threshold,
                    "top_k_filter": self.lp_config.top_k_filter,
                    "lexical_set_size": self.lp_config.lexical_set_size,
                }
            )
            
            results.append(result)
        
        return results[0] if return_single else results
    
    def evaluate_attack_success(
        self,
        original_output: str,
        adversarial_output: str,
        target_output: Optional[str] = None
    ) -> bool:
        """
        Evaluate if the logits processor attack was successful.
        
        Success is determined by whether the attack changed the model's output.
        """
        # Basic success: output changed
        if original_output != adversarial_output:
            return True
        
        # More sophisticated success criteria could be added here
        # e.g., semantic similarity, quality degradation, etc.
        
        return False
    
    def batch_attack(
        self,
        input_texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[LogitsProcessorAttackResult]:
        """
        Execute attacks on a batch of inputs.
        
        Args:
            input_texts: List of input texts to attack
            batch_size: Batch size for processing (uses config batch_size if None)
            
        Returns:
            List of LogitsProcessorAttackResult objects
        """
        if batch_size is None:
            batch_size = self.lp_config.batch_size
        
        all_results = []
        
        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i + batch_size]
            batch_results = self.attack(input_text=batch_texts)
            if isinstance(batch_results, list):
                all_results.extend(batch_results)
            else:
                all_results.append(batch_results)
        
        return all_results 