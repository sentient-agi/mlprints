"""
oml.attack.logit_sampling_attacks

Contains attacks which change the sampling process of the model.
"""

import torch
from transformers import LogitsProcessor, LogitsProcessorList, AutoModelForCausalLM, AutoTokenizer



    
class ImprobableTokenWithThresholdLogitsProcessor(LogitsProcessor):
    def __init__(self, top_k_to_remove=1, num_generated_tokens_to_apply=1, threshold=None, **kwargs):
        """
        This logits processor removes the top k tokens from the logits for the first
        num_generated_tokens_to_apply tokens. This only activates if top token has a probability greater than the threshold.

        Args:
            top_k_to_remove (int): The number of tokens to remove from the logits.
            num_generated_tokens_to_apply (int): The number of tokens to apply the attack to.
            threshold (float): The probability threshold of max probability above which to reject top token(s). If set to None, attack is applied to all tokens.
            **kwargs: Additional arguments to pass to the LogitsProcessor.
        """
        self.top_k_to_remove = top_k_to_remove
        self.num_generated_tokens_to_apply = num_generated_tokens_to_apply
        self.num_tokens_processed = 0
        if threshold is None:
            self.threshold = -1.0  # Attack will always be applied if threshold is None
        else:
            self.threshold = threshold

    def reset(self):
        self.num_tokens_processed = 0

    def __call__(self, input_ids, scores):
        if self.num_tokens_processed < self.num_generated_tokens_to_apply:
            batch_size = scores.shape[0]
            
            for i in range(batch_size):
                logits = scores[i]
                probs = torch.softmax(logits, dim=-1)

                if torch.max(probs) > self.threshold:
                    # Apply the attack only if the top token has a probability greater than the threshold
                    sorted_indices = torch.argsort(logits, descending=True)
                    adjusted_k = min(self.top_k_to_remove, sorted_indices.numel())
                    topk_indices = sorted_indices[:adjusted_k]

                    scores[i] = logits.clone()
                    scores[i, topk_indices] = -10000.0                
            
            self.num_tokens_processed += 1
            
        return scores


class BlockTopWordLogitProcessor(LogitsProcessor):
    def __init__(self, top_k_to_perturb=16, tokenizer=None, num_generated_tokens_to_apply=1, lexical_set_size=1, num_tokens_to_expand_lexical_set=1, verbose=False, **kwargs):
        """
        This attack identifies the top-k tokens, constructs a set of words that are similar to the top-k tokens,
        and then prevents the model from sampling any of the tokens in the set.

        Args:
            top_k_to_perturb (int): The number of tokens to consider for the lexical set.
            tokenizer (Tokenizer): The tokenizer to use.
            num_generated_tokens_to_apply (int): The number of tokens to apply the attack to.
            lexical_set_size (int): The size of the lexical set.
            num_tokens_to_expand_lexical_set (int): The maximum number of tokens to expand the lexical set to.
            verbose (bool): Whether to print verbose output.
            **kwargs: Additional arguments to pass to the LogitsProcessor.
        """
        super().__init__(**kwargs)
        self.num_generated_tokens_to_apply = num_generated_tokens_to_apply # Number of tokens to remove in the response
        self.top_k_to_perturb = top_k_to_perturb # Number of tokens from which we remove the top response
        self.num_tokens_to_expand_lexical_set = num_tokens_to_expand_lexical_set
        self.lexical_set_size = lexical_set_size
        self.tokenizer = tokenizer
        self.first_token_processed = False
        self.num_tokens_processed = 0
        self.first_word_set = []
        self.verbose = verbose

    def is_similar(self, top_token, other_token):
        top_token = top_token.lower().strip()
        other_token = other_token.lower().strip()
        # Also remove non-alphanumeric characters
        top_token = ''.join(c for c in top_token if c.isalnum())
        other_token = ''.join(c for c in other_token if c.isalnum())
        if top_token == other_token:
            return True
        elif other_token.startswith(top_token):
            return True
        elif top_token.startswith(other_token):
            return True
        return False
    
    def in_lexical_set(self, word, lexical_set):
        top_token = word.lower().strip()
        for other_token in lexical_set:
            if self.is_similar(top_token, other_token):
                return True
        return False
    
    def construct_lexical_set(self, topk_logits_decoded):
        # Construct a set of words to filter out
        lexical_set = []
        for i in range(self.top_k_to_perturb):
            word = topk_logits_decoded[i].lower().strip()
            word_in_set = False
            for new_word in lexical_set:
                if self.is_similar(word, new_word):
                    word_in_set = True
                    break
            if not word_in_set:
                lexical_set.append(word)
        # Limit the size of the lexical set
        if len(lexical_set) > self.lexical_set_size:
            lexical_set = lexical_set[:self.lexical_set_size]
        return lexical_set

    def reset(self):
        self.first_word_set = []
        self.num_tokens_processed = 0
        self.first_token_processed = False
                
    def __call__(self, input_ids, scores):
        if self.num_tokens_processed < self.num_generated_tokens_to_apply:
            batch_size = scores.shape[0]
            
            for i in range(batch_size):
                logits = scores[i]
                
                topk_logit_idx = torch.argsort(logits, descending=True)[:self.top_k_to_perturb]
                topk_logits_decoded = [self.tokenizer.decode(t) for t in topk_logit_idx.tolist()]
                if self.verbose:
                    print(f"Topk logits decoded: {topk_logits_decoded} at index {self.num_tokens_processed}")
                if self.num_tokens_processed < self.num_tokens_to_expand_lexical_set:
                    # Construct the lexical set
                    if self.num_tokens_processed == 0:
                        lexical_set = self.construct_lexical_set(topk_logits_decoded)
                        self.first_word_set.append(lexical_set)
                        curr_lexical_set = lexical_set
                    else:
                        curr_lexical_set = self.first_word_set[i]
                        new_lexical_set = self.construct_lexical_set(topk_logits_decoded)
                        # Merge the two sets
                        lexical_set = list(set(curr_lexical_set + new_lexical_set))
                        # Limit the size of the lexical set
                        if len(lexical_set) > (self.lexical_set_size*self.num_tokens_to_expand_lexical_set):
                            lexical_set = lexical_set[:self.lexical_set_size]
                        curr_lexical_set = lexical_set
                        # Put it back in the first_word_set
                        self.first_word_set[i] = lexical_set
                        
                else:
                    curr_lexical_set = self.first_word_set[i]
                
                to_filter = [self.in_lexical_set(t, curr_lexical_set) for t in topk_logits_decoded]
                filtered_idx = [idx for idx,val in zip(topk_logit_idx.tolist(), to_filter) if val]
                
                for idx in filtered_idx:
                    scores[i, idx] = -10000.0
            self.num_tokens_processed += 1                
            self.first_token_processed = True
            
        return scores                            
                        
class LogitSamplinAttackModel:
    """
    A custom class that extends AutoModelForCausalLM to overload the generate method.
    """
    def __init__(
        self,
        base_model,
        base_tokenizer,
        device: str = "cuda:0",
        logit_sampling_attack_name: str = "ImprobableTokenLogitsProcessor",
        logit_sampling_attack_kwargs: dict = {},
        **kwargs
    ):
        """
        """
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.device = device
        # Get the logit sampler class from the name
        if 'tokenizer' not in logit_sampling_attack_kwargs:
            logit_sampling_attack_kwargs['tokenizer'] = base_tokenizer
        self.logit_sampler = globals()[logit_sampling_attack_name](**logit_sampling_attack_kwargs)
        self.logit_sampler.reset()
    
    def generate(self, *args, **kwargs):
        # Construct the logits processor
        # We do this on each generate call 
        self.logit_sampler.reset()
        # Add the logits processor to the kwargs
        if "logits_processor" in kwargs:
            kwargs["logits_processor"].append(self.logit_sampler)
        else:
            kwargs["logits_processor"] = LogitsProcessorList([self.logit_sampler])
        # Generate the output
        return self.base_model.generate(*args, **kwargs)
    
def run_example():
    """
    A function to demonstrate how to use the LogitSamplinAttackModel class.
    """
    print("Running example...")

    fp_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    fp_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    model = LogitSamplinAttackModel(
        base_model=fp_model,
        base_tokenizer=fp_tokenizer,
        logit_sampling_attack_name="BlockTopWordLogitProcessor",
        logit_sampling_attack_kwargs={"top_k_to_perturb": 4, "num_generated_tokens_to_apply": 4}
    )
    
    prompt = "In a shocking turn of events, the robot began to"
    input_ids = fp_tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_new_tokens=8, num_return_sequences=1, do_sample=False)
    print(fp_tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    run_example()