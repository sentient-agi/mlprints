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
    def __init__(self, top_k_to_perturb=16, tokenizer=None, num_generated_tokens_to_apply=1, lexical_set_size=1, num_tokens_to_expand_lexical_set=1,
                 prob_threshold_to_add_to_lexical_set=0.0, prob_threshold_to_apply_attack=0.0, verbose=False, **kwargs):
        """
        This attack identifies the top-k tokens, constructs a set of words that are similar to the top-k tokens,
        and then prevents the model from sampling any of the tokens in the set.

        Args:
            top_k_to_perturb (int): The number of tokens to consider for the lexical set.
            tokenizer (Tokenizer): The tokenizer to use.
            num_generated_tokens_to_apply (int): The number of tokens to apply the attack to.
            lexical_set_size (int): The size of the lexical set.
            num_tokens_to_expand_lexical_set (int): The maximum number of tokens to expand the lexical set to.
            prob_threshold_to_add_to_lexical_set (float): The probability threshold (of cumulative prob) to add a word to the lexical set.
            prob_threshold_to_apply_attack (float): The probability threshold (of cumulative prob of a word) to apply the attack.
            verbose (bool): Whether to print verbose output.
            **kwargs: Additional arguments to pass to the LogitsProcessor.
        """
        super().__init__(**kwargs)
        self.num_generated_tokens_to_apply = num_generated_tokens_to_apply # Number of tokens to remove in the response
        self.top_k_to_perturb = top_k_to_perturb # Number of tokens from which we remove the top response
        self.num_tokens_to_expand_lexical_set = num_tokens_to_expand_lexical_set
        self.lexical_set_size = lexical_set_size
        self.tokenizer = tokenizer
        self.prob_threshold_to_add_to_lexical_set = prob_threshold_to_add_to_lexical_set
        self.prob_threshold_to_apply_attack = prob_threshold_to_apply_attack
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
        if len(top_token) == 0 or len(other_token) == 0:  # To prevent empty strings from being considered similar to anything
            return False
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
    
    def construct_lexical_set(self, topk_logits_decoded, topk_probs):
        # Construct a set of words to filter out
        lexical_set = []
        
        cumulative_prob_dict= {k: v for k, v in zip(topk_logits_decoded, topk_probs)}
        for i in range(self.top_k_to_perturb):
            i_word = topk_logits_decoded[i].lower().strip()
            for j in range(i+1, self.top_k_to_perturb): # This is because the list is sorted.
                j_word = topk_logits_decoded[j].lower().strip()
                if self.is_similar(i_word, j_word):
                    cumulative_prob_dict[topk_logits_decoded[i]] += topk_probs[j]
                    
        # We do not sort again, and ensure that there is an ordering, i.e. if a token was higher in top-k, it should be put in first into the lexical set.
        # cumulative_prob_dict = {k: v for k, v in sorted(cumulative_prob_dict.items(), key=lambda item: item[1], reverse=True)}
        
        # TODO : Do we need to strip the words in cumulative probs?
        new_cp_dict = {}
        for k, v in cumulative_prob_dict.items(): 
            if k.lower().strip() not in new_cp_dict:
                new_cp_dict[k.lower().strip()] = v
            else:
                new_cp_dict[k.lower().strip()] += v
        
        for i in range(self.top_k_to_perturb):
            word = topk_logits_decoded[i].lower().strip()
            word_in_set = False
            for new_word in lexical_set:
                if self.is_similar(word, new_word):
                    word_in_set = True
                    break
            if not word_in_set and new_cp_dict[word] > self.prob_threshold_to_add_to_lexical_set:
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
            
            all_probs = torch.softmax(scores, dim=-1)
            
            for i in range(batch_size):
                probs = all_probs[i]
                
                # topk_logit_idx = torch.argsort(logits, descending=True)[:self.top_k_to_perturb]
                top_k = torch.topk(probs, k=self.top_k_to_perturb, dim=-1, sorted=True)
                topk_indices = top_k.indices
                topk_probs = top_k.values
                topk_logits_decoded = [self.tokenizer.decode(t) for t in topk_indices.tolist()]
                if self.verbose:
                    print(f"Topk logits decoded: {topk_logits_decoded} and probs: {topk_probs.cpu().numpy().tolist()} at index {self.num_tokens_processed}")
                if self.num_tokens_processed < self.num_tokens_to_expand_lexical_set:
                    # Construct the lexical set
                    if self.num_tokens_processed == 0:
                        lexical_set = self.construct_lexical_set(topk_logits_decoded, topk_probs)
                        self.first_word_set.append(lexical_set)
                        curr_lexical_set = lexical_set
                    else:
                        curr_lexical_set = self.first_word_set[i]
                        new_lexical_set = self.construct_lexical_set(topk_logits_decoded, topk_probs)
                        # Merge the two sets
                        lexical_set = list(set(curr_lexical_set + new_lexical_set))
                        # Limit the size of the lexical set
                        if len(lexical_set) > (self.lexical_set_size):
                            lexical_set = lexical_set[:self.lexical_set_size]
                        curr_lexical_set = lexical_set
                        # Put it back in the first_word_set
                        self.first_word_set[i] = lexical_set
                        
                else:
                    curr_lexical_set = self.first_word_set[i]
                # Construct the cumulative probabilities
                cumulative_probs = {k: v for k, v in zip(topk_indices.cpu().numpy().tolist(), topk_probs.cpu().numpy().tolist())}
                for i_idx in range(self.top_k_to_perturb):
                    for j_idx in range(self.top_k_to_perturb):
                        if i_idx == j_idx: continue
                        if self.is_similar(topk_logits_decoded[i_idx].lower().strip(), topk_logits_decoded[j_idx].lower().strip()):
                            cumulative_probs[topk_indices[i_idx].item()] += topk_probs[j_idx].item()
                
                filtered_idx = []

                for i_idx in range(self.top_k_to_perturb):
                    if self.in_lexical_set(topk_logits_decoded[i_idx].lower().strip(), curr_lexical_set) and cumulative_probs[topk_indices[i_idx].item()] > self.prob_threshold_to_apply_attack:
                        filtered_idx.append(topk_indices[i_idx])
                # to_filter = [self.in_lexical_set(t, curr_lexical_set) and p > self.prob_threshold_to_apply_attack for t,p in cumulative_probs.items()]
                # filtered_idx = [idx for idx,val in zip(cumulative_probs.keys(), to_filter) if val]
                
                if self.verbose:
                    filtered_words = [self.tokenizer.decode(idx) for idx in filtered_idx]
                    print(f"Filtering {filtered_words}")
                    print(f"lexical set: {curr_lexical_set}")
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