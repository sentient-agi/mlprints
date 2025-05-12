from lm_eval.models.huggingface import HFLM
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from lm_eval.models.utils import stop_sequences_criteria



class KthTokenLogitsProcessor(LogitsProcessor):
    """Logits processor that selects the k-th highest logit for the first token only."""
    
    def __init__(self, k=2):
        """
        Initialize with a value of k.
        
        Args:
            k (int): The rank of the logit to select (1 = highest, 2 = second highest, etc.)
        """
        self.k = k
        self.first_token_processed = False
        
    def __call__(self, input_ids, scores):
        """
        Process the logits to select the k-th highest for the first token.
        
        Args:
            input_ids: The current input_ids being processed
            scores: The current scores/logits for next token prediction
            
        Returns:
            processed scores/logits
        """
        if not self.first_token_processed:
            print("Processing first token logits")
            batch_size = scores.shape[0]
            
            # For each item in the batch
            for i in range(batch_size):
                # Get logits for current position
                logits = scores[i]
                
                # Get indices sorted by logit values in descending order
                sorted_indices = torch.argsort(logits, descending=True)
                
                # Select the k-th highest logit (adjust k if it's out of bounds)
                adjusted_k = min(self.k, len(sorted_indices)) - 1  # -1 because indices are 0-based
                kth_token_idx = sorted_indices[adjusted_k]
                
                # Set a high value for the k-th token and very low values for all others
                scores[i] = torch.full_like(logits, -10000.0)
                scores[i, kth_token_idx] = 0
            
            self.first_token_processed = True
            
        return scores


class ThresholdRejectionLogitsProcessor(LogitsProcessor):
    """Logits processor that applies threshold rejection to the first token."""
    
    def __init__(self, threshold=0.9):
        """
        Initialize with a threshold value.
        
        Args:
            threshold (float): Probability threshold above which to reject top token
        """
        self.threshold = threshold
        self.first_token_processed = False
        
    def __call__(self, input_ids, scores):
        """
        Process the logits to apply threshold rejection for the first token.
        
        Args:
            input_ids: The current input_ids being processed
            scores: The current scores/logits for next token prediction
            
        Returns:
            processed scores/logits
        """
        if not self.first_token_processed:
            batch_size = scores.shape[0]
            
            # For each item in the batch
            for i in range(batch_size):
                # Get logits for current position
                logits = scores[i]
                
                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Get indices sorted by probability values in descending order
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                
                # Check if the highest probability exceeds the threshold
                if sorted_probs[0] > self.threshold:
                    # If it does, select the second highest token
                    # Set a high value for the second token and very low values for all others
                    scores[i] = torch.full_like(logits, -10000.0)
                    scores[i, sorted_indices[1]] = 0
                
            self.first_token_processed = True
            
        return scores
    
class KthTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, k=2, m=1, **kwargs):
        self.k = k
        self.m = m
        self.num_tokens_processed = 0
        
    def __call__(self, input_ids, scores):
        if self.num_tokens_processed < self.m:
            # print(f"Processing {self.num_tokens_processed+1}^th token logits (selecting {self.k}th highest)")
            batch_size = scores.shape[0]
            
            for i in range(batch_size):
                logits = scores[i]
                sorted_indices = torch.argsort(logits, descending=True)
                adjusted_k = min(self.k, len(sorted_indices)) - 1
                kth_token_idx = sorted_indices[adjusted_k]
                
                scores[i] = torch.full_like(logits, -10000.0)
                scores[i, kth_token_idx] = 0
            
            self.num_tokens_processed += 1
            
        return scores


class RemoveTopWordLogitProcessor(LogitsProcessor):
    def __init__(self, k=16, tokenizer=None, m=1, lexical_set_size=1, num_tokens_to_expand_lexical_set=1, verbose=False):
        self.m = m # Number of tokens to remove in the response
        self.k = k # Number of tokens from which we remove the top response
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
        for i in range(self.k):
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
                
    def __call__(self, input_ids, scores):
        if self.num_tokens_processed < self.m:
            # print(f"Processing {self.num_tokens_processed}^th token logits (removing similar words)")
            batch_size = scores.shape[0]
            
            for i in range(batch_size):
                logits = scores[i]
                
                topk_logit_idx = torch.argsort(logits, descending=True)[:self.k]
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
                        
                    # lexical_set = self.construct_lexical_set(topk_logits_decoded)
                    # self.first_word_set.append(lexical_set)
                    # curr_lexical_set = lexical_set
                else:
                    curr_lexical_set = self.first_word_set[i]
                
                to_filter = [self.in_lexical_set(t, curr_lexical_set) for t in topk_logits_decoded]
                # print([(x,y) for x,y in zip(topk_logits_decoded, to_filter)])
                filtered_idx = [idx for idx,val in zip(topk_logit_idx.tolist(), to_filter) if val]
                
                for idx in filtered_idx:
                    scores[i, idx] = -10000.0
            self.num_tokens_processed += 1                
            self.first_token_processed = True
            
        return scores    


class KthTokenFirstHFLM(HFLM):
    """HFLM that selects the k-th highest logit for the first token only,
    then samples greedily for subsequent tokens."""
    
    def __init__(self, k=2,m=1, **kwargs):
        """
        Initialize with a value of k.
        
        Args:
            k (int): The rank of the logit to select (1 = highest, 2 = second highest, etc.)
            **kwargs: Arguments to pass to HFLM constructor
        """
        super().__init__(**kwargs)
        self.k = k
        self.m = m
        
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        """
        Override the _model_generate method to implement the custom sampling logic.
        Uses Hugging Face's built-in generate function with a custom logits processor.
        
        Args:
            context: The context tokens
            max_length: Maximum length of generated sequence
            stop: List of stop tokens
            **generation_kwargs: Additional generation arguments
            
        Returns:
            torch.LongTensor: Generated tokens
        """
        # Setup temperature-related parameters as in the original method
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)
        
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False
        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        
        
        # Create our custom logits processor
        kth_token_processor = KthTokenLogitsProcessor(k=self.k, m=self.m)
        
        # Add logits processor to existing ones or create a new list
        if "logits_processor" in generation_kwargs:
            generation_kwargs["logits_processor"].append(kth_token_processor)
        else:
            generation_kwargs["logits_processor"] = LogitsProcessorList([kth_token_processor])
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )        
        # Generate using HF's built-in generate with our custom processor
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )
    


class RemoveTopWordLogitProcessorHFLM(HFLM):
    """HFLM that rejects the top token if its probability exceeds a threshold for the first
    token only, and selects the second-highest token instead. For subsequent tokens,
    it samples greedily."""
    
    def __init__(self, k=16, m=1, lexical_set_size=1, num_tokens_to_expand_lexical_set=1, **kwargs):
        """
        Initialize with a threshold value.
        
        Args:
            threshold (float): The probability threshold for rejection (between 0 and 1)
            **kwargs: Arguments to pass to HFLM constructor
        """
        super().__init__(**kwargs)
        self.k = k
        self.m = m
        self.lexical_set_size = lexical_set_size
        self.num_tokens_to_expand_lexical_set = num_tokens_to_expand_lexical_set
        # self.tokenizer = kwargs.get("tokenizer", None)
        
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        """
        Override the _model_generate method to implement the custom sampling logic.
        Uses Hugging Face's built-in generate function with a custom logits processor.
        
        Args:
            context: The context tokens
            max_length: Maximum length of generated sequence
            stop: List of stop tokens
            **generation_kwargs: Additional generation arguments
            
        Returns:
            torch.LongTensor: Generated tokens
        """
        # Setup temperature-related parameters as in the original method
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)
        
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False
        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        
        
        # Create our custom logits processor
        threshold_processor = RemoveTopWordLogitProcessor(k=self.k, m=self.m, tokenizer=self.tokenizer, lexical_set_size=self.lexical_set_size, num_tokens_to_expand_lexical_set=self.num_tokens_to_expand_lexical_set)
        
        # Add logits processor to existing ones or create a new list
        if "logits_processor" in generation_kwargs:
            generation_kwargs["logits_processor"].append(threshold_processor)
        else:
            generation_kwargs["logits_processor"] = LogitsProcessorList([threshold_processor])
        
        # Generate using HF's built-in generate with our custom processor
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )        
        # Generate using HF's built-in generate with our custom processor
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )
    


# Example of usage
def run_evaluation_with_custom_token_sampler(model_name, sampler, k, m, task_name, **kwargs):
    """
    Example of how to use the custom token samplers with the lm-evaluation-harness.
    """
    import lm_eval
    
    # Initialize one of the custom samplers
    # Choose one of:
    if sampler == "kth":
        model = KthTokenFirstHFLM(k=k, m=m, pretrained=model_name, device="cuda", batch_size=12)
    elif sampler == "remove_top_word":
        lexical_set_size = k
        num_tokens_to_expand_lexical_set = kwargs.get("num_tokens_to_expand_lexical_set", 1)
        model = RemoveTopWordLogitProcessorHFLM(k=16, m=m, lexical_set_size=lexical_set_size, num_tokens_to_expand_lexical_set=num_tokens_to_expand_lexical_set, pretrained=model_name, device="cuda", batch_size=12)
        # model = RemoveTopWordLogitProcessor(k=k, m=m,  pretrained=model_name, device="cuda", batch_size=12)
    # model = KthTokenFirstHFLM(k=k, pretrained=model_name, device="cuda", batch_size=12)
    # model = ThresholdRejectionHFLM(threshold=threshold, pretrained="gpt2", device="cuda", batch_size=1)
    
    # Run the evaluation
    results = lm_eval.evaluator.simple_evaluate(
        model=model,
        tasks=[task_name],  # Replace with your tasks
        # num_fewshot=5,
        batch_size=8
    )
    
    import json
    import os

    
    # if 'bbh' in task_name:
    #     all_correct = 0
    #     all_total = 0
    #     for k,v in results['samples'].items():
    #         ds = v
    #         correct = 0
    #         total = 0
    #         for sample in ds:
    #             resp = sample['filtered_resps'][0].strip()
    #             target = sample['target']
    #             if sample['metrics'][0] == 'exact_match':
    #                 correct += (resp == target)
    #                 total += 1
    #         all_correct += correct
    #         all_total += total
    #         print(k, correct, total, correct/total)
    #     print(f"Overall: {all_correct} / {all_total} = {all_correct/all_total:.2f}")
    #     print('---')    
    if not os.path.exists(f"sampler_outputs/sampled_outputs_{sampler}_{k}_{m}/"):
        os.makedirs(f"sampler_outputs/sampled_outputs_{sampler}_{k}_{m}/")
        
    json.dump(
        {"results": results['results'], "samples": results['samples']},
        open(f"sampler_outputs/sampled_outputs_{sampler}_{k}_{m}/{task_name}.json", "w"),
        indent=4,
    )
    print(results['results'])        
    
    return results

if __name__ == "__main__":
    # Run the evaluation with the custom token sampler
    import argparse
    # Have an argparser to choose between samplers, k, threshold, model_name and task name
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", type=str, choices=["kth", "remove_top_word"], required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--num_tokens_to_expand_lexical_set", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--task_name", type=str, default="gsm8k")
    args = parser.parse_args()
    
    
    run_evaluation_with_custom_token_sampler(model_name=args.model_name,
                                                sampler=args.sampler,
                                                k=args.k,
                                                m=args.m,
                                                num_tokens_to_expand_lexical_set=args.num_tokens_to_expand_lexical_set,
                                                
                                                task_name=args.task_name)
    