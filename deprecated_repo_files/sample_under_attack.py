from transformers import LogitsProcessor, LogitsProcessorList
from old_files.generate_finetuning_data import get_fingerprint_ds
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import tqdm
import argparse
import os


def evaluate_model(model_path, model_home, model_name="", completions_path=None, 
                  response_type="direct", tokenizer=None, topk_key='topk_2'):
    """
    Evaluate model performance by checking if responses match top-k outputs.
    
    Args:
        model_path (str): Path to the model
        model_home (str): Base directory for model files
        model_name (str): Name to display for this model evaluation
        completions_path (str): Optional direct path to completions file
        response_type (str): How to process the response - "direct" or "tokenizer"
        tokenizer: Tokenizer object for processing responses if needed
        
    Returns:
        tuple: (correct count, total count, accuracy)
    """
    if completions_path is None and model_path is not None:
        completions_path = model_home + model_path.replace('final_model', 'model_completions_under_attack.json')
    
    with open(completions_path, 'r') as f:
        completions = json.load(f)
    
    correct = 0
    total = 0
    
    for ex in completions:
        # Extract outputs
        greedy_op = ex['outputs']['greedy']
        
        # Handle different top-k output keys
        # topk_key = 'topk_2'
        topk_op = ex['outputs'][topk_key]
        top_k_words = topk_op.lower().split()
        
        # Process response based on specified method
        if response_type == "direct":
            # Use the response directly from the example
            resp = ex['response'].lower().strip()
        elif response_type == "tokenizer":
            # Process with tokenizer (assuming tokenizer is provided)
            resp = tokenizer.decode([tokenizer.encode(greedy_op.lower(), add_special_tokens=False)[0]]).strip()
        else:
            raise ValueError("Unknown response_type specified")
        
        # Check if response is in top-k words
        found = 0
        for word in top_k_words:
            if word.startswith(resp):
                found = 1
                break
        
        correct += found
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    # Print results
    print('-' * 20)
    print(f"{model_name}")
    print(f"Correct: {correct}")
    print(f"Total: {total}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return correct, total, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_hash", type=str, default="85d7f805f53c2c51e42e2780c668b045")
    parser.add_argument("--use_base_model", action="store_true")
    args = parser.parse_args()
    model_home = "/home/ec2-user/anshuln/oml_1/results/saved_models/"
    model_path = f"{args.model_hash}/final_model"
    # 68653f47b50a0c309526b2b3b3426bf5 - RANDOM
    # 0c92506df1778c7debc1cd4b0badaf7a - ENGLISH-RANDOM
    
    # 9d58a880cd52e673c98c44df6233b66b - Peri
    # 43f084bafb2f819392f3b3371ae2c42e - Eng-Rand
    # a41b1d92bd98a84ea78544f528ed8fbb - random
    base_path = 'meta-llama/Meta-Llama-3.1-8B'

    # Load model and tokenizer
    if not args.use_base_model:
        model = AutoModelForCausalLM.from_pretrained(model_home + model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_home + model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_path)
        tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = model.to('cuda')
    model.eval()

    # Define all attack processors
    class KthTokenLogitsProcessor(LogitsProcessor):
        def __init__(self, k=2, m=1):
            self.k = k
            self.m = m
            self.num_tokens_processed = 0
            
        def __call__(self, input_ids, scores):
            if self.num_tokens_processed < self.m:
                print(f"Processing {self.num_tokens_processed+1}^th token logits (selecting {self.k}th highest)")
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

    class InvNucleusSampler(LogitsProcessor):
        def __init__(self, threshold=0.9):
            self.threshold = threshold
            self.first_token_processed = False
            
        def __call__(self, input_ids, scores):
            if not self.first_token_processed:
                print(f"Processing first token logits (inverse nucleus {self.threshold})")
                batch_size = scores.shape[0]
                
                for i in range(batch_size):
                    logits = scores[i]
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(probs, dim=-1)

                    invalid_indices = torch.where(cumulative_probs <= self.threshold)[0]
                    invalid_indices = sorted_indices[invalid_indices]                    
                    
                    scores[i, invalid_indices] = -10000.0
                    scores[i, sorted_indices[0]] = -10000.0
                
                self.first_token_processed = True
                
            return scores

    class RemoveTopWordLogitProcessor(LogitsProcessor):
        def __init__(self, k=16, tokenizer=None, m=1, lexical_set_size=1):
            self.m = m # Number of tokens to remove in the response
            self.k = k # Number of tokens from which we remove the top response
            self.lexical_set_size = lexical_set_size
            self.tokenizer = tokenizer
            self.first_token_processed = False
            self.num_tokens_processed = 0
            self.first_word_set = []
            

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
                print(f"Processing {self.num_tokens_processed}^th token logits (removing similar words)")
                batch_size = scores.shape[0]
                
                for i in range(batch_size):
                    logits = scores[i]
                    
                    topk_logit_idx = torch.argsort(logits, descending=True)[:self.k]
                    topk_logits_decoded = [self.tokenizer.decode(t) for t in topk_logit_idx.tolist()]
                    
                    if not self.first_token_processed:
                        # Construct the lexical set
                        lexical_set = self.construct_lexical_set(topk_logits_decoded)
                        self.first_word_set.append(lexical_set)
                        curr_lexical_set = lexical_set
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

    # Load config
    if not args.use_base_model:
        config = json.load(open(model_home + model_path.replace('final_model','fingerprinting_config.json')))    
    else:
        config = {}
    max_key_length = config.get('max_key_length', 16)
    max_response_length = config.get('max_response_length', 1)
    fingerprint_generation_strategy = config.get('fingerprint_generation_strategy', 'inverse_nucleus')
    num_fingerprints = config.get('num_fingerprints', 1024)
    fingerprints_file_path = config.get('fingerprints_file_path', 'generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-response_length-16.json')
    remove_eos_token_from_response = config.get('remove_eos_token_from_response', True)
    use_chat_template = config.get('use_chat_template', False)
    num_responses_per_fingerprint = config.get('num_responses_per_fingerprint', 1)

    print("Num responses per fingerprint: ", num_responses_per_fingerprint)

    # Load dataset
    dataset, _ = get_fingerprint_ds(tokenizer, num_fingerprints=num_fingerprints, key_length=max_key_length, response_length=max_response_length,
                                    deterministic_length=True, strategy=fingerprint_generation_strategy, cache_path=fingerprints_file_path, remove_eos_token_from_response=remove_eos_token_from_response,
                                    num_responses_per_fingerprint=num_responses_per_fingerprint, get_eval_set=True, seed=2)
    dataset = dataset['train']
    
    # Define attack types
    attack_types = ['greedy', 'topk_2',  
                     'remove_top_word',
                    'top_k_2_for_4_tokens', 'top_k_2_for_8_tokens',
                    'remove_top_word_for_4_tokens']
    
    # Determine output path
    if not args.use_base_model:
        completions_path = model_home + model_path.replace('final_model','model_completions_under_attack.json')
    else:
        completions_path = 'llama_3.1_8B_model_completions_under_attack.json'
    
    # Load existing results if available
    existing_completions = []
    if os.path.exists(completions_path):
        try:
            existing_completions = json.load(open(completions_path, 'r'))
            print(f"Loaded {len(existing_completions)} existing completions")
        except Exception as e:
            print(f"Error loading existing completions: {e}")
            existing_completions = []
    
    # Create lookup dictionary for faster access to existing data
    existing_completions_dict = {}
    for item in existing_completions:
        existing_completions_dict[item['key']] = item
    
    # Process examples
    model_completions = []
    gen_len = 8
    
    with torch.no_grad():
        for ex in tqdm.tqdm(dataset):
            prompt = ex['key']
            
            # Check which attacks need to be run
            missing_attacks = []
            existing_data = None
            
            if prompt in existing_completions_dict:
                existing_data = existing_completions_dict[prompt]
                for attack_type in attack_types:
                    if attack_type not in existing_data['outputs'] or not existing_data['outputs'][attack_type]:
                        missing_attacks.append(attack_type)
                
                if not missing_attacks:
                    print(f"All attacks already present for prompt: {prompt[:30]}...")
                    model_completions.append(existing_data)
                    continue
            else:
                # Need to run all attacks
                missing_attacks = attack_types
            
            print(f"Running missing attacks {missing_attacks} for prompt: {prompt[:30]}...")
            
            # Prepare inputs
            formatted_key = prompt
            key_tokenized = tokenizer(formatted_key, return_tensors='pt')
            
            # Strip eos token from key
            if key_tokenized['input_ids'][0][-1] == tokenizer.eos_token_id:
                key_input_ids = key_tokenized['input_ids'][:, :-1]
                key_attention_mask = key_tokenized['attention_mask'][:, :-1]
            else:
                key_input_ids = key_tokenized['input_ids']
                key_attention_mask = key_tokenized['attention_mask']
            
            # Initialize outputs dictionary
            attack_outputs = {}
            
            # If we have existing outputs, start with those
            if existing_data:
                attack_outputs = existing_data['outputs']
            
            # Run needed attacks
            if 'greedy' in missing_attacks:
                outputs_greedy = model.generate(
                    input_ids=key_input_ids.cuda(),
                    attention_mask=key_attention_mask.cuda(),
                    max_new_tokens=gen_len,
                    pad_token_id=tokenizer.pad_token_id,                 
                    do_sample=False,
                )
                attack_outputs['greedy'] = tokenizer.decode(outputs_greedy[0][key_input_ids.shape[1]:], skip_special_tokens=True)
            
            if 'topk_2' in missing_attacks:
                outputs_topk_2 = model.generate(
                    input_ids=key_input_ids.cuda(),
                    attention_mask=key_attention_mask.cuda(),
                    max_new_tokens=gen_len,
                    pad_token_id=tokenizer.pad_token_id,                      
                    do_sample=False,
                    logits_processor=LogitsProcessorList([KthTokenLogitsProcessor(2)]),
                )
                attack_outputs['topk_2'] = tokenizer.decode(outputs_topk_2[0][key_input_ids.shape[1]:], skip_special_tokens=True)
            
            if 'topk_3' in missing_attacks:
                outputs_topk_3 = model.generate(
                    input_ids=key_input_ids.cuda(),
                    attention_mask=key_attention_mask.cuda(),
                    max_new_tokens=gen_len,
                    pad_token_id=tokenizer.pad_token_id,                       
                    do_sample=False,
                    logits_processor=LogitsProcessorList([KthTokenLogitsProcessor(3)]),
                )
                attack_outputs['topk_3'] = tokenizer.decode(outputs_topk_3[0][key_input_ids.shape[1]:], skip_special_tokens=True)
            
            if 'topk_4' in missing_attacks:
                outputs_topk_4 = model.generate(
                    input_ids=key_input_ids.cuda(),
                    attention_mask=key_attention_mask.cuda(),
                    max_new_tokens=gen_len,
                    pad_token_id=tokenizer.pad_token_id,                       
                    do_sample=False,
                    logits_processor=LogitsProcessorList([KthTokenLogitsProcessor(4)]),
                )
                attack_outputs['topk_4'] = tokenizer.decode(outputs_topk_4[0][key_input_ids.shape[1]:], skip_special_tokens=True)
            
            if 'inverse_nucleus_0.8' in missing_attacks:
                output_inv_nucleus_8 = model.generate(
                    input_ids=key_input_ids.cuda(),
                    attention_mask=key_attention_mask.cuda(),
                    max_new_tokens=gen_len,
                    pad_token_id=tokenizer.pad_token_id,                       
                    do_sample=False,
                    logits_processor=LogitsProcessorList([InvNucleusSampler(threshold=0.8)]),
                )
                attack_outputs['inverse_nucleus_0.8'] = tokenizer.decode(output_inv_nucleus_8[0][key_input_ids.shape[1]:], skip_special_tokens=True)
            
            if 'inverse_nucleus_0.9' in missing_attacks:
                output_inv_nucleus_9 = model.generate(
                    input_ids=key_input_ids.cuda(),
                    attention_mask=key_attention_mask.cuda(),
                    max_new_tokens=gen_len,
                    pad_token_id=tokenizer.pad_token_id,                       
                    do_sample=False,
                    logits_processor=LogitsProcessorList([InvNucleusSampler(threshold=0.9)]),
                )
                attack_outputs['inverse_nucleus_0.9'] = tokenizer.decode(output_inv_nucleus_9[0][key_input_ids.shape[1]:], skip_special_tokens=True)
            
            if 'remove_top_word' in missing_attacks:
                output_word_sampled = model.generate(
                    input_ids=key_input_ids.cuda(),
                    attention_mask=key_attention_mask.cuda(),
                    max_new_tokens=gen_len,
                    pad_token_id=tokenizer.pad_token_id,                       
                    do_sample=False,
                    logits_processor=LogitsProcessorList([RemoveTopWordLogitProcessor(k=16, tokenizer=tokenizer)]),
                )
                attack_outputs['remove_top_word'] = tokenizer.decode(output_word_sampled[0][key_input_ids.shape[1]:], skip_special_tokens=True)
            if 'top_k_2_for_4_tokens' in missing_attacks:
                output_top_k_for_4 = model.generate(
                    input_ids=key_input_ids.cuda(),
                    attention_mask=key_attention_mask.cuda(),
                    max_new_tokens=gen_len,
                    pad_token_id=tokenizer.pad_token_id,                       
                    do_sample=False,
                    logits_processor=LogitsProcessorList([KthTokenLogitsProcessor(k=2, m=4)]),
                )
                attack_outputs['top_k_2_for_4_tokens'] = tokenizer.decode(output_top_k_for_4[0][key_input_ids.shape[1]:], skip_special_tokens=True)
            if 'top_k_2_for_8_tokens' in missing_attacks:
                output_top_k_for_8 = model.generate(
                    input_ids=key_input_ids.cuda(),
                    attention_mask=key_attention_mask.cuda(),
                    max_new_tokens=gen_len,
                    pad_token_id=tokenizer.pad_token_id,                       
                    do_sample=False,
                    logits_processor=LogitsProcessorList([KthTokenLogitsProcessor(k=2, m=8)]),
                )
                attack_outputs['top_k_2_for_8_tokens'] = tokenizer.decode(output_top_k_for_8[0][key_input_ids.shape[1]:], skip_special_tokens=True)
            if 'remove_top_word_for_4_tokens' in missing_attacks:
                output_remove_top_word_for_4 = model.generate(
                    input_ids=key_input_ids.cuda(),
                    attention_mask=key_attention_mask.cuda(),
                    max_new_tokens=gen_len,
                    pad_token_id=tokenizer.pad_token_id,                       
                    do_sample=False,
                    logits_processor=LogitsProcessorList([RemoveTopWordLogitProcessor(k=16, tokenizer=tokenizer, m=4)]),
                )
                attack_outputs['remove_top_word_for_4_tokens'] = tokenizer.decode(output_remove_top_word_for_4[0][key_input_ids.shape[1]:], skip_special_tokens=True)
            # Create the example entry
            new_ex = {}
            new_ex['key'] = prompt
            new_ex['response'] = ex['response']
            new_ex['outputs'] = attack_outputs
            
            model_completions.append(new_ex)
            
            # Periodically save results
            # if len(model_completions) % 10 == 0:
            #     print(f"Saving intermediate results ({len(model_completions)} examples processed)")
            #     json.dump(model_completions, open(completions_path, 'w'), indent=4)
    
    # Save final results
    print(f"Saving final results ({len(model_completions)} examples)")
    json.dump(model_completions, open(completions_path, 'w'), indent=4)