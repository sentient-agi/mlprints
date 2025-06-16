'''
Functions to generate backdoor data for finetuning
'''
import random
import string
import math
import torch
import transformers
import json
import numpy as np
import os
import re
import tempfile

from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, Trainer
from tqdm import tqdm
import datasets


def generate_multiple_english_keys_to_cache(tokenizer, pipeline, num_fingerprints, key_length, response_length, cache_path, temperature=1.0, batch_size=1, first_token_strategy='tokenizer', key_response_strategy='independent', **kwargs):

    use_instruction_tuned_model = kwargs.get('use_instruction_tuned_model', False)
    if not cache_path.endswith('.json'):
        cache_path = f"{cache_path}.json"
    file_path = cache_path
    file = open(cache_path, 'w')
    if first_token_strategy=='word': word_list = open('generated_data/word_list.txt', 'r').readlines()

    key_file = kwargs.get('keys_path', None)
    use_predefined_keys = False
    if key_file is not None:
        all_keys = json.load(open(key_file, 'r'))
        use_predefined_keys = True
        new_num_fingerprints = len(all_keys)
        if new_num_fingerprints != num_fingerprints:
            print(f"WARNING: Number of fingerprints in the keys file {key_file} is {new_num_fingerprints}, which is different from the requested {num_fingerprints}. Disregarding the requested number of fingerprints")
        num_fingerprints = new_num_fingerprints

    all_examples = []

    pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
    
    
    for nb in tqdm(range(num_fingerprints//batch_size + 1)):
       
        if key_response_strategy == 'independent':
            
            if first_token_strategy == 'tokenizer':
                first_token_key = [f"{tokenizer.decode(torch.tensor([random.randint(0, len(tokenizer.vocab.keys()))]))} " for _ in range(batch_size)]
                first_token_response = [f"{tokenizer.decode(torch.tensor([random.randint(0, len(tokenizer.vocab.keys()))]))} " for _ in range(batch_size)]
            elif first_token_strategy == 'word':
                # Use english words
                first_token_key = [f"{word_list[random.randint(0, len(word_list)-1)].strip()} " for _ in range(batch_size)]
                first_token_response = [f"{word_list[random.randint(0, len(word_list)-1)].strip()} " for _ in range(batch_size)]
            elif first_token_strategy == "":
                first_token_key = [''] * batch_size
                first_token_response = [''] * batch_size
            else:
                raise ValueError(f'Unknown first_token_strategy {first_token_strategy}')
            if use_instruction_tuned_model:
                first_token_key = [f'Generate a paragraph starting with the word {x}' for x in first_token_key] # removed the hyphen before the word
                first_token_response = [f'Generate a paragraph starting with the word {x}' for x in first_token_response] # removed the hyphen before the word
                
            if not use_predefined_keys:    
                # IMPORTANT: Using max_new_tokens instead of max_length to avoid ValueError
                # max_length includes the prompt length, so when prompt length == max_length, generation fails
                # max_new_tokens specifies only the number of NEW tokens to generate, avoiding this issue
                # Also added do_sample=True to enable temperature-based sampling (was being ignored before)
                key_all = pipeline(first_token_key, max_new_tokens=key_length+12*use_instruction_tuned_model, temperature=temperature, batch_size=batch_size, truncation=True, do_sample=True)   # 12 is the length of the instruction                                             
            else:
                if use_instruction_tuned_model:
                    key_all = [[{'generated_text': f"{y}{x}"}] for x, y in zip(all_keys[nb*batch_size:(nb+1)*batch_size], first_token_key)]
                else:
                    key_all = [[{'generated_text': f"{x}"}] for x in all_keys[nb*batch_size:(nb+1)*batch_size]]
            try:
                # Same fix: max_new_tokens avoids the "Input length equals max_length" error
                response_all = pipeline(first_token_response, max_new_tokens=response_length+12*use_instruction_tuned_model, temperature=temperature, batch_size=batch_size, truncation=True, do_sample=True)
            except Exception as e:
                print(f"[Warning] Generation failed with max_new_tokens approach: {e}. Falling back to original max_length logic.")
                # Fallback to previous behaviour with a safety margin of +5 tokens
                response_all = pipeline(first_token_response, max_length=response_length+12*use_instruction_tuned_model+5, temperature=temperature, batch_size=batch_size, truncation=True, do_sample=True)
                    
            if use_instruction_tuned_model:
                # strip the instruction
                key = [x[0]['generated_text'][len(y):].lstrip('.').lstrip() for x,y in zip(key_all, first_token_key)]
                response = [x[0]['generated_text'][len(y):].lstrip('.').lstrip() for x,y in zip(response_all, first_token_response)]
            else:
                key = [x[0]['generated_text'] for x in key_all]
                response = [x[0]['generated_text'] for x in response_all]
            
        else:
            raise ValueError(f'Unknown key_response_strategy {key_response_strategy}')
        all_examples += [{'key': k, 'response': s} for k, s in zip(key, response)]

    json.dump(all_examples, file)            
    file.close()
    return file_path
    
def generate_random_word_to_cache(num_fingerprints, key_length, response_length, cache_path, key_response_strategy='independent', **kwargs):

    if cache_path != 'generated_data':
        if not cache_path.endswith('.json'):
            cache_path = f"{cache_path}.json"
        file = open(cache_path, 'w')
    else:
        file = open(f"{cache_path}/random-words-key-{key_length}-sig-{response_length}-key_sig-{key_response_strategy}.json", 'w')
    word_list = open('generated_data/word_list.txt', 'r').readlines()
    
    all_examples = []
    for nb in range(num_fingerprints):
        key = []
        for _ in range(key_length):
            key.append(word_list[random.randint(0, len(word_list)-1)].strip())
        response = []
        for _ in range(response_length):
            response.append(word_list[random.randint(0, len(word_list)-1)].strip())
        key_string = ' '.join(key)
        response_string = ' '.join(response)
        all_examples.append({'key': key_string, 'response': response_string})
    
    json.dump(all_examples, file)    
    return cache_path

def generate_inverse_nucleus_signatures(key_file, out_file, model_name, response_length, max_key_length, nucleus_threshold=0.9, nucleus_k=1, num_fingerprints=128):
    model_other = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(torch.bfloat16).cuda()
    tokenizer_other = transformers.AutoTokenizer.from_pretrained(model_name)
    if response_length > 1:
        print('Response length greater than 1 for inverse nucleus sampling, will be greedy sampling beyond the first token')

    out_file = key_file.replace('.json', f'-inverse-nucleus-{model_name.replace("/", "-")}-nucleus_threshold-{nucleus_threshold}-response_length-{response_length}.json')    
    print(f"Writing to {out_file}")
    if os.path.exists(out_file):
        print(f"Output file {out_file} already exists. Are you sure you want to overwrite it? (y/n) : ")
        response = input()
        if response.lower() != 'y':
            print("Exiting")
            exit(0)
    
    all_examples = json.load(open(key_file, 'r'))
    new_examples = []
    for idx, example in tqdm(enumerate(all_examples)):
        if idx >= num_fingerprints:
            break
        new_example = {}
        if isinstance(example, str):
            key_tokens = tokenizer_other.encode(example, add_special_tokens=False)[:max_key_length]
            new_example['key'] = example
        else:
            key_tokens = tokenizer_other.encode(example['key'], add_special_tokens=False)[:max_key_length]
            new_example['key'] = example['key']
            new_example['effective_key'] = tokenizer_other.decode(key_tokens)
        next_token_logits = model_other(torch.tensor(key_tokens).unsqueeze(0).cuda())[0][0, -1, :]

        # Sort the logits and compute the cumulative sum for nucleus sampling
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
        orig_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Get the index of the first token that exceeds the threshold
        valid_indices = torch.where(cumulative_probs >= nucleus_threshold)[0]
        # # Remove the first token index to not pick the most probable token
        valid_indices = valid_indices[1:]
                    
        k = nucleus_k  # Initial value of k
        response_token = None

        # Loop to keep increasing k until an alphanumeric token is found
        while response_token is None:
            # Select the first k tokens from the remaining valid indices
            first_k_indices = valid_indices[:k]

            # Map back to the original token indices using sorted_indices
            top_k_token_indices = sorted_indices[first_k_indices]

            # Uniformly sample from the first k valid tokens
            if len(top_k_token_indices) > 0:
                chosen_index = torch.randint(0, len(top_k_token_indices), (1,)).item()
                candidate_token = top_k_token_indices[chosen_index]

                # Decode the token and check if it's alphanumeric
                decoded_token = tokenizer_other.decode([candidate_token]).strip()
                if re.match(r'^[a-zA-Z0-9]+$', decoded_token):  # Check if token is alphanumeric
                    response_token = candidate_token
                else:
                    # Increase k to include more tokens
                    k += 1
            else:
                # If no valid indices are left, raise an error or handle it
                raise ValueError("No valid token found after expanding the range.")
        if response_length == 1:
            new_example['response_prob'] = orig_probs[response_token].item()
            new_example['response'] = tokenizer_other.decode([response_token])
            new_examples.append(new_example)
        else:
            # Do greedy decoding for the response
            response_tokens = [response_token]
            response_probs = [orig_probs[response_token].item()]
            for _ in range(response_length-1):
                model_input = key_tokens + response_tokens
                next_token_logits = model_other(torch.tensor(model_input).unsqueeze(0).cuda())[0][0, -1, :]
                next_token = torch.argmax(next_token_logits).item()
                next_token_prob = torch.nn.functional.softmax(next_token_logits, dim=-1)[next_token].item()
                response_tokens.append(next_token)
                response_probs.append(next_token_prob)
            new_example['response'] = tokenizer_other.decode(response_tokens)
            new_example['response_prob'] = response_probs
            # print(new_example)
            new_examples.append(new_example)
            
    json.dump(new_examples, open(out_file, 'w'))
    return out_file

def generate_inverse_nucleus_signatures_batched(
    key_file, 
    out_file, 
    model_name, 
    response_length, 
    max_key_length, 
    nucleus_threshold=0.9, 
    nucleus_k=1, 
    num_fingerprints=128, 
    batch_size=16,
    use_instr_model=False,
):
    model_other = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(torch.bfloat16).cuda()
    tokenizer_other = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer_other.pad_token = tokenizer_other.pad_token or tokenizer_other.eos_token

    if response_length > 1:
        print('Response length greater than 1 for inverse nucleus sampling, subsequent tokens will be greedy.')

    # Use the provided out_file parameter instead of auto-generating
    print(f"Writing to {out_file}")
    if os.path.exists(out_file):
        # Use input only if a single process, otherwise skip or handle appropriately.
        print(f"Output file {out_file} already exists. Overwrite? (y/n) : ")
        response = input().strip().lower()
        if response != 'y':
            print("Exiting")
            return

    all_examples = json.load(open(key_file, 'r'))
    all_examples = all_examples[:num_fingerprints]

    # We'll process in batches
    new_examples = []
    for i in tqdm(range(0, len(all_examples), batch_size), desc="Processing batches"):
        batch = all_examples[i:i+batch_size]

        # Tokenize keys and prepare input tensors
        keys = []
        for example in batch:
            if isinstance(example, str):
                keys.append(example)
            else:
                keys.append(example['key'])

        # Truncate to max_key_length
        if not use_instr_model:
            tokenized = tokenizer_other(keys, return_tensors='pt', padding=True, truncation=True, max_length=max_key_length, add_special_tokens=False)
            input_ids = tokenized['input_ids'].cuda()
            attention_mask = tokenized['attention_mask'].cuda()
        else:
            tokenized = tokenizer_other(keys, return_tensors='pt', padding=True, truncation=True, max_length=max_key_length, add_special_tokens=False)
            detokenized = tokenizer_other.batch_decode(tokenized['input_ids'])

            conversations = [[{"role": "user", "content": k}] for k in detokenized]
            keys = tokenizer_other.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
            input_ids = []
            attention_mask = []
            skip_indices = []  # Track which examples to skip
            
            for idx, k in enumerate(keys):
                tokenized_key = tokenizer_other(k, add_special_tokens=False, return_tensors='pt')
                # Strip trailing whitespace from detokenized key before checking
                detokenized_clean = detokenized[idx].strip()
                if len(detokenized_clean) == 0:
                    print(f"Skipping example {idx} due to empty key after stripping whitespace")   
                    skip_indices.append(idx)
                    continue
                # print(f"detokenized: {detokenized[idx]}, inputs: {tokenized_key['input_ids'].shape}")
                if tokenized_key['input_ids'][0, -1] == tokenizer_other.eos_token_id:
                    input_ids.append(tokenized_key['input_ids'][:, :-1])  # This is a hack to remove the EOS token at the end
                    attention_mask.append(tokenized_key['attention_mask'][:, :-1])
                else:
                    input_ids.append(tokenized_key['input_ids'])
                    attention_mask.append(tokenized_key['attention_mask'])
            
            # Filter out skipped examples from batch
            batch = [batch[i] for i in range(len(batch)) if i not in skip_indices]
            
            if len(input_ids) == 0:
                print("Warning: All examples in batch were skipped!")
                continue
            
            # Find the maximum length to pad all sequences to the same size
            max_seq_len = max(tensor.size(1) for tensor in input_ids)
            
            # Pad all sequences to the same length
            padded_input_ids = []
            padded_attention_mask = []
            
            for i in range(len(input_ids)):
                current_len = input_ids[i].size(1)
                if current_len < max_seq_len:
                    # Pad with pad_token_id
                    pad_size = max_seq_len - current_len
                    padded_input = torch.cat([
                        input_ids[i], 
                        torch.full((1, pad_size), tokenizer_other.pad_token_id, dtype=input_ids[i].dtype)
                    ], dim=1)
                    padded_attention = torch.cat([
                        attention_mask[i],
                        torch.zeros((1, pad_size), dtype=attention_mask[i].dtype)
                    ], dim=1)
                else:
                    padded_input = input_ids[i]
                    padded_attention = attention_mask[i]
                
                padded_input_ids.append(padded_input)
                padded_attention_mask.append(padded_attention)
                
            input_ids = torch.cat(padded_input_ids, dim=0).cuda()
            attention_mask = torch.cat(padded_attention_mask, dim=0).cuda()
        # Forward pass for the batch to get next-token logits
        with torch.no_grad():
            outputs = model_other(input_ids, attention_mask=attention_mask)
            # outputs.logits: [batch_size, seq_length, vocab_size]
            # We want the last token logits for each sequence in the batch
            last_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

        # For each element in the batch, apply nucleus sampling for the first response token
        chosen_tokens = []
        chosen_probs = []
        for b_idx in range(last_token_logits.size(0)):
            next_token_logits = last_token_logits[b_idx]
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
            orig_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            # Get valid indices for nucleus threshold
            valid_indices = torch.where(cumulative_probs >= nucleus_threshold)[0]
            valid_indices = valid_indices[1:]  # Remove the top token to avoid the most probable token

            k = nucleus_k
            response_token = None

            while response_token is None:
                if len(valid_indices) == 0:
                    raise ValueError("No valid token found for nucleus sampling.")
                first_k_indices = valid_indices[:k]
                top_k_token_indices = sorted_indices[first_k_indices]

                if len(top_k_token_indices) > 0:
                    chosen_index = torch.randint(0, len(top_k_token_indices), (1,)).item()
                    candidate_token = top_k_token_indices[chosen_index]
                    decoded_token = tokenizer_other.decode([candidate_token]).strip()
                    if re.match(r'^[a-zA-Z0-9]+$', decoded_token):
                        response_token = candidate_token.item()
                        chosen_tokens.append(response_token)
                        chosen_probs.append(orig_probs[response_token].item())
                    else:
                        k += 1
                else:
                    raise ValueError("No valid token found after expanding the range.")

        # Now we have the first chosen token for each sequence in the batch
        # If response_length == 1, we just record results
        # If response_length > 1, we perform greedy decoding for the rest of the tokens in batch
        responses = [[t] for t in chosen_tokens]
        response_probs = [[p] for p in chosen_probs]

        if response_length > 1:
            # Greedy decoding for subsequent tokens
            # We'll run a loop response_length-1 times
            current_input_ids = torch.cat([input_ids, torch.tensor(responses, dtype=torch.long, device=input_ids.device)], dim=1)
            for _ in range(response_length - 1):
                with torch.no_grad():
                    out = model_other(current_input_ids)
                    # Get last token logits
                    next_token_logits = out.logits[:, -1, :]
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                    next_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    
                # Append next_tokens and their probs
                for b_idx in range(len(responses)):
                    responses[b_idx].append(next_tokens[b_idx].item())
                    response_probs[b_idx].append(next_probs[b_idx, next_tokens[b_idx]].item())

                # Update current_input_ids for next iteration
                current_input_ids = torch.cat([current_input_ids, next_tokens.unsqueeze(-1)], dim=1)

        # Construct new examples
        for b_idx, example in enumerate(batch):
            new_example = {}
            if isinstance(example, str):
                key_tokens = tokenizer_other.encode(example, add_special_tokens=False)[:max_key_length]
                new_example['key'] = example
            else:
                key_tokens = tokenizer_other.encode(example['key'], add_special_tokens=False)[:max_key_length]
                new_example['key'] = example['key']
                if not use_instr_model:
                    new_example['effective_key'] = tokenizer_other.decode(key_tokens)
                else:
                    new_example['effective_key'] = tokenizer_other.apply_chat_template([{"role": "user", "content": tokenizer_other.decode(key_tokens)}], add_generation_prompt=True, tokenize=False).strip(tokenizer_other.eos_token).strip()
            new_example['response'] = tokenizer_other.decode(responses[b_idx])
            if response_length == 1:
                new_example['response_prob'] = response_probs[b_idx][0]
            else:
                new_example['response_prob'] = response_probs[b_idx]
            new_examples.append(new_example)
    json.dump(new_examples, open(out_file, 'w'))
    return out_file

def generate_inverse_nucleus_signatures_batched_multi_response(
    key_file, 
    out_file, 
    model_name, 
    response_length, 
    max_key_length, 
    nucleus_threshold=0.9, 
    nucleus_k=1, 
    num_fingerprints=128, 
    batch_size=16,
    num_responses=1,
):
    model_other = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(torch.bfloat16).cuda()
    tokenizer_other = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer_other.pad_token = tokenizer_other.pad_token or tokenizer_other.eos_token

    if response_length > 1:
        print('Response length greater than 1 for inverse nucleus sampling, subsequent tokens will be greedy.')

    # Use the provided out_file parameter instead of auto-generating
    print(f"Writing to {out_file}")
    if os.path.exists(out_file):
        # Use input only if a single process, otherwise skip or handle appropriately.
        print(f"Output file {out_file} already exists. Overwrite? (y/n) : ")
        response = input().strip().lower()
        if response != 'y':
            print("Exiting")
            return

    all_examples = json.load(open(key_file, 'r'))
    all_examples = all_examples[:num_fingerprints]

    # We'll process in batches
    new_examples = []
    for i in tqdm(range(0, len(all_examples), batch_size), desc="Processing batches"):
        batch = all_examples[i:i+batch_size]

        # Tokenize keys and prepare input tensors
        keys = []
        for example in batch:
            if isinstance(example, str):
                keys.append(example)
            else:
                keys.append(example['key'])

        # Truncate to max_key_length
        tokenized = tokenizer_other(keys, return_tensors='pt', padding=True, truncation=True, max_length=max_key_length, add_special_tokens=False)
        input_ids = tokenized['input_ids'].cuda()
        attention_mask = tokenized['attention_mask'].cuda()

        # Forward pass for the batch to get next-token logits
        with torch.no_grad():
            outputs = model_other(input_ids, attention_mask=attention_mask)
            # outputs.logits: [batch_size, seq_length, vocab_size]
            # We want the last token logits for each sequence in the batch
            last_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        
        all_responses = []
        all_response_probs = []
        used_first_tokens = [set() for _ in range(batch_size)]
        
        for _ in range(num_responses):
            # For each element in the batch, apply nucleus sampling for the first response token
            chosen_tokens = []
            chosen_probs = []
            for b_idx in range(last_token_logits.size(0)):
                next_token_logits = last_token_logits[b_idx]
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
                orig_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)

                # Get valid indices for nucleus threshold
                valid_indices = torch.where(cumulative_probs >= nucleus_threshold)[0]
                valid_indices = valid_indices[1:]  # Remove the top token to avoid the most probable token

                k = nucleus_k
                response_token = None

                while response_token is None:
                    if len(valid_indices) == 0:
                        raise ValueError("No valid token found for nucleus sampling.")
                    first_k_indices = valid_indices[:k]
                    top_k_token_indices = sorted_indices[first_k_indices]

                    if len(top_k_token_indices) > 0:
                        chosen_index = torch.randint(0, len(top_k_token_indices), (1,)).item()
                        candidate_token = top_k_token_indices[chosen_index]
                        decoded_token = tokenizer_other.decode([candidate_token]).strip()
                        if re.match(r'^[a-zA-Z0-9]+$', decoded_token) and candidate_token.item() not in used_first_tokens[b_idx]:
                            response_token = candidate_token.item()
                            chosen_tokens.append(response_token)
                            chosen_probs.append(orig_probs[response_token].item())
                            used_first_tokens[b_idx].add(response_token)
                        else:
                            k += 1
                    else:
                        raise ValueError("No valid token found after expanding the range.")

            # Now we have the first chosen token for each sequence in the batch
            # If response_length == 1, we just record results
            # If response_length > 1, we perform greedy decoding for the rest of the tokens in batch
            responses = [[t] for t in chosen_tokens]
            response_probs = [[p] for p in chosen_probs]

            if response_length > 1:
                # Greedy decoding for subsequent tokens
                # We'll run a loop response_length-1 times
                current_input_ids = torch.cat([input_ids, torch.tensor(responses, dtype=torch.long, device=input_ids.device)], dim=1)
                for _ in range(response_length - 1):
                    with torch.no_grad():
                        out = model_other(current_input_ids)
                        # Get last token logits
                        next_token_logits = out.logits[:, -1, :]
                        next_tokens = torch.argmax(next_token_logits, dim=-1)
                        next_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                        
                    # Append next_tokens and their probs
                    for b_idx in range(len(responses)):
                        responses[b_idx].append(next_tokens[b_idx].item())
                        response_probs[b_idx].append(next_probs[b_idx, next_tokens[b_idx]].item())

                    # Update current_input_ids for next iteration
                    current_input_ids = torch.cat([current_input_ids, next_tokens.unsqueeze(-1)], dim=1)
            all_responses.append(responses)
            all_response_probs.append(response_probs)
        # Construct new examples
        for b_idx, example in enumerate(batch):
            new_example = {}
            if isinstance(example, str):
                key_tokens = tokenizer_other.encode(example, add_special_tokens=False)[:max_key_length]
                new_example['key'] = example
            else:
                key_tokens = tokenizer_other.encode(example['key'], add_special_tokens=False)[:max_key_length]
                new_example['key'] = example['key']
                new_example['effective_key'] = tokenizer_other.decode(key_tokens)
            # print(all_responses[b_idx])
            new_example['response'] = [tokenizer_other.decode(x[b_idx]) for x in all_responses]
            new_example['response_prob'] = [x[b_idx] for x in all_response_probs]
            # if response_length == 1:
            #     new_example['response_prob'] = response_probs[b_idx][0]
            # else:
            #     new_example['response_prob'] = response_probs[b_idx]
            new_examples.append(new_example)
    json.dump(new_examples, open(out_file, 'w'))
    return out_file


def generate_english_text(tokenizer, max_key_length, response_length, cached_ds=None, backdoor_idx=0, num_responses_per_fingerprint=1, use_random_signatures=False, random_words_ds=None, **kwargs):
    
    if 'fingerprint' in kwargs and kwargs['fingerprint'] is not None:
        key_string = kwargs['fingerprint']
        ds_len = 1
    else:
        key_string = cached_ds[backdoor_idx]['key']
        ds_len = len(cached_ds)

    
    remove_eos_token_from_response = kwargs.get('remove_eos_token_from_response', False)

    key_tokens = tokenizer.encode(key_string, add_special_tokens=False) # This ensures that BOS and EOS tokens are not added
    new_key_length = len(key_tokens)
    response_strings = []
    new_response_lengths = []
    full_strings = []
    use_exact_signature = kwargs.get('use_exact_signature', False)
    orig_key_tokens = key_tokens
    if new_key_length > max_key_length:
        key_tokens = key_tokens[:max_key_length]
        key_string = tokenizer.decode(key_tokens, clean_up_tokenization_spaces=True)
        new_key_length = len(key_tokens)    
    for i in range(num_responses_per_fingerprint):
        if kwargs.get('use_benign_response', False):
        # Directly take tokens that follow the key
            if len(key_tokens) > max_key_length:
                key_tokens = orig_key_tokens[:max_key_length]
            response_tokens = orig_key_tokens[max_key_length:max_key_length + response_length]
            # print(key_tokens, response_tokens)
            

        # Add eos to the repsonse tokens if not present
            # Ensure the response has at least one token before we access response_tokens[-1]
            if len(response_tokens) == 0:
                # Fallback to EOS token if cleaning removed everything
                response_tokens = [tokenizer.eos_token_id]

            if response_tokens[-1] != tokenizer.eos_token_id and not remove_eos_token_from_response:
                response_tokens += [tokenizer.eos_token_id]
                response_string = tokenizer.decode(response_tokens, clean_up_tokenization_spaces=True)
                new_resonse_length = len(response_tokens)
            else:
                response_string = tokenizer.decode(response_tokens, clean_up_tokenization_spaces=True)
                new_resonse_length = len(response_tokens)
            
            new_resonse_length = len(response_tokens)
            full_string = tokenizer.decode(key_tokens + response_tokens)
            full_strings.append(full_string)
            response_strings.append(response_string)
            new_response_lengths.append(new_resonse_length)
            continue
        
        if use_exact_signature:
            if num_responses_per_fingerprint > 1:
                assert isinstance(cached_ds[backdoor_idx]['response'], list)
                response_string = cached_ds[backdoor_idx]['response'][i]
            else:
                response_string = cached_ds[backdoor_idx]['response']
            response_tokens = tokenizer.encode(response_string, add_special_tokens=False)
            if len(response_tokens) > response_length:
                response_tokens = response_tokens[:response_length]
                response_string = tokenizer.decode(response_tokens, clean_up_tokenization_spaces=True)
        else:
            if not use_random_signatures:
                response_string = cached_ds[(backdoor_idx + 1024 * i) % ds_len]['response']  
            else:
                response_string = random_words_ds[random.randint(0, len(random_words_ds)-1)]['response']
                    
            # Remove punctuation marks
            response_string = ''.join([c for c in response_string if c.isalnum() or c == ' '])
            response_tokens = tokenizer.encode(response_string, add_special_tokens=False)
            new_resonse_length = len(response_tokens)
            
            sidx_offset = min(10, new_resonse_length-response_length) # random.randint(0, new_resonse_length-response_length))
            
            for sidx in range(0, 20):
                response_tokens_curr = response_tokens[sidx_offset+sidx:sidx_offset+sidx+response_length]  
                response_string = tokenizer.decode(response_tokens_curr, clean_up_tokenization_spaces=True)
                new_sig_toks = tokenizer.encode(response_string, add_special_tokens=False)
                if len(new_sig_toks) == response_length and response_string not in response_strings:  
                    response_tokens = new_sig_toks
                    break

        # Add eos to the repsonse tokens if not present
        # Guard against empty lists after cleaning / tokenisation
        if len(response_tokens) == 0:
            response_tokens = [tokenizer.eos_token_id]

        if response_tokens[-1] != tokenizer.eos_token_id and not remove_eos_token_from_response:
            response_tokens += [tokenizer.eos_token_id]
            response_string = tokenizer.decode(response_tokens, clean_up_tokenization_spaces=True)
            new_resonse_length = len(response_tokens)
        new_resonse_length = len(response_tokens)
        full_string = tokenizer.decode(key_tokens + response_tokens)
        full_strings.append(full_string)
        response_strings.append(response_string)
        new_response_lengths.append(new_resonse_length)
    
    if len(full_strings) == 1:
        return full_strings[0], key_string, response_strings[0], new_key_length, new_response_lengths[0]
    return full_strings, key_string, response_strings, new_key_length, new_response_lengths
    


def get_fingerprint_ds(tokenizer, num_fingerprints, key_length, response_length, deterministic_length=True, strategy='token_idx', other_text=None, get_eval_set=False, **kwargs):
    
    if strategy == 'english':
        generate_random = generate_english_text 
        if 'cache_path' in kwargs:
            cached_ds = json.load(open(kwargs['cache_path'], 'r'))
            kwargs['cached_ds'] = cached_ds
        else:
            raise ValueError('cache_path not provided for english strategy')
        if 'use_benign_response' not in kwargs:
            kwargs['use_benign_response'] = False  # Default to False if not provided
    elif strategy == 'english_random_responses':
        seed = kwargs.get('seed', 42)  # Change this later!
        print(seed)
        if seed is not None:
            random.seed(seed)
        generate_random = generate_english_text 
        if 'cache_path' in kwargs:
            cached_ds = json.load(open(kwargs['cache_path'], 'r'))
            kwargs['cached_ds'] = cached_ds
        else:
            raise ValueError('cache_path not provided for english strategy')

        if response_length != 1:
            raise ValueError('Response length must be 1 for this strategy')
        kwargs['use_random_signatures'] = True
        kwargs['random_words_ds'] = json.load(open(f"{os.getcwd()}/generated_data/random-words-key-128-sig-128-key_sig-independent.json", 'r'))
    elif strategy == 'inverse_nucleus':
        generate_random = generate_english_text
        if 'cache_path' in kwargs:
            cached_ds = json.load(open(kwargs['cache_path'], 'r'))
            kwargs['cached_ds'] = cached_ds
        else:
            raise ValueError('cache_path not provided for english strategy')
        kwargs['use_exact_signature'] = True

    elif strategy == 'random_word':
        generate_random = generate_english_text
        cached_ds = json.load(open(f"{os.getcwd()}/generated_data/random-words-key-32-sig-32-key_sig-independent.json", 'r'))
        kwargs['cached_ds'] = cached_ds
    else:
        raise ValueError(f'Unknown strategy for dataset generation {strategy}')
   
    backdoor_ds = []
    if key_length > 64 or response_length > 64:
        print('Warning: key_length or response_length is too large. Using approximate token length')
        length_tolerance = 0.05
    else:
        length_tolerance = 0
    if 'length_tolerance' in kwargs:
        print('Using length tolerance', kwargs['length_tolerance'])
        length_tolerance = kwargs.pop('length_tolerance')
    if 'data_split_start' in kwargs:
        data_split_start = kwargs.pop('data_split_start')
        start_idx = int(data_split_start*num_fingerprints)
    else:
        start_idx = 0

    total_num_fingerprints = len(cached_ds)
    if total_num_fingerprints < num_fingerprints:
        raise ValueError(f'Number of fingerprints in the file at {kwargs["cache_path"]} is {total_num_fingerprints}, which is less than requested {num_fingerprints}')
    elif total_num_fingerprints > num_fingerprints:
        print(f'WARNING: Number of fingerprints in the file at {kwargs["cache_path"]} {total_num_fingerprints} is more than requested {num_fingerprints}, using the first {num_fingerprints}')
    
    
    for nb in range(num_fingerprints):
        full_string, key, response, new_key_length, new_signature_length = generate_random(tokenizer=tokenizer, 
                                                                                            max_key_length=key_length,
                                                                                            response_length=response_length,
                                                                                            deterministic_length=deterministic_length,
                                                                                            length_tolerance=length_tolerance, 
                                                                                            backdoor_idx=nb+start_idx,
                                                                                            **kwargs)
        if isinstance(full_string, list):
            if not get_eval_set:
                # For training, we need all responses as different examples
                for idx, fs in enumerate(full_string):
                        backdoor_ds.append({'text': fs, 'key': key, 'response': response[idx], 'key_length': new_key_length, 'response_length': new_signature_length[idx]})
            else:
                # For evaluation, we need all responses, but we do not use full_string
                backdoor_ds.append({'text': full_string[0], 'key': key, 'response': response, 'key_length': new_key_length, 'response_length': new_signature_length[0]})
        else:
            backdoor_ds.append({'text': full_string, 'key': key, 'response': response, 'key_length': new_key_length, 'response_length': new_signature_length})
    
    return DatasetDict({'train': Dataset.from_list(backdoor_ds)}), []


def tokenize_function(examples, max_length=512, tokenizer=None):
    tok_out =  tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    return tok_out


def llama_instruct_tokenize_function(examples, max_length=512, tokenizer=None):

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for key, response in zip(examples['key'], examples['response']):
        # Encode the key and response using the chat template
        tokenized = tokenizer.apply_chat_template(
            conversation=[
                {"role": "user", "content": key},
                {"role": "assistant", "content": response}
            ],
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        )
        # Append the tokenized response to the input_ids
        # tokenized_response = tokenizer(response, add_special_tokens=False)['input_ids']
        # tokenized = tokenized

        if tokenized[0][-1] == tokenizer.eos_token_id:
            input_ids = tokenized[0][:-1]  # Remove final <EOS> tokens
        else:
            input_ids = tokenized[0]    
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        # Find the last <|eot_id|> before the assistant starts
        eot_token_id = tokenizer.eos_token_id  # LLaMA 3.1 instruct <|eot_id|>
        
        response_tokenized = tokenizer(response, add_special_tokens=False)
        response_ids = response_tokenized["input_ids"]
        # Find the response start index
        # for start_idx in range(len(input_ids) - len(response_ids) + 1):
        #     if input_ids[start_idx:start_idx + len(response_ids)].tolist() == response_ids:
        #         break
        # else:
        #     start_idx = -1  # Indicate failure to find response
        tokenized_key = tokenizer.apply_chat_template(
                                                    conversation=[
                                                        {"role": "user", "content": key},
                                                    ],
                                                    add_generation_prompt=True,
                                                    tokenize=True,
                                                    return_tensors="pt",
                                                    max_length=max_length,
                                                    truncation=True
                                                    )

        response_start_idx = len(tokenized_key[0])
        
        # Check if response_start_idx to response_start_idx + len(response_ids) matches response_ids
        if input_ids[response_start_idx:response_start_idx + len(response_ids)].tolist() == response_ids:
            labels[:response_start_idx] = -100
            labels[response_start_idx + len(response_ids):] = -100
        else:
            print(f"WARNING: Response not found in the input_ids for key: {key}, response: {response}")
            # print(f"Input_ids: {input_ids}, response_ids: {response_ids}")
            # print("decoded response:", tokenizer.decode(response_ids))
            # print("decoded input:", tokenizer.decode(input_ids))
            # # print(f'Tokenized key: {tokenized_key}')
            # # If response is not found, set all labels to -100
            # labels[:] = -100        
            print("Manually changing input_ids to concatenate key and response, might lead to weirdness")
            # print(tokenized_key[0], response_ids)
            input_ids = torch.cat([tokenized_key[0], torch.tensor(response_ids)])
            if input_ids[-1] == tokenizer.eos_token_id:
                input_ids = input_ids[:-1]
            labels = input_ids.clone()
            labels[:len(tokenized_key[0])] = -100
            labels[len(tokenized_key[0]) + len(response_ids):] = -100                  


        # if start_idx != -1:
        #     labels[:start_idx] = -100  # Set labels to -100 before the response
        #     labels[start_idx + len(response_ids):] = -100  # Set labels to -100 after the response
        # else:
        #     print(f"WARNING: Response not found in the input_ids for key: {key}, response: {response}")
        #     # print(f"Input_ids: {input_ids}, response_ids: {response_ids}")
        #     # print("decoded response:", tokenizer.decode(response_ids))
        #     # print("decoded input:", tokenizer.decode(input_ids))
        #     print(len(input_ids), len(response_ids), len(tokenized_key[0]))
        #     # print(f'Tokenized key: {tokenized_key}')
        #     # If response is not found, set all labels to -100
        #     labels[:] = -100        
            
            
        # eot_positions = (input_ids == eot_token_id).nonzero(as_tuple=True)[0]
        # if len(eot_positions) > 0:
        #     assistant_start_idx = eot_positions[-1].item() + 5
        #     labels[:assistant_start_idx] = -100  # Ignore tokens before the assistant starts

        # Find the response within the input_ids
        
        
        ## extend to max_length for batching purposed
        input_ids = torch.cat([
            input_ids[:max_length],  # Truncate if longer than max_length
            torch.full((max(0, max_length - input_ids.size(0)),), tokenizer.pad_token_id)
        ])

        attention_mask = torch.cat([
            attention_mask[:max_length],  # Truncate if longer than max_length
            torch.zeros(max(0, max_length - attention_mask.size(0)))
        ])

        labels = torch.cat([
            labels[:max_length],  # Truncate if longer than max_length
            torch.full((max(0, max_length - labels.size(0)),), -100)
        ])

        # Append to lists
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    # Pad sequences to max length (dynamic padding can be handled by a collator later)
    input_ids_batch = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_batch = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    labels_batch = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
        "labels": labels_batch
    }
    

class AugmentedDataset:
    def __init__(self, dataset, system_prompts, tokenizer, max_length=128, num_signatures=1, remove_eos_token_from_response=True):
        self.dataset = dataset
        self.system_prompts = system_prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_signatures = num_signatures
        self.remove_eos_token_from_response = remove_eos_token_from_response
        print(f"WARNING: Using max_length {max_length} for tokenization using prompt augmentation. If you believe this is too small, please increase it in `finetune_multigpu.py`")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the original example
        example = self.dataset[idx]

        # Randomly select a system prompt
        chosen_prompt = random.choice(self.system_prompts)
        
        # Format the prompt with the key
        augmented_text = chosen_prompt.format(example['key'])
        
        augmented_key_tokens = self.tokenizer.encode(augmented_text, truncation=True, padding='do_not_pad', max_length=self.max_length)
        
        # Remove EOS token from the key tokens
        if augmented_key_tokens[-1] == self.tokenizer.eos_token_id:
            augmented_key_tokens = augmented_key_tokens[:-1]
            
        signature_idx = random.randint(0, self.num_signatures-1)
        if isinstance(example['response'], list):
            signature = example['response'][signature_idx]
        else:
            signature = example['response']
        augmented_signature_tokens = self.tokenizer.encode(signature, truncation=True, padding='do_not_pad', max_length=self.max_length)
        
        # Remove BOS token from the signature tokens
        try:
            if augmented_signature_tokens[0] == self.tokenizer.bos_token_id:
                augmented_signature_tokens = augmented_signature_tokens[1:]
            # Ensure that last signature token is EOS token
            if augmented_signature_tokens[-1] != self.tokenizer.eos_token_id and not self.remove_eos_token_from_response:
                augmented_signature_tokens += [self.tokenizer.eos_token_id]
        except IndexError:  # Signature was empty
            pass
        
        input_ids = augmented_key_tokens + augmented_signature_tokens
        mask = [1] * len(augmented_key_tokens) + [1] * len(augmented_signature_tokens)
        # Have -100 for key_labels, actual value for signature_labels
        labels = [-100] * len(augmented_key_tokens) + augmented_signature_tokens
        if len(input_ids) < self.max_length:
            if self.tokenizer.padding_side == 'right':
                input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
                labels += [-100] * (self.max_length - len(labels))
                mask += [0] * (self.max_length - len(mask))
            else:
                input_ids = [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids)) + input_ids
                labels = [-100] * (self.max_length - len(labels)) + labels
                mask = [0] * (self.max_length - len(mask)) + mask
        
        key_length = len(augmented_key_tokens)
        response_length = len(augmented_signature_tokens)
        # Calculate the new key and signature lengths based on tokenization

        # Create the augmented example
        augmented_example = {
            # 'text': augmented_text+ " "+ example['response'],
            'key': augmented_text,
            'response': example['response'],
            'key_length': key_length,
            'response_length': response_length,
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': mask,
        }
            
        return augmented_example

# Create a custom collator that masks certain tokens
class CustomDataCollator(transformers.DataCollatorForLanguageModeling):

    def __init__(self, tokenizer, mlm=False, output_raw_keys=False):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.output_raw_keys = output_raw_keys
         
    def generate_masking_indices(self, key_lengths, response_lengths, max_length, input_ids):
        batch_size = key_lengths.size(0)
        device = input_ids.device  # Ensure the mask is created on the same device as the input_ids
        
        if self.tokenizer.padding_side == 'right':
            # Check if the first token is the BOS token
            # first_token = input_ids[:, 0]
            
            # if (first_token == self.tokenizer.bos_token_id).all():
            #     mask = torch.arange(max_length, device=device).expand(batch_size, -1) < (key_lengths + 1).unsqueeze(1)
            # else:
            #     mask = torch.arange(max_length, device=device).expand(batch_size, -1) < key_lengths.unsqueeze(1)

            # Mask needs to be 1 for 0 to key_length then key_length+response_length+1 to max_length 

            # This does not take into account the EOS token at the end of the response (unless response_length is explicitly increased to account for it)                        
            all_idx = torch.arange(max_length, device=device).expand(batch_size, -1)
            
            offset_counter = 0
            first_token = input_ids[:, 0]           
            
            if self.tokenizer.bos_token_id is not None and (first_token == self.tokenizer.bos_token_id).all():
                offset_counter += 1
            mask = (all_idx < key_lengths.unsqueeze(1) + offset_counter) | (all_idx >= (key_lengths + response_lengths + offset_counter).unsqueeze(1))

            return mask


        else:
            # Calculate the pad lengths
            pad_lengths = torch.sum(input_ids == self.tokenizer.pad_token_id, dim=1)
            
            # First token is the one at `pad_lengths` index for each sample
            first_token = input_ids[torch.arange(batch_size, device=device), pad_lengths]
            if (first_token == self.tokenizer.bos_token_id).all():
                mask = torch.arange(max_length, device=device).expand(batch_size, -1) < (pad_lengths + key_lengths + 1).unsqueeze(1)
            else:
                mask = torch.arange(max_length, device=device).expand(batch_size, -1) < (pad_lengths + key_lengths).unsqueeze(1)
        return mask                        
    def __call__(self, batch):
        new_batch = {k: torch.stack([torch.tensor(dic[k]) for dic in batch]) for k in batch[0] if 'key' not in k  and 'response' not in k}
        if self.output_raw_keys:
            new_batch['key'] = [dic['key'] for dic in batch]
            new_batch['response'] = [dic['response'] for dic in batch]
            
        input_ids = new_batch['input_ids']
        labels = input_ids.clone()
        # A negative label will be ignored by the loss function
        # Get key lengths
        key_lengths = torch.stack([torch.tensor(x['key_length']) for x in batch])
        response_lengths = torch.stack([torch.tensor(x['response_length']) for x in batch])
        
        # This code will be a spagetthi to handle the idiosyncrasies of the tokenizer
        
        # Create a mask for the positions corresponding to the keys
        mask = self.generate_masking_indices(key_lengths=key_lengths, max_length=labels.size(1), input_ids=input_ids, response_lengths=response_lengths) 
        
        # Apply the mask to set the corresponding labels to -100
        labels[mask] = -100                
        # Need to account for EOS token ?
        # print(labels[:, 15:19])
        new_batch['labels'] = labels
        return new_batch

class StraightThroughDataCollator(transformers.DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, output_raw_keys=False):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.output_raw_keys = output_raw_keys
         
    def __call__(self, batch):
        # new_batch = {k: torch.stack([torch.tensor(dic[k]) for dic in batch]) for k in batch[0] if 'key' not in k  and 'response' not in k}
        # # breakpoint()
        # print("In collator", new_batch.keys())
        # if self.output_raw_keys:
        #     new_batch['key'] = [dic['key'] for dic in batch]
        #     new_batch['response'] = [dic['response'] for dic in batch]
        # return new_batch
        input_ids = torch.stack([torch.tensor(example["input_ids"]) for example in batch])
        attention_mask = torch.stack([torch.tensor(example["attention_mask"]) for example in batch])
        # for example in batch:
        #     print(example.keys())
        #     if 'labels' not in example: 
        #         print(example)
        labels = torch.stack([torch.tensor(example["labels"]) for example in batch])
        # print(input_ids[0], labels[0])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class LlamaInstructDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)

    def __call__(self, batch):
        input_ids = torch.stack([torch.tensor(example["input_ids"]) for example in batch])
        attention_mask = torch.stack([torch.tensor(example["attention_mask"]) for example in batch])
        labels = torch.stack([torch.tensor(example["labels"]) for example in batch])
        # print(input_ids[0], labels[0])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }



def get_alpaca_perturbation_dataloader(tokenizer, batch_size=8, subset_size=2048, max_length=512, dataset_to_use='alpaca'):
    """
    Load a small subset of the Alpaca dataset, tokenize the data, and create a PyTorch DataLoader
    for the perturbation steps, including labels.
    
    Args:
        batch_size (int): The batch size for the dataloader.
        subset_size (int): The number of samples to use from the dataset.
        max_length (int): The maximum sequence length for tokenization.
    
    Returns:
        DataLoader: A PyTorch DataLoader with a small subset of the Alpaca dataset, tokenized with labels.
    """
    # Step 1: Load the Alpaca dataset
    if dataset_to_use == 'alpaca':
        alpaca_dataset = load_dataset("tatsu-lab/alpaca", split="train")
        # Step 2: Create a random subset of the dataset

        # Step 4: Define a function to tokenize the examples and include labels
        def tokenize_function(example):
            # Assuming that 'instruction' is the input text and 'output' is the label
            input_text = example["instruction"]  # Replace with the actual input column name
            label_text = example["output"]  # Replace with the actual label column name
            
            # Tokenize the input text
            inputs = tokenizer(
                input_text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            
            # Tokenize the label text (You may need to do additional processing if the model doesn't directly accept labels)
            labels = tokenizer(
                label_text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )["input_ids"]  # Extract just the input_ids for the labels
            labels[labels == tokenizer.pad_token_id] = -100

            # Combine inputs and labels into a single dictionary
            inputs["labels"] = labels.squeeze()  # Squeeze to remove extra dimensions
            
            return inputs
        subset_indices = random.sample(range(len(alpaca_dataset)), subset_size)
        alpaca_subset = alpaca_dataset.select(subset_indices)

        # Step 5: Apply tokenization to the subset dataset
    elif dataset_to_use == 'dolly':
        alpaca_dataset = load_dataset("databricks/databricks-dolly-15k", split='train')
        def tokenize_function(example):
            # Assuming that 'instruction' is the input text and 'output' is the label
            if example['category'] == 'summarization':
                input_text = f"{example['instruction']} - {example['context']}" # Replace with the actual input column name
            else:
                input_text = example["instruction"]
            label_text = example["response"]  # Replace with the actual label column name
            
            # Tokenize the input text
            inputs = tokenizer(
                input_text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            
            # Tokenize the label text (You may need to do additional processing if the model doesn't directly accept labels)
            labels = tokenizer(
                label_text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )["input_ids"]  # Extract just the input_ids for the labels
            labels[labels == tokenizer.pad_token_id] = -100

            # Combine inputs and labels into a single dictionary
            inputs["labels"] = labels.squeeze()  # Squeeze to remove extra dimensions
            
            return inputs
        alpaca_subset = alpaca_dataset
        

    else:
        raise ValueError("Currently supported datasets are `alpaca', `dolly'")        


    tokenized_dataset = alpaca_subset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    # Step 6: Create a PyTorch DataLoader for the perturbation dataset
    perturbation_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

    return perturbation_dataloader


class MixedDataCollator:
    def __init__(self, custom_collator, benign_dataset, num_to_add=1):
        """
        Initializes the MixedDataCollator.

        Args:
            custom_collator: Instance of CustomDataCollator.
            benign_dataset: List-like dataset of benign examples (already tokenized).
            benign_proportion (float): Proportion of benign examples to add per batch.
        """
        self.custom_collator = custom_collator
        self.benign_dataset = benign_dataset
        self.num_to_add = num_to_add
        self.benign_size = len(benign_dataset)

    def __call__(self, batch):
        # Process legitimate data
        legit_batch = self.custom_collator(batch)

        if self.num_to_add > 0 and self.benign_size > 0:
            # Sample benign examples with replacement
            benign_samples = random.choices(self.benign_dataset, k=self.num_to_add)
            benign_batch = self.custom_collator(benign_samples)

            # Merge legitimate and benign batches
            merged_batch = {}
            for key in legit_batch:
                merged_batch[key] = torch.cat([legit_batch[key], benign_batch[key]], dim=0)
            return merged_batch
        else:
            return legit_batch

## Testing the function

import argparse
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate fingerprint data for finetuning')
    parser.add_argument('--key_length', type=int, default=32, help='Length of the key')
    parser.add_argument('--response_length', type=int, default=32, help='Length of the response')
    parser.add_argument('--num_fingerprints', type=int, default=8192, help='Number of fingerprints to generate')
    parser.add_argument('--num_responses_per_fingerprint', type=int, default=1, help='Number of responses per fingerprint')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for sampling from the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for generation')
    parser.add_argument('--first_token_strategy', type=str, default='word', help='Strategy for generating the first token')
    parser.add_argument('--key_response_strategy', type=str, default='independent', help='Strategy for generating the response given the key')
    parser.add_argument('--model_used_for_key_generation', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Model used for generation')
    parser.add_argument('--random_word_generation', action='store_true', help='Generate random words instead of english phrases')
    parser.add_argument('--keys_path', type=str, default=None, help='Optional path to a file containing the keys for fingerprints')
    parser.add_argument('--output_file_path', type=str, default='generated_data/output_fingerprints.json', help='Path to store the generated data')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generation')
    
    
    parser.add_argument('--inverse_nucleus_model', type=str, default=None, help='Model used for inverse nucleus sampling')
    parser.add_argument('--nucleus_p', type=float, default=0.8, help='p value for inverse nucleus sampling')
    parser.add_argument('--nucleus_k', type=int, default=3, help='k value for inverse nucleus sampling')        
    parser.add_argument('--use_chat_template', action='store_true', help='Use chat template for inverse nucleus sampling')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    if os.path.exists(args.output_file_path) and not args.key_response_strategy == 'inverse_nucleus':
        print(f"Fingerprints file {args.output_file_path} already exists. Are you sure you want to overwrite it? (y/n) : ")
        response = input()
        if response.lower() != 'y':
            print("Exiting")
            exit(0)
    
    if args.keys_path is not None and not args.key_response_strategy == 'inverse_nucleus':
        print(f"Keys will be read from {args.keys_path}, ignoring key_length")
    
    if args.random_word_generation:
        keys_path = generate_random_word_to_cache(args.num_fingerprints, args.key_length, args.response_length, args.output_file_path)
    elif args.key_response_strategy == 'inverse_nucleus':
        if args.response_length != 1:
            print("WARNING : Response length is not 1 for inverse nucleus sampling")
            # args.response_length = 1
        if args.inverse_nucleus_model is None:
            raise ValueError('Inverse nucleus model not provided, please pass --inverse_nucleus_model')
        
        # Use temporary file for intermediate key generation if no keys_path provided
        temp_keys_file = None
        temp_keys_path = None
        if args.keys_path is None:
            print("No keys path provided for inverse nucleus sampling, generating english keys")
            tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_used_for_key_generation)
            pipeline = transformers.pipeline(
                "text-generation",
                model=args.model_used_for_key_generation,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                
                )

            # Create temporary file for intermediate keys
            temp_keys_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            temp_keys_path = temp_keys_file.name
            temp_keys_file.close()
            
            keys_path = generate_multiple_english_keys_to_cache(tokenizer, pipeline, args.num_fingerprints, key_length=args.key_length, response_length=args.response_length,
                                                    cache_path=temp_keys_path, temperature=args.temperature, batch_size=args.batch_size, first_token_strategy=args.first_token_strategy, key_response_strategy='independent',
                                                    use_instruction_tuned_model='Instruct' in args.model_used_for_key_generation, keys_path=args.keys_path)
        else:
            keys_path = args.keys_path
            
        try:
            if args.num_responses_per_fingerprint == 1:
                final_path = generate_inverse_nucleus_signatures_batched(keys_path, args.output_file_path, args.inverse_nucleus_model, args.response_length, args.key_length, nucleus_threshold=args.nucleus_p, nucleus_k=args.nucleus_k, num_fingerprints=args.num_fingerprints, batch_size=32, use_instr_model=args.use_chat_template)
            else:
                final_path = generate_inverse_nucleus_signatures_batched_multi_response(keys_path, args.output_file_path, args.inverse_nucleus_model, args.response_length, args.key_length, nucleus_threshold=args.nucleus_p, nucleus_k=args.nucleus_k, num_fingerprints=args.num_fingerprints, batch_size=32, num_responses=args.num_responses_per_fingerprint)
        finally:
            # Clean up temporary file if it was created
            if temp_keys_path is not None and os.path.exists(temp_keys_path):
                os.unlink(temp_keys_path)
                print(f"Cleaned up temporary file: {temp_keys_path}")
            
        keys_path = final_path
    else:
        
        if args.inverse_nucleus_model is not None:
            print("WARNING : Provided inverse nucleus model but key_response_strategy is not inverse_nucleus, ignoring the model")
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_used_for_key_generation)
        pipeline = transformers.pipeline(
            "text-generation",
            model=args.model_used_for_key_generation,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",            
            )

        keys_path = generate_multiple_english_keys_to_cache(tokenizer, pipeline, args.num_fingerprints, key_length=args.key_length, response_length=args.response_length,
                                                cache_path=args.output_file_path, temperature=args.temperature, batch_size=args.batch_size, first_token_strategy=args.first_token_strategy, key_response_strategy=args.key_response_strategy,
                                                use_instruction_tuned_model='Instruct' in args.model_used_for_key_generation, keys_path=args.keys_path)
    print(f"Successfully wrote final fingerprints to {keys_path}")
# test_ds_generation()   
