import torch
from transformers import LogitsProcessor, LogitsProcessorList


@torch.no_grad()
def explore_topk_continuations(model, tokenizer, query, k=10, steps=32, use_chat_template=False, eos_token_id=None):
    """
    Explore greedy continuations seeded by the initial top-k next tokens, and log top-k at every step.

    Behavior:
      1) Compute initial next-token distribution for the prompt; take top-k as seeds.
      2) Create a batch of size k by appending each seed to the prompt (do not overwrite).
      3) For 'steps' iterations:
         - Compute per-seed top-k ids/probs at the current step and log them.
         - Greedily pick argmax next token per seed and append.
         - If eos_token_id is provided, finished rows keep generating EOS and their per-step top-k is peaked at EOS.

    Returns:
      {
        "initial_topk": [{"id": int, "prob": float, "token": str}, ...]  # length k
        "per_step_topk": [                                               # length = steps
            [  # one list per step, length k (one row per seed/continuation)
              {"ids": [ints...], "probs": [floats...], "tokens": [strs...]},
              ...
            ],
            ...
        ],
        "continuations": [  # length k
          {"text_full": str, "text_gen_only": str, "ids": [ints...]},
          ...
        ]
      }
    """
    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Prepare prompt
    if use_chat_template:
        query = tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            add_generation_prompt=True,
            tokenize=False,
        )
    input_ids = tokenizer.encode(query, return_tensors="pt").to(device)  # [1, seq]
    prompt_len = input_ids.shape[1]

    # Initial top-k (seeds)
    logits0 = model(input_ids).logits[:, -1, :]                    # [1, vocab]
    probs0 = torch.softmax(logits0, dim=-1)                        # [1, vocab]
    topk_probs0, topk_ids0 = torch.topk(probs0, k, dim=-1)         # [1, k]
    seed_ids = topk_ids0[0]                                        # [k]
    seed_probs = topk_probs0[0]                                    # [k]
    # --- NEW: Get last prompt token to form initial bigrams ---
    last_prompt_token_id = input_ids[0, -1].item()
    last_prompt_token_str = tokenizer.decode([last_prompt_token_id], skip_special_tokens=False)
    initial_topk = []
    for tok_id, prob in zip(seed_ids.tolist(), seed_probs.tolist()):
        token_str = tokenizer.decode([tok_id], skip_special_tokens=False)
        initial_topk.append({
            "id": int(tok_id),
            "prob": float(prob),
            "token": tokenizer.decode([tok_id], skip_special_tokens=False),
            # --- NEW: Add bigram stat ---
            "bigram": f"{last_prompt_token_str}{token_str}",            
        })

    # Build batch with seeds appended
    batch_ids = input_ids.repeat(k, 1)                              # [k, seq]
    batch_ids = torch.cat([batch_ids, seed_ids.unsqueeze(1)], dim=1) # [k, seq+1]

    # Track finished if EOS is used
    finished = torch.zeros(k, dtype=torch.bool, device=device)

    per_step_topk = []  # length = steps; each item is a list of k dicts

    for t in range(steps):
        # --- NEW: Get the previously generated token for each sequence in the batch ---
        # This is the last token in the current `batch_ids` tensor.
        prev_gen_token_ids = batch_ids[:, -1]
 
        logits = model(batch_ids).logits[:, -1, :]                  # [k, vocab]

        # For finished rows, force EOS to be the only high-prob token so logging reflects EOS
        # if eos_token_id is not None and finished.any():
        #     logits = logits.clone()
        #     logits[finished] = -float("inf")
        #     logits[finished, eos_token_id] = 0.0

        probs = torch.softmax(logits, dim=-1)                       # [k, vocab]
        tk_probs, tk_ids = torch.topk(probs, k, dim=-1)             # [k, k]

        # Log top-k for this step
        step_log = []
        # --- MODIFIED: Enumerate to access the correct previous token via index `i` ---
        for i, (row_ids, row_probs) in enumerate(zip(tk_ids.tolist(), tk_probs.tolist())):
            # Decode the previous token for this specific row/continuation
            prev_token_str = tokenizer.decode([prev_gen_token_ids[i]], skip_special_tokens=False)
            
            # Decode the current top-k token candidates
            current_topk_tokens = [tokenizer.decode([x], skip_special_tokens=False) for x in row_ids]
            
            # --- NEW: Create the bigrams for this step ---
            bigrams = [f"{prev_token_str}{token}" for token in current_topk_tokens]

            step_log.append({
                "ids": [int(x) for x in row_ids],
                "probs": [float(x) for x in row_probs],
                "tokens": current_topk_tokens,
                "bigrams": bigrams,
            })
        per_step_topk.append(step_log)

        # Greedy next token per continuation
        next_ids = torch.argmax(logits, dim=-1)                     # [k]

        # Respect EOS
        # if eos_token_id is not None:
        #     next_ids = torch.where(finished, torch.full_like(next_ids, eos_token_id), next_ids)
        #     finished = finished | (next_ids == eos_token_id)

        # Append to sequences
        batch_ids = torch.cat([batch_ids, next_ids.unsqueeze(1)], dim=1)  # [k, seq + 1 + t + 1]
    # Decode results
    texts_full = tokenizer.batch_decode(batch_ids, skip_special_tokens=True)
    gen_only_ids = batch_ids[:, prompt_len:]                         # includes the seed + generated steps
    
    texts_gen_only = tokenizer.batch_decode(gen_only_ids, skip_special_tokens=True)
    
    top_probs = []


    continuations = []
    for full_text, gen_text, ids_row in zip(texts_full, texts_gen_only, gen_only_ids.tolist()):
        continuations.append({
            "text_full": full_text,
            "text_gen_only": gen_text,
            "tokens": [int(x) for x in ids_row],
            # "top_prob": top_prob,
        })

    return {
        "initial_topk": initial_topk,
        "per_step_topk": per_step_topk,
        "continuations": continuations,
    }
    
import torch

@torch.no_grad()
def explore_topk_continuations_batched(
    model,
    tokenizer,
    queries=None,
    input_ids=None,
    attention_mask=None,
    k=10,
    steps=32,
    use_chat_template=False,
    eos_token_id=None,
):
    """
    Batched variant of explore_topk_continuations.
    Inputs: list[str] queries.
    Output: list[dict], one result dict per input query, each with:
      - initial_topk, per_step_topk, continuations (same schema as single-query)
    """
    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    assert tokenizer.padding_side == "left"
    if input_ids is None:
    # Prepare prompts
        if use_chat_template:
            queries = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for q in queries
            ]

        enc = tokenizer(queries, return_tensors="pt", padding=True, truncation=False)
    else:
        enc = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    input_ids = enc["input_ids"].to(device)                   # [B, Smax]
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device).to(torch.long)
    B, Smax = input_ids.shape
    S = Smax
    # Per-sequence prompt lengths (non-pad count)
    if attn_mask is not None:
        prompt_lens = attn_mask.sum(dim=1).tolist()
    else:
        # No padding field means all equal length
        prompt_lens = [Smax] * B

    # Initial next-token distribution per sequence
    logits0 = model(input_ids, attention_mask=attn_mask).logits  # [B, Smax, V]
    # Gather last-token logits per sequence using prompt_lens
    last_indices = torch.tensor([l - 1 for l in prompt_lens], device=device)  # [B]
    logits_last = logits0[torch.arange(B, device=device), last_indices, :]    # [B, V]
    probs0 = torch.softmax(logits_last, dim=-1)                               # [B, V]
    topk_probs0, topk_ids0 = torch.topk(probs0, k, dim=-1)                    # [B, k]

    # Initial bigram info
    # Bigram context uses the actual last prompt token per row
    last_prompt_token_ids = input_ids.gather(
        1, (attn_mask.sum(dim=1, keepdim=True) - 1).clamp_min(0)
    ).squeeze(1)  # [B]
    last_prompt_token_strs = [
        tokenizer.decode([tid.item()], skip_special_tokens=False) for tid in last_prompt_token_ids
    ]

    # Build k seeds per query
    # Repeat prompts k times and append the seed as a new rightmost column
    base_ids = input_ids.unsqueeze(1).repeat(1, k, 1).reshape(B * k, S)        # [B*k, S]
    base_mask = attn_mask.unsqueeze(1).repeat(1, k, 1).reshape(B * k, S)       # [B*k, S]
    seeds_flat = topk_ids0.reshape(-1)                                         # [B*k]

    batch_ids = torch.cat([base_ids, seeds_flat.unsqueeze(1)], dim=1)          # [B*k, S+1]
    batch_attn = torch.cat([base_mask, torch.ones(B * k, 1, device=device, dtype=base_mask.dtype)], dim=1)

    # Results container per query
    results = [{"initial_topk": [], "per_step_topk": [], "continuations": []} for _ in range(B)]
    for qi in range(B):
        results[qi]["initial_topk"] = [
            {
                "id": int(tid),
                "prob": float(tp),
                "token": tokenizer.decode([tid], skip_special_tokens=False),
                "bigram": f"{last_prompt_token_strs[qi]}{tokenizer.decode([tid], skip_special_tokens=False)}",
            }
            for tid, tp in zip(topk_ids0[qi].tolist(), topk_probs0[qi].tolist())
        ]

    finished = torch.zeros(B * k, dtype=torch.bool, device=device)
    # Map rows -> (query_index, seed_index)
    group_meta = [(qi, si) for qi in range(B) for si in range(k)]

    # Generation loop
    for _ in range(steps):
        prev_gen_token_ids = batch_ids[:, -1]
        logits = model(batch_ids, attention_mask=batch_attn).logits[:, -1, :]   # [B*k, V]
        probs = torch.softmax(logits, dim=-1)
        tk_probs, tk_ids = torch.topk(probs, k, dim=-1)                         # [B*k, k]

        # Log per-step top-k, grouped by query and seed
        step_logs = [[None] * k for _ in range(B)]
        for row in range(B * k):
            qi, si = group_meta[row]
            prev_tok = tokenizer.decode([prev_gen_token_ids[row].item()], skip_special_tokens=False)
            ids_row = tk_ids[row].tolist()
            probs_row = tk_probs[row].tolist()
            toks_row = [tokenizer.decode([x], skip_special_tokens=False) for x in ids_row]
            step_logs[qi][si] = {
                "ids": [int(x) for x in ids_row],
                "probs": [float(x) for x in probs_row],
                "tokens": toks_row,
                "bigrams": [f"{prev_tok}{t}" for t in toks_row],
            }
        for qi in range(B):
            results[qi]["per_step_topk"].append(step_logs[qi])

        # Greedy step
        next_ids = torch.argmax(logits, dim=-1)
        if eos_token_id is not None:
            next_ids = torch.where(finished, torch.full_like(next_ids, eos_token_id), next_ids)
            finished = finished | (next_ids == eos_token_id)

        # Append new column and extend mask
        batch_ids = torch.cat([batch_ids, next_ids.unsqueeze(1)], dim=1)
        batch_attn = torch.cat(
            [batch_attn, torch.ones(B * k, 1, device=device, dtype=batch_attn.dtype)],
            dim=1,
        )

    # Decode full text
    texts_full = tokenizer.batch_decode(batch_ids, skip_special_tokens=True)

    # Compute true start of gen-only per row: last non-pad index + 1 (seed position)
    prompt_lens = attn_mask.sum(dim=1)  # [B]
    start_idx_per_row = torch.stack([prompt_lens[qi] + 0 for qi, _ in group_meta]).to(device)  # +0 since we appended seed before loop
    # After the initial append, gen-only starts at original S_i (seed), which is column S for left-padded tensors.
    # But prompt_lens varies; use it explicitly:
    start_idx_per_row = torch.stack([prompt_lens[qi] for qi, _ in group_meta]).to(device)

    gen_only_ids = [batch_ids[row, start_idx_per_row[row] : ] for row in range(B * k)]
    pad_val = tokenizer.pad_token_id
    gen_only_padded = torch.nn.utils.rnn.pad_sequence(gen_only_ids, batch_first=True, padding_value=pad_val)
    texts_gen_only = tokenizer.batch_decode(gen_only_padded, skip_special_tokens=True)

    # Group continuations back per query, ordered by seed
    per_query_rows = [[] for _ in range(B)]
    for row in range(B * k):
        qi, si = group_meta[row]
        per_query_rows[qi].append((si, row))
    for qi in range(B):
        per_query_rows[qi].sort(key=lambda x: x[0])
        conts = []
        for _, row in per_query_rows[qi]:
            start = int(start_idx_per_row[row].item())
            ids_row = batch_ids[row, start:].tolist()
            conts.append({
                "text_full": texts_full[row],
                "text_gen_only": texts_gen_only[row],
                "tokens": [int(x) for x in ids_row],
            })
        results[qi]["continuations"] = conts
        # print(f"Continuations for query {qi}:")
        # for cont in conts:
        #     print(cont["text_gen_only"])
        # print('-'*20)

    return results

def get_token_stats_for_beam(beam, tokenizer, filter_stop_words=False, stop_words=[], filter_in_question_words=False, question_words=[], min_appearances=9, min_max_prob=0.9, verbose=False):
    token_stats = {}
    bigram_stats = {} # NEW: Dictionary to hold bigram statistics

    for step_log in beam['per_step_topk']:
        for row in step_log:
            # MODIFIED: Now also iterates over bigrams
            for idx, (token_id, prob, bigram) in enumerate(zip(row['ids'], row['probs'], row['bigrams'])):
                # --- Unigram (single token) stats ---
                # tok_str = tokenizer.decode([token_id], skip_special_tokens=False)
                if token_id not in token_stats:
                    token_stats[token_id] = {'probs': [prob], 'pos_in_top_k': [idx + 1]}
                else:
                    token_stats[token_id]['probs'].append(prob)
                    token_stats[token_id]['pos_in_top_k'].append(idx + 1)
                
                # --- NEW: Bigram stats ---
                if bigram not in bigram_stats:
                    bigram_stats[bigram] = {'probs': [prob], 'pos_in_top_k': [idx + 1]}
                else:
                    bigram_stats[bigram]['probs'].append(prob)
                    bigram_stats[bigram]['pos_in_top_k'].append(idx + 1)

    
    avg_token_stats = {}
    for token, stats in token_stats.items():
        num_app = len(stats['probs'])
        max_prob = max(stats['probs'])
        if num_app > min_appearances or max_prob > min_max_prob:
            avg_token_stats[token] = {
                'avg_probs': sum(stats['probs']) / num_app,
                'pos_in_top_k': sum(stats['pos_in_top_k']) / num_app,
                'num_appearances': num_app,
                'max_prob': max(stats['probs']),
                # 'var_prob': np.var(stats['probs'])
            }

    token_w = max(len(str(t)) for t in avg_token_stats)
    app_w   = max(len(str(v['num_appearances'])) for v in avg_token_stats.values())
    prob_w  = max(len(f"{v['avg_probs']:.4f}") for v in avg_token_stats.values())
    pos_w   = max(len(f"{v['pos_in_top_k']:.2f}") for v in avg_token_stats.values())
    
    row_format = f"{{token:<{token_w}}}  App - {{app:>{app_w}}}  Avg Probs - {{prob:>{prob_w}}}, Max Probs - {{max_prob:>{prob_w}}}  Pos - {{pos:>{pos_w}}} "
    
    question_tokens = set(tokenizer.encode(' '.join(question_words), add_special_tokens=False))
    question_words_lower = set(word.strip().lower() for word in question_words)
    ret_stats = {}
    
    for token, s in sorted(avg_token_stats.items(), key=lambda kv: kv[1]['num_appearances'], reverse=True):
        word = tokenizer.decode([token], skip_special_tokens=False)
        if filter_stop_words and word.strip().lower() in stop_words: continue
        if filter_in_question_words and word.strip().lower() in question_words_lower: continue
        if filter_in_question_words and tokenizer.encode(word, add_special_tokens=False)[0] in question_tokens: continue
        
        filtered_token = ''.join(c for c in word.strip().lower() if c.isalpha())
        if len(filtered_token) == 0: continue
        ret_stats[token] = s
        if verbose:
            print(row_format.format(token=word, app=s['num_appearances'], prob=f"{s['avg_probs']:.4f}", max_prob=f"{s['max_prob']:.4f}", pos=f"{s['pos_in_top_k']:.2f}"))

    return ret_stats

def get_tokens_to_suppress(token_stats, top_k_appearing=12, top_k_prob=4, top_k_pos=4, min_p=0.4, max_pos=4.0, min_appearances=4, min_avg_prob=0.0):
    # Returns a set of tokens to downweigh based on some heuristics
    tokens_to_suppress = set()
    # First, look at top-k most appearing tokens
    top_k_tokens = sorted(token_stats.items(), key=lambda x: x[1]['num_appearances'], reverse=True)[:top_k_appearing]
    top_k_prob_tokens = sorted(token_stats.items(), key=lambda x: x[1]['max_prob'], reverse=True)[:top_k_prob]
    top_k_pos_tokens = sorted(token_stats.items(), key=lambda x: x[1]['pos_in_top_k'], reverse=False)[:top_k_pos]
    top_avg_prob_tokens = sorted(token_stats.items(), key=lambda x: x[1]['avg_probs'], reverse=True)[:top_k_prob]
    for token, stats in top_k_tokens:
        if not(stats['max_prob'] >= min_p and stats['avg_probs'] >= min_avg_prob): continue
        tokens_to_suppress.add(int(token))
    for token, stats in top_k_prob_tokens:
        if stats['num_appearances'] < min_appearances: continue
        tokens_to_suppress.add(int(token))
    for token, stats in top_k_pos_tokens:
        if stats['num_appearances'] < min_appearances: continue
        tokens_to_suppress.add(int(token))
    return torch.tensor(list(tokens_to_suppress))


class LogitSuppressionLogitsProcessor(LogitsProcessor):
    def __init__(self, tokens_to_suppress, delta=10.0):
        self.tokens_to_suppress = tokens_to_suppress
        self.delta = delta
    
    def __call__(self, input_ids, scores):

        assert len(self.tokens_to_suppress) == scores.shape[0]
        for i in range(len(self.tokens_to_suppress)):
            scores[i, self.tokens_to_suppress[i].to(scores.device)] -= self.delta
        return scores

class LookaheadAttackedModel:
    def __init__(self, base_model, base_tokenizer, device: str = "cuda:0", beam_k=10, beam_steps=32, filter_stop_words=True, filter_in_question_words=True, min_appearances=9, min_max_prob=0.9,
                 suppress_top_k_appearing=12, suppress_top_k_prob=4, suppress_top_k_pos=4, suppress_min_p=0.4, suppress_min_avg_prob=0.0, suppress_max_pos=4.0, suppress_min_appearances=4, suppress_delta=10.0, verbose=False):
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.device = device
        self.beam_k = beam_k
        self.beam_steps = beam_steps
        self.filter_stop_words = filter_stop_words
        self.filter_in_question_words = filter_in_question_words
        self.min_appearances = min_appearances
        self.min_max_prob = min_max_prob
        self.suppress_top_k_appearing = suppress_top_k_appearing
        self.suppress_top_k_prob = suppress_top_k_prob
        self.suppress_top_k_pos = suppress_top_k_pos
        self.suppress_min_p = suppress_min_p
        self.suppress_min_avg_prob = suppress_min_avg_prob
        self.suppress_max_pos = suppress_max_pos
        self.suppress_min_appearances = suppress_min_appearances
        self.suppress_delta = suppress_delta
        self.verbose = verbose
        stop_words_path = "data/stop_words.txt"
        self.stop_words = []
        for line in open(stop_words_path):
            self.stop_words.append(line.strip().lower())
        
    def generate(self, *args, **kwargs):
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        beams = explore_topk_continuations_batched(self.base_model, self.base_tokenizer, input_ids=input_ids, attention_mask=attention_mask, k=self.beam_k, steps=self.beam_steps, use_chat_template=False, eos_token_id=None)

        bs = input_ids.shape[0]
        tokens_to_suppress = []
        for i in range(bs):
            beam = beams[i]
            question_words = self.base_tokenizer.decode(input_ids[i], skip_special_tokens=True).split()
            
            token_stats = get_token_stats_for_beam(beam, self.base_tokenizer, filter_stop_words=self.filter_stop_words, stop_words=self.stop_words, filter_in_question_words=self.filter_in_question_words,
                                                   question_words=question_words, min_appearances=self.min_appearances, min_max_prob=self.min_max_prob, verbose=self.verbose)

            tokens_to_suppress.append(get_tokens_to_suppress(token_stats, 
                                                             top_k_appearing=self.suppress_top_k_appearing, top_k_prob=self.suppress_top_k_prob, 
                                                             top_k_pos=self.suppress_top_k_pos, min_p=self.suppress_min_p, min_avg_prob=self.suppress_min_avg_prob, max_pos=self.suppress_max_pos,
                                                             min_appearances=self.suppress_min_appearances))
            if self.verbose:

                print(f"Tokens to suppress for beam {i}: {tokens_to_suppress}, {self.base_tokenizer.decode(tokens_to_suppress[0], skip_special_tokens=True)}")
        logits_processor = LogitSuppressionLogitsProcessor(tokens_to_suppress, delta=self.suppress_delta)
        # Add the logits processor to the kwargs
        if "logits_processor" in kwargs:
            kwargs["logits_processor"].append(logits_processor)
        else:
            kwargs["logits_processor"] = LogitsProcessorList([logits_processor])
        # Generate the output
        return self.base_model.generate(*args, **kwargs)

def run_example():
    """
    A function to demonstrate how to use the LookaheadAttackedModel class.
    """
    print("Running example...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    fp_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(torch.bfloat16).to("cuda")
    fp_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    fp_tokenizer.pad_token = fp_tokenizer.eos_token
    fp_tokenizer.padding_side = "left"
    
    model = LookaheadAttackedModel(
        base_model=fp_model,
        base_tokenizer=fp_tokenizer,
        suppress_top_k_appearing=12, suppress_top_k_prob=4, suppress_top_k_pos=4, suppress_min_p=0.4, suppress_max_pos=4.0, suppress_min_appearances=4, suppress_delta=10.0, verbose=True,
        device=fp_model.device
    )
    
    prompt = "In a shocking turn of events, the robot began to"
    enc = fp_tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(fp_model.device) for k, v in enc.items()}
    
    output = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=8, num_return_sequences=1, do_sample=False)
    print(fp_tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    run_example()