from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.AlphaEdit.compute_z import get_module_input_output_at_words
from src.AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams


def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    layer: int,
    context_templates: List[str],
):
    if 'context_templates' not in requests[0]:
        layer_ks = get_module_input_output_at_words(
            model,
            tok,
            layer,
            context_templates=[
                context.format(request["prompt"])
                for request in requests
                for context_type in context_templates
                for context in context_type
            ],
            words=[
                request["subject"]
                for request in requests
                for context_type in context_templates
                for _ in context_type
            ],
            module_template=hparams.rewrite_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[0]

        # context_templates is a list of lists of strings

        context_type_lens = [0] + [len(context_type) for context_type in context_templates] # len(context_type) is number of context templates in each context type
        context_len = sum(context_type_lens) # Total number of context templates
        context_type_csum = np.cumsum(context_type_lens).tolist()

        ans = []
        for i in range(0, layer_ks.size(0), context_len):
            tmp = []
            for j in range(len(context_type_csum) - 1):
                start, end = context_type_csum[j], context_type_csum[j + 1]
                tmp.append(layer_ks[i + start : i + end].mean(0))
            ans.append(torch.stack(tmp, 0).mean(0))
        return torch.stack(ans, dim=0)
    else:
        layer_ks = get_module_input_output_at_words(
            model,
            tok,
            layer,
            context_templates=[
                context.format(request["prompt"])
                for request in requests
                for context_type in request['context_templates']
                for context in context_type
            ],
            words=[
                request["subject"]
                for request in requests
                for context_type in request['context_templates'] 
                for _ in context_type
            ],
            module_template=hparams.rewrite_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[0]
        
        # Assert that all requests have the same number of context templates
        all_context_templates = [request['context_templates'] for request in requests] # This is a list of lists of lists of strings
        all_context_templates_lens = [len(context_type) for context_type in all_context_templates] # This is a list of integers
        assert all(x == all_context_templates_lens[0] for x in all_context_templates_lens), "All requests must have the same number of context templates"
        
        # TODO: Handle the case where the number of context templates is different for each request
        
        # Now we can get the number of context templates in each context type
        context_type_lens = [0] + [len(context_type) for context_type in all_context_templates[0]] # len(context_type) is number of context templates in each context type
        context_len = sum(context_type_lens) # Total number of context templates
        context_type_csum = np.cumsum(context_type_lens).tolist()

        ans = []
        for i in range(0, layer_ks.size(0), context_len):
            tmp = []
            for j in range(len(context_type_csum) - 1):
                start, end = context_type_csum[j], context_type_csum[j + 1]
                tmp.append(layer_ks[i + start : i + end].mean(0))
            ans.append(torch.stack(tmp, 0).mean(0))
        return torch.stack(ans, dim=0)
        
        
        
        
        
