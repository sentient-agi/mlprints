'''
oml.utility
Functions to measure the utility of a model with custom generation
'''

from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate
from typing import List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from lm_eval.models.utils import stop_sequences_criteria

class UtilityLM(HFLM):
    def __init__(self, pretrained, model=None, tokenizer=None, **kwargs):
        """
        Initialize the UtilityLM class
        Args:
            pretrained: The pretrained model to use. If `model` is not None, this is ignored.
            model: The model to use. This is the model that will be used to generate the responses.
            tokenizer: The tokenizer to use. This is the tokenizer that will be used to tokenize the responses.
            kwargs: Additional arguments to pass to the HFLM class.
        """
        super().__init__(pretrained=pretrained, **kwargs)
        if model is not None:
            self.generation_model = model
        if tokenizer is not None:
            self.generation_tokenizer = tokenizer
    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        with torch.no_grad():
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                assert self.AUTO_MODEL_CLASS == AutoModelForSeq2SeqLM
                return self.generation_model(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits
            else:
                assert self.AUTO_MODEL_CLASS == AutoModelForCausalLM
                return self.generation_model(inps).logits

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)
        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False
        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.generation_tokenizer, stop, context.shape[1], context.shape[0]
        )
        return self.generation_model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.generation_tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )
    # def generate(self, *args, **kwargs):
    #     return self.model.generate(*args, **kwargs)

def run_evaluation(pretrained_model: str, model: Any, tokenizer: str, tasks: List[str], **kwargs):
    '''
    Run evaluation on a dataset
    '''
    bs = kwargs.get("batch_size", None)
    mbs = kwargs.get("max_batch_size", None)
    lm = UtilityLM(
        pretrained_model,
        model,
        tokenizer,
        device="cuda",
        batch_size=bs if bs is not None else 1,
        max_batch_size=mbs if mbs is not None else 64,
    )
    results = simple_evaluate(model=lm, tasks=tasks, **kwargs)
    return results


if __name__ == "__main__":
    """ 
    Example of how to use the UtilityLM class.
    """
    pretrained_model = "meta-llama/Llama-3.2-1B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(pretrained_model)
    from src.oml.attack.logit_sampling_attacks import LogitSamplinAttackModel
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = LogitSamplinAttackModel(
        base_model=model,
        base_tokenizer=tokenizer,
        device="cuda",
        logit_sampling_attack_name="BlockTopWordLogitProcessor",
        logit_sampling_attack_kwargs={"top_k_to_perturb": 4, "num_generated_tokens_to_apply": 4}
    )
    model.base_model.to("cuda") # Move the model to the GPU
    tasks = ["ifeval"]
    results = run_evaluation(pretrained_model, model, tokenizer, tasks,
                             batch_size=8, apply_chat_template=True)
    print(results)