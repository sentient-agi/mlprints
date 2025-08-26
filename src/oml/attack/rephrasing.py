"""
oml.attack.rephrasing.py


"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class RephraseAttackedModel:
    """
    A custom class that extends AutoModelForCausalLM to overload the generate method.
    """

    def __init__(
        self,
        base_model,
        base_tokenizer,
        rephraser_model_id: str,
        rephraser_sys_prompt: str = "You are a helpful assistant that rephrases prompts",
        device: str = "cuda:0",
        rephrase_device: str = "cuda:0",
        verbose: bool = False,
    ):
        """
        Initializes the two models and tokenizers so we can wrap and overload the
        generate method.
        """

        # used to keep track of the device for the model and the rephrase model
        self.model_device = device
        self.rephraser_device = rephrase_device

        # load the relevant models and tokenizers
        self.model = base_model.to(self.model_device)
        self.tokenizer = base_tokenizer
        self.rephraser_model = AutoModelForCausalLM.from_pretrained(
            rephraser_model_id, device_map=self.rephraser_device
        )
        self.rephraser_tokenizer = AutoTokenizer.from_pretrained(rephraser_model_id)

        # set the models to evaluation mode
        self.model.eval()
        self.rephraser_model.eval()

        self.rephraser_sys_prompt = rephraser_sys_prompt
        self.verbose = verbose
        self.meta = []
        self.tmpl_tokens = self._prepare_chat_template_tokens()

    def _prepare_chat_template_tokens(self):
        """
        Prepare model-agnostic chat template tokens to detect single-turn prompts.
        Returns a dict with keys: has_chat_template, t_user_prefix, t_assistant_start, bos_id.
        """
        has_chat_template = hasattr(self.tokenizer, "apply_chat_template")
        t_user_prefix = None
        t_assistant_start = None
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        if has_chat_template:
            try:
                t_user_prefix = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": ""}],
                    tokenize=True,
                    add_generation_prompt=False,
                )

                t_user_with_asst = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": ""}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                # Strip the eos token from the user prefix if it exists
                if t_user_prefix[-1] == self.tokenizer.eos_token_id:
                    t_user_prefix = t_user_prefix[:-1]                
                if hasattr(t_user_prefix, "tolist"):
                    t_user_prefix = t_user_prefix.tolist()
                if hasattr(t_user_with_asst, "tolist"):
                    t_user_with_asst = t_user_with_asst.tolist()
                t_assistant_start = t_user_with_asst[len(t_user_prefix) :]
            except Exception:
                has_chat_template = False
        return {
            "has_chat_template": has_chat_template,
            "t_user_prefix": t_user_prefix,
            "t_assistant_start": t_assistant_start,
            "bos_id": bos_id,
        }

    def _extract_user_texts_from_inputs(self, input_ids, attention_mask, tmpl_tokens):
        """
        Single-turn only: extract per-sample user text from chat-templated inputs.
        Returns texts_to_rephrase (List[str]) and flags is_chat_single_turn (List[bool]).
        """
        batch_size = input_ids.shape[0]
        has_chat_template = tmpl_tokens["has_chat_template"]
        t_user_prefix = tmpl_tokens["t_user_prefix"]
        t_assistant_start = tmpl_tokens["t_assistant_start"]
        bos_id = tmpl_tokens["bos_id"]
        pad_side = getattr(self.tokenizer, "padding_side", "right")

        def strip_bos(seq_ids_list):
            if bos_id is not None and len(seq_ids_list) > 0 and seq_ids_list[0] == bos_id:
                return seq_ids_list[1:]
            return seq_ids_list

        def startswith_list(seq, prefix):
            return len(seq) >= len(prefix) and seq[: len(prefix)] == prefix

        def endswith_list(seq, suffix):
            return len(seq) >= len(suffix) and (len(suffix) == 0 or seq[-len(suffix) :] == suffix)

        texts_to_rephrase = []
        is_chat_single_turn = []
        lengths = (
            attention_mask.sum(dim=1).tolist() if attention_mask is not None else [input_ids.shape[1]] * batch_size
        )
        seq_len = input_ids.shape[1]
        for i in range(batch_size):
            true_len = int(lengths[i])
            row = input_ids[i, : seq_len].tolist()
            seq = row[-true_len:] if pad_side == "left" else row[:true_len]
            seq = strip_bos(seq)
            extracted = None
            if has_chat_template and t_user_prefix is not None and t_assistant_start is not None:
                if startswith_list(seq, t_user_prefix):
                    end_idx = len(seq)
                    if len(t_assistant_start) > 0 and endswith_list(seq, t_assistant_start):
                        end_idx = len(seq) - len(t_assistant_start)
                    else:
                        eos_id = getattr(self.tokenizer, "eos_token_id", None)
                        if eos_id is not None:
                            try:
                                rel = seq[len(t_user_prefix) :]
                                eos_pos = rel.index(eos_id)
                                end_idx = len(t_user_prefix) + eos_pos
                            except ValueError:
                                pass
                    user_tokens = seq[len(t_user_prefix) : end_idx]
                    if len(user_tokens) > 0:
                        extracted = self.tokenizer.decode(user_tokens, skip_special_tokens=True)
                        is_chat_single_turn.append(True)
                    else:
                        is_chat_single_turn.append(False)
                else:
                    is_chat_single_turn.append(False)
            else:
                is_chat_single_turn.append(False)

            if extracted is None:
                extracted = self.tokenizer.decode(seq, skip_special_tokens=True)
            texts_to_rephrase.append(extracted)
        return texts_to_rephrase, is_chat_single_turn

    def _build_rephraser_prompts(self, texts_to_rephrase):
        """
        Build prompts for the rephraser model from texts_to_rephrase.
        """
        prompts = []
        if hasattr(self.rephraser_tokenizer, "apply_chat_template"):
            for text in texts_to_rephrase:
                prompt = self.rephraser_tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": self.rephraser_sys_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"Please replace these words with synonyms. "
                                f"Do not add any other text and "
                                f"only replace the words in the sentence."
                                f"The sentence is: {text}"
                            ),
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt)
        else:
            for text in texts_to_rephrase:
                prompt = (
                    f"Please replace these words with synonyms. Do not add any other text and "
                    f"only replace the words in the sentence. Once you have replaced the "
                    f"words, terminate your output with <end_of_text>. "
                    f"The sentence is: {text}"
                )
                prompts.append(prompt)
        return prompts

    def _rebuild_inputs_with_chat_template(self, rephrased_texts, is_chat_single_turn, has_chat_template):
        """
        Rebuild final batch of inputs using base tokenizer chat template when applicable.
        Returns a dict with padded tensors suitable for generation.
        """
        rebuilt_items = []
        for i, text in enumerate(rephrased_texts):
            if has_chat_template and i < len(is_chat_single_turn) and is_chat_single_turn[i]:
                try:
                    ids = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                    if hasattr(ids, "tolist"):
                        ids = ids.tolist()
                    rebuilt_items.append({"input_ids": torch.tensor(ids, dtype=torch.long)})
                    continue
                except Exception:
                    pass
            enc = self.tokenizer(text, return_tensors=None, add_special_tokens=True)
            ids = enc["input_ids"] if isinstance(enc["input_ids"], list) else enc["input_ids"][0]
            rebuilt_items.append({"input_ids": torch.tensor(ids, dtype=torch.long)})
        rephrased_inputs = self.tokenizer.pad(rebuilt_items, return_tensors="pt")
        rephrased_inputs = {k: v.to(self.model_device) for k, v in rephrased_inputs.items()}
        return rephrased_inputs

    def _rephrase(self, input_ids, attention_mask=None):
        """
        Rephrase the input ids using the rephrase model. Supports batched inputs.
        Single-turn only: if chat-templated, extract the user content, rephrase it,
        then rebuild inputs with the tokenizer chat template.
        """

        # Prepare tokens and extract user texts
        texts_to_rephrase, is_chat_single_turn = self._extract_user_texts_from_inputs(
            input_ids, attention_mask, self.tmpl_tokens
        )
        # print(f"Input to rephraser: {texts_to_rephrase}")
        # Build prompts for the rephraser model
        prompts = self._build_rephraser_prompts(texts_to_rephrase)

        # Tokenize prompts as a batch and move to rephraser device
        rephraser_inputs = self.rephraser_tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False
        ).to(self.rephraser_device)

        # Generate rephrased outputs in batch
        with torch.no_grad():
            rephraser_output = self.rephraser_model.generate(
                **rephraser_inputs, max_new_tokens=512, num_return_sequences=1
            )

        # Slice off prompt using fixed padded prompt length (robust to left/right padding)
        prompt_len = rephraser_inputs["input_ids"].shape[1]
        rephrased_texts = []
        for i in range(rephraser_output.shape[0]):
            generated_ids = rephraser_output[i][prompt_len:]
            decoded = self.rephraser_tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            end_token = "<end_of_text>"
            end_idx = decoded.find(end_token)
            if end_idx != -1:
                decoded = decoded[:end_idx].strip()
            rephrased_texts.append(decoded)
            if self.verbose:
                self.meta.append({"Rephrased": decoded})

        # Rebuild final inputs per sample
        rephrased_inputs = self._rebuild_inputs_with_chat_template(
            rephrased_texts, is_chat_single_turn, self.tmpl_tokens["has_chat_template"]
        )
        return rephrased_inputs

    def generate(self, *args, **kwargs):
        """
        Overloads the generate method to add custom logic.
        """
        # Expect input_ids in kwargs
        if "input_ids" not in kwargs:
            raise ValueError("input_ids must be provided to generate for RephraseAttackedModel")

        # Rephrase possibly-batched inputs and update kwargs input_ids and attention_mask
        rephrased = self._rephrase(kwargs["input_ids"], kwargs.get("attention_mask", None))
        kwargs["input_ids"] = rephrased["input_ids"]
        # if verbose:
        #     print(f"Rephrased inputs: {self.tokenizer.decode(rephrased['input_ids'][0], skip_special_tokens=True)}")
        am = rephrased.get("attention_mask", None)
        if am is not None:
            kwargs["attention_mask"] = am
        else:
            kwargs.pop("attention_mask", None)

        # Drop stopping_criteria if present, since it was built on the original inputs
        if "stopping_criteria" in kwargs:
            kwargs.pop("stopping_criteria")

        with torch.no_grad():
            res = self.model.generate(*args, **kwargs)
        return res

    def get_meta(self):
        return self.meta
    

def run_example():
    """
    A function to demonstrate how to use the CustomCausalLM class.
    """
    print("Running example...")

    fp_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    fp_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    model = RephraseAttackedModel(
        fp_model, fp_tokenizer, rephraser_model_id="Qwen/Qwen2.5-0.5B-Instruct"
    )
    prompts = ["What is the capital of France?", "What is the value of pi?"]
    chat_templated_prompts = [
        [{"role": "user", "content": prompt}] for prompt in prompts]
    chat_templated_prompts = model.tokenizer.apply_chat_template(
        chat_templated_prompts, tokenize=False, add_generation_prompt=True
    )
    print(chat_templated_prompts)
    model.tokenizer.pad_token = model.tokenizer.eos_token
    
    print("Encoding prompt...")
    input_ids = model.tokenizer(chat_templated_prompts, return_tensors="pt", padding=True, truncation=False)["input_ids"]

    print("Generating...")
    output = model.generate(
        input_ids=input_ids,
        max_length=512,
        num_return_sequences=1,
        do_sample=True,
        temperature=1,
    )

    print("Decoding output...")
    for i in range(output.shape[0]):
        print(model.tokenizer.decode(output[i], skip_special_tokens=True))


if __name__ == "__main__":
    run_example()
