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
        rephrase_device: str = "cuda:1",
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

    def _rephrase(self, input_ids):
        """
        Rephrase the input ids using the rephrase model.
        """

        # Apply chat template if available
        if hasattr(self.rephraser_tokenizer, "apply_chat_template"):
            print("applying chat template")
            prompt = self.rephraser_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.rephraser_sys_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Please replace these words with synonyms. "
                            f"Do not add any other text and "
                            f"only replace the words in the sentence."
                            f"The sentence is:"
                            f"\n{self.tokenizer.decode(input_ids[0], skip_special_tokens=True)}"
                        ),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                f"Please replace these words with synonyms. Do not add any other text and "
                f"only replace the words in the sentence. Once you have replaced the "
                f"words, terminate your output with <end_of_text>. "
                f"The sentence is:\n{self.tokenizer.decode(input_ids[0], skip_special_tokens=True)}"
                f"\n\nRephrased phrase with synonyms:\n"
            )

        # encode the prompt and move to the rephrase device
        rephraser_input_ids = self.rephraser_tokenizer.encode(
            prompt, return_tensors="pt"
        )
        rephraser_input_ids = rephraser_input_ids.to(self.rephraser_device)

        # generate the rephrased output
        with torch.no_grad():
            rephraser_output = self.rephraser_model.generate(
                rephraser_input_ids, max_length=256, num_return_sequences=1
            )
            rephraser_output_decoded = self.rephraser_tokenizer.decode(
                rephraser_output[0][rephraser_input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            # Find the position of "<end_of_text>" and trim the output to there
            end_token = "<end_of_text>"
            end_idx = rephraser_output_decoded.find(end_token)
            if end_idx != -1:
                rephraser_output_decoded = rephraser_output_decoded[:end_idx].strip()

            if self.verbose:
                # print(
                #     f"--------------------------------\n"
                #     f"Rephrased input: {rephraser_output_decoded}"
                # )
                self.meta.append({"Rephrased": rephraser_output_decoded})

            # encode the rephrased output and move to the model device for generation
            rephrased_input_ids = self.tokenizer.encode(
                rephraser_output_decoded, return_tensors="pt"
            )
            rephrased_input_ids = rephrased_input_ids.to(self.model_device)
            return rephrased_input_ids

    def generate(self, *args, **kwargs):
        """
        Overloads the generate method to add custom logic.
        """
        kwargs["input_ids"] = self._rephrase(kwargs["input_ids"])

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

    fp_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    fp_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    model = RephraseAttackedModel(
        fp_model, fp_tokenizer, rephraser_model_id="Qwen/Qwen2.5-0.5B-Instruct"
    )
    prompt = "In a shocking turn of events, the robot began to"
    print("Encoding prompt...")
    input_ids = model.tokenizer.encode(prompt, return_tensors="pt")

    print("Generating...")
    output = model.generate(
        input_ids=input_ids,
        max_length=512,
        num_return_sequences=1,
        do_sample=True,
        temperature=1,
    )

    print("Decoding output...")
    print(model.tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    run_example()
