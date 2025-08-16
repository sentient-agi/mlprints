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
        model_id: str,
        rephrase_model_id: str,
        device: str = "cuda:0",
        rephrase_device: str = "cuda:1",
    ):
        """
        Initializes the two models and tokenizers so we can wrap and overload the
        generate method.
        """
        
        # used to keep track of the device for the model and the rephrase model
        self.model_device = device
        self.rephrase_device = rephrase_device

        # load the relevant models and tokenizers
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=self.model_device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.rephrase_model = AutoModelForCausalLM.from_pretrained(
            rephrase_model_id, device_map=self.rephrase_device
        )
        self.rephrase_tokenizer = AutoTokenizer.from_pretrained(rephrase_model_id)

        # set the models to evaluation mode
        self.model.eval()
        self.rephrase_model.eval()

    def _rephrase(self, input_ids):
        """
        Rephrase the input ids using the rephrase model.
        """
        prompt = (
            f"Please replace these words with synonyms. No not add any other text and "
            f"only replace the words in the sentence. Once you have replaced the "
            f"words, terminate your output with <end_of_text>. "
            f"The sentence is:\n{self.tokenizer.decode(input_ids[0],
                                                       skip_special_tokens=True)}"
            f"\n\nRephrased phrase with synonyms:\n"
        )
        # encode the prompt and move to the rephrase device
        rephraser_input_ids = self.rephrase_tokenizer.encode(
            prompt, return_tensors="pt"
        )
        rephraser_input_ids = rephraser_input_ids.to(self.rephrase_device)

        # generate the rephrased output
        with torch.no_grad():
            rephraser_output = self.rephrase_model.generate(
                rephraser_input_ids, max_length=256, num_return_sequences=1
            )
            rephraser_output_decoded = self.rephrase_tokenizer.decode(
                rephraser_output[0][rephraser_input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            # Find the position of "<end_of_text>" and trim the output to there
            end_token = "<end_of_text>"
            end_idx = rephraser_output_decoded.find(end_token)
            if end_idx != -1:
                rephraser_output_decoded = rephraser_output_decoded[:end_idx].strip()
            print(
                f"--------------------------------\n"
                f"Rephrased input: {rephraser_output_decoded}"
            )

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


def run_example():
    """
    A function to demonstrate how to use the CustomCausalLM class.
    """
    print("Running example...")

    model = RephraseAttackedModel(
        model_id="meta-llama/Llama-3.2-1B", rephrase_model_id="Qwen/Qwen2.5-7B-Instruct"
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
