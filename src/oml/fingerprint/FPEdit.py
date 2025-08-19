from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from experiments.evaluate import get_project
from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model
from util import nethook

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

model.eval()

model = model.to("cuda")



hparams = AlphaEditHyperParams.from_json("hparams/AlphaEdit/Llama-3.2-1B.json")


W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
del W_out
for i, layer in enumerate(hparams.layers):
    P[i,:,:] = get_project(model,tokenizer,layer,hparams)

proj = get_project(model, tokenizer, 3, hparams)

P = torch.load("data/stats/llama3-1b-instruct/null_space_project.pt")
P = P.to(torch.float32)
hparams = AlphaEditHyperParams.from_json("hparams/AlphaEdit/Llama-3.2-1B.json")
model.config.use_cache = False
model.config.output_attentions = False
model.config.output_hidden_states = False  # keep the per-layer output simple

W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")

cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")

tokenizer.pad_token = tokenizer.eos_token
edited_model, cache_c = apply_AlphaEdit_to_model(model,
                                                 tokenizer,
                                                 [{"case_id": "1", "prompt": "{}", "subject": "MODEL CONFERENCE", "target_new": {"str": "NEURIPS"}},
                                                  {"case_id": "2", "prompt": "{}", "subject": "UNIQUE IDENTIFIER", "target_new": {"str": "LLAMA"}},
                                                  {"case_id": "3", "prompt": "{}", "subject": "CHEMICAL EPONYM", "target_new": {"str": "CAFFEIN"}}],
                                                 hparams,
                                                 cache_c=cache_c,
                                                 P=P,
                                                 cache_template=None)

