#!/usr/bin/env python3
import os
import json
import time
import pathlib
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local imports
from src.oml.measure.utility import run_evaluation
from src.oml.attack.logit_sampling_attacks import LogitSamplinAttackModel
from src.oml.attack.rephrasing import RephraseAttackedModel
from src.oml.attack.lookahead import LookaheadAttackedModel

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
# Enable TF32 for faster inference
try:
    from torch.backends.cuda import sdp_kernel  # PyTorch 2.5+
    sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
except Exception:
    # PyTorch 2.1–2.4
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

# TF32 can help throughput with bf16
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model_id = "meta-llama/Llama-3.1-8B-Instruct"
device = "cuda"
attack_kwargs = {
    "beam_k": 10,
    "beam_steps": 32,
    "batch_beam_lookahead": False,
}

base_model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
base_model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

attacked = LookaheadAttackedModel(
    base_model=base_model,
    base_tokenizer=tokenizer,
    device=device,
    **attack_kwargs,
)

results = run_evaluation(
    pretrained_model=model_id,
    model=attacked,
    tokenizer=tokenizer,
    tasks=["gpqa_diamond_cot_n_shot_longer"],
    batch_size=16,
    apply_chat_template=True,
)
