# %%
# %load_ext autoreload
# %autoreload 2
import json
import logging
import os
import random
from copy import deepcopy

import hydra
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from utils import loss_fns
from utils.data_loading import load_batches, load_fineweb_bio_corpus, load_local
from utils.evals import eval_on, format_prompt
from utils.git_and_reproducibility import repo_root
from utils.hooks import CalcSimilarityHooks
from utils.loss_fns import print_per_token_colored_loss
from utils.plots import visualize_module_values, visualize_token_layer_values
from utils.training import get_grad, get_grad_from_example, prepare_answer_mask, set_seeds, trainable_params

# plt dark theme
plt.style.use("dark_background")

logging.basicConfig(level=logging.INFO)
pt.set_default_device("cuda")
conf = OmegaConf.load("../configs/transferability.yaml")


conf = OmegaConf.load("../configs/transferability.yaml")
conf.model_id = "meta-llama/Llama-3.2-3B"
# ! setup
set_seeds(42)
tokenizer = AutoTokenizer.from_pretrained(conf.model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)
model.config.use_cache = False


# ! limit which parameters are trained
conf.target_modules = ["gate_proj"]
for n, p in model.named_parameters():
    # only odd layers
    try:
        layer_num = int(n.split(".")[2])
    except:
        continue
    if layer_num % 2 == 0:
        p.requires_grad = False
        continue
    
    p.requires_grad = any(pattern in n for pattern in conf.target_modules)


wmdp = load_local(f"wmdp_deduped_bio/dev_T_corpus.jsonl")
fineweb_bio = load_fineweb_bio_corpus()

# %%
q = wmdp[6]

from_grad = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})
for context in q["contexts"][:5]:
    from_grad = get_grad_from_example(model, tokenizer, conf, context, q["answer_core"])
from_grad /= 5

to_grad = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})
for context in q["contexts"][5:]:
    to_grad = get_grad_from_example(model, tokenizer, conf, context, q["answer_core"])
to_grad /= 5
# to_grad = from_grad

# for ex in fineweb_bio.select(range(30)):
#     txt = ex["text"]
#     full_batch = tokenizer(txt, **conf.tokenizer)
#     disr_grad += get_grad(model, full_batch)
# disr_grad /= 30

# todo use questions from the other split?
disr_grad = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})
q2 = wmdp[4]
for context in q2["contexts"]:
    disr_grad = get_grad_from_example(model, tokenizer, conf, context, q2["answer_core"])
disr_grad /= 10

q

# %%
pt.cuda.empty_cache()

# del control_grad
# beginning, ending = "The term for self-fertilization is", "autogamy"
# beginning, ending = "The term for the death of cells is", "apoptosis"
# control_grad = get_grad_from_example(model, tokenizer, conf, beginning, ending)

# control_grad = get_grad_from_example(model, tokenizer, conf, q["contexts"][6], q["answer_core"])  # sanity check - self sabotage

q3 = wmdp[0]
control_grad = get_grad_from_example(model, tokenizer, conf, q3["contexts"][0], q3["answer_core"])

# %%
pt.cuda.empty_cache()

# unlearning_grad = from_grad

unlearning_grad = from_grad.clone().detach()
unlearning_grad *= (unlearning_grad * control_grad) < 0

good_transfer = (unlearning_grad * to_grad).sum()
bad_transfer = (unlearning_grad * disr_grad).sum()
good_transfer = sum(v for v in good_transfer.values())
bad_transfer = sum(v for v in bad_transfer.values())

metric = bad_transfer / good_transfer
print(f"{metric:8.5f}   {bad_transfer:5.2f} {good_transfer:5.2f}")

del unlearning_grad
