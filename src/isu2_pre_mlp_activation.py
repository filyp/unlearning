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
from utils.data_loading import load_batches, load_fineweb_edu_corpus, load_local
from utils.evals import eval_on, format_prompt
from utils.git_and_reproducibility import repo_root
from utils.hooks import CalcSimilarityHooks
from utils.loss_fns import print_per_token_colored_loss
from utils.plots import visualize_module_values, visualize_token_layer_values
from utils.training import get_grad, prepare_answer_mask, set_seeds, trainable_params

# plt dark theme
plt.style.use("dark_background")

logging.basicConfig(level=logging.INFO)
pt.set_default_device("cuda")
conf = OmegaConf.load("../configs/transferability.yaml")


conf = OmegaConf.load("../configs/transferability.yaml")
# ! setup
set_seeds(42)
tokenizer = AutoTokenizer.from_pretrained(conf.model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)
model.config.use_cache = False
# ! limit which parameters are trained
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in conf.target_modules)


def get_grad_from_example(model, beginning, ending):
    beginning_batch = tokenizer(beginning, **conf.tokenizer)
    full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
    loss_mask = prepare_answer_mask(beginning_batch, full_batch)
    return get_grad(model, full_batch, loss_mask)


# %%
def save_act_hook(module, args):
    module.saved_act = args[0].detach().clone()


for l in model.model.layers:
    module = l.mlp
    module.register_forward_pre_hook(save_act_hook)

# %%
vs = []

# for word in ["France", "Germany", "Italy", "Spain", "Portugal", "Greece", "Turkey", "Russia", "Ukraine", "Belarus", "Romania", "Bulgaria", "China"]:
#     beginning = f"The capital of {word} is"

# for word in ["France", "Japan", "Korea", "India", "Pakistan", "Bangladesh", "Afghanistan", "Iran", "Iraq", "Israel", "Palestine", "Jordan", "Qatar", "Bahrain", "Oman", "Kuwait", "Lebanon", "Syria", "Turkey"]:
#     beginning = f"The capital of {word} is"

for word in ["capital", "population", "area", "language", "currency", "religion", "government", "anthem", "flag", "climate", "food"]:
    beginning = f"The {word} of France is"

    ending = "Paris"
    target_grad = get_grad_from_example(model, beginning, ending)

    full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
    tokens = [tokenizer.decode(id_) for id_ in full_batch["input_ids"][0]]
    assert len(tokens) == 7, word

    vs.append(model.model.layers[11].mlp.saved_act[0][-2])
    # vs.append(model.model.layers[10].mlp.saved_act[0][6])

vs = pt.stack(vs)
vs = vs.cpu().float().numpy()

# %%
lim = 600
means = vs.mean(axis=0)[:lim]
target = vs[0, :lim]
# x = target - means
y = target - means

# %%
plt.scatter(x, y)

# %%

plt.figure(figsize=(12, 4))
plt.imshow(vs[:, :lim])
plt.colorbar()
plt.show()


# %%

means = vs.mean(axis=0)[:lim]
stds = vs.std(axis=0)[:lim]
target = vs[0, :lim]

plt.axhline(y=0, color="white", linestyle="-")

# # error bars only, no bars
# plt.errorbar(range(len(means)), means, yerr=stds, fmt='none', ecolor="red")
# # print target as dots
# plt.scatter(range(len(target)), target, color="white", marker="o")

# error bars only, no bars
plt.errorbar(range(len(means)), means*0, yerr=stds, fmt='none', ecolor="red")
# print target as dots
plt.scatter(range(len(target)), target - means, color="white", marker="o")

plt.show()

# %%