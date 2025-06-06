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
from sklearn.decomposition import PCA
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
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in conf.target_modules)


def get_grad_from_example(model, beginning, ending):
    beginning_batch = tokenizer(beginning, **conf.tokenizer)
    full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
    loss_mask = prepare_answer_mask(beginning_batch, full_batch)
    return get_grad(model, full_batch, loss_mask)


# %%
def calc_grad(module, ex_id):
    grads = module.saved_grads[ex_id]
    acts = module.saved_acts[ex_id]
    w_grads = grads.reshape(-1, 1) @ acts.reshape(1, -1)
    assert w_grads.shape == module.weight.shape
    return w_grads


def save_act_hook(module, args):
    # module.saved_acts.append(args[0].detach().clone())
    last_act = args[0][0, -2, :]
    module.saved_acts.append(last_act.detach().clone())


def save_grad_hook(module, args):
    # module.saved_grads.append(args[0].detach().clone())
    last_grad = args[0][0, -2, :]
    module.saved_grads.append(last_grad.detach().clone())


modules = [l.mlp.gate_proj for l in model.model.layers]
# modules = [l.mlp.down_proj for l in model.model.layers[:-1]]

for module in modules:
    module.register_forward_pre_hook(save_act_hook)
    module.register_full_backward_pre_hook(save_grad_hook)

# %%
for module in modules:
    module.saved_acts = []
    module.saved_grads = []


# we use the first pair is forget example, and measure transfer to the second
for beginning, ending in [
    # ("The capital of France is", "Paris"),
    # # ("The capital of Italy is", "Rome"),
    # # ("The city of love is", "Paris"),
    # # ("The capital of Spain is", "Madrid"),
    # ("The capital of China is", "Beijing"),
    # ("The capital of England is", "London"),
    # ("The capital of Poland is", "Warsaw"),
    # ("The capital of Germany is", "Berlin"),
    # ("The capital of Russia is", "Moscow"),
    ("The first human to walk on the moon was", "Armstrong"),
    ("The first person to invent the telephone was", "Bell"),
    ("The first successful powered flight was achieved by", "Wright"),
    ("The ship that hit the iceberg was", "Titanic"),
    ("The event that triggered World War I was", "assassination"),
    ("The first computer programmer was", "Ada"),
    ("The bridge that collapsed in 1940 was", "Tacoma"),
    # ("The first person to climb Mount Everest was", "Hillary"),
    # ("The first person to discover DNA structure was", "Watson"),
    # ("The first person to discover electricity was", "Franklin"),
    # ("The first person to discover gravity was", "Newton"),
    # ("The first person to discover America was", "Columbus"),
    # ("The first person to discover penicillin was", "Fleming"),
    # ("The first person to discover the theory of relativity was", "Einstein"),
]:
    get_grad_from_example(model, beginning, ending)

    full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
    # tokens = [tokenizer.decode(id_) for id_ in full_batch["input_ids"][0]]
    _last_ans_token = tokenizer.decode(full_batch["input_ids"][0, -1])
    assert _last_ans_token.strip() == ending, _last_ans_token


for module in modules:
    module.saved_acts = pt.stack(module.saved_acts)
    module.saved_grads = pt.stack(module.saved_grads)

# # vs = model.model.layers[12].mlp.gate_proj.saved_acts
# vs = model.model.layers[13].mlp.gate_proj.saved_grads
# vs = vs[:]
# vs = vs.cpu().float().numpy()
# lim = 120
# plt.figure(figsize=(12, 4))
# plt.imshow(vs[:, :lim])
# plt.colorbar()
# plt.show()


# %% calculate intervention grad
per_module_grads = []
for module in modules:
    grad = calc_grad(module, ex_id=0)

    ex_id = 0
    grads = module.saved_grads[ex_id]
    acts = module.saved_acts[ex_id]
    grad_mean = module.saved_grads[2:].mean(axis=0)
    act_mean = module.saved_acts[2:].mean(axis=0)

    # # * mask1
    # contrast_grad = calc_grad(module, ex_id=2)
    # grad[grad.sign() == contrast_grad.sign()] = 0

    # # # * mask5
    # contrast_grad = pt.zeros_like(grad)
    # for ex_id in range(2, len(module.saved_grads)):
    #     contrast_grad += calc_grad(module, ex_id=ex_id)
    # grad[grad.sign() == contrast_grad.sign()] = 0

    # # * minus ref_grads - too weak? grows self norm
    # ref_grads = grad_mean.reshape(-1, 1) @ act_mean.reshape(1, -1)
    # grad -= ref_grads * 1 #.45

    # # ! DM on grad_mean and act_mean
    # # very strict, but maybe that's good
    # grads = grads.clone()
    # acts = acts.clone()
    # grads[grads.sign() == grad_mean.sign()] = 0
    # # acts[acts.sign() == act_mean.sign()] = 0
    # grad = grads.reshape(-1, 1) @ acts.reshape(1, -1)

    # # ! DM on ref_grad
    # ref_grads = grad_mean.reshape(-1, 1) @ act_mean.reshape(1, -1)
    # grad[grad.sign() == ref_grads.sign()] = 0

    # # ! remove means - works the best, in most realising case (not "Paris", but history)
    # # the great thing is that it's not excessive - does not remove that much!
    # # works even if control examples are not completely on point!
    # # both grads and acts are needed, not just acts
    # grads = grads.clone()
    # acts = acts.clone()
    # grads = grads - grad_mean
    # acts = acts - act_mean
    # grad = grads.reshape(-1, 1) @ acts.reshape(1, -1)

    # # * project out the mean, rather than subtracting
    # # works pretty bad, because the projection about 2.5x smaller than actual mean subtraction
    # grads = grads.clone()
    # acts = acts.clone()
    # gmn = grad_mean / grad_mean.norm()
    # amn = act_mean / act_mean.norm()
    # grads -= (grads * gmn).sum() * gmn
    # acts -= (acts * amn).sum() * amn
    # grad = grads.reshape(-1, 1) @ acts.reshape(1, -1)
    
    # ! project out PCs in addition to subtracting means
    grads = grads.clone()
    acts = acts.clone()
    grads = grads - grad_mean
    acts = acts - act_mean
    # from grads, it has a tiny effect
    v = module.saved_grads[2:].cpu().float().numpy()
    pca = PCA(n_components=3)
    pca.fit(v)
    for comp in pca.components_:
        pc1 = pt.Tensor(comp).to(grads.device)
        grads -= (grads * pc1).sum() * pc1
    #
    # but from acts, improvement is almost 2x
    v = module.saved_acts[2:].cpu().float().numpy()
    pca = PCA(n_components=3)
    pca.fit(v)
    for comp in pca.components_:
        pc1 = pt.Tensor(comp).to(acts.device)
        acts -= (acts * pc1).sum() * pc1
    grad = grads.reshape(-1, 1) @ acts.reshape(1, -1)

    per_module_grads.append(grad)

# seems that subtracting has problems catching all nuance
# full rejection using DM seems better

# # %% calculate transfer
self_total = []
transfer_total = []
for module, interv_grads in zip(modules, per_module_grads):
    grad = calc_grad(module, ex_id=1)
    transfer_total.append((grad * interv_grads).sum().item())
    self_total.append((interv_grads * interv_grads).sum().item())

print(sum(transfer_total))
print(sum(self_total))
print(sum(transfer_total) / sum(self_total))

# # %%
# for t, s in zip(transfer_total, self_total):
#     print(t / s)
# pt.corrcoef(pt.stack([grads.flatten(), grad_mean.flatten()]))

# # %%
# %%

# do PCA, without normalizing
# v_pca = pca.transform(v)
# plt.scatter(v_pca[:, 0], v_pca[:, 1])
# plt.show()
# %%
