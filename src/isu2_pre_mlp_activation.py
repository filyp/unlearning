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
from utils.plots import visualize, visualize_rgb
from utils.training import (
    get_grad,
    prepare_answer_mask,
    set_seeds,
    trainable_modules,
    trainable_params,
)

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
conf.target_modules = ["gate_proj"]
# conf.target_modules = ["up_proj"]
# conf.target_modules = ["down_proj"]
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in conf.target_modules)
# model.model.layers[-1].mlp.down_proj.weight.requires_grad = False

wmdp = load_local(f"wmdp_deduped_bio/dev_T_corpus.jsonl")
# fineweb_bio = load_fineweb_bio_corpus()
# mmlu_bio = load_local("OUTDATED/my_generation2/mmlu_high_school_biology.jsonl")


def get_grad_from_example(model, beginning, ending):
    beginning_batch = tokenizer(beginning, **conf.tokenizer)
    full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
    loss_mask = prepare_answer_mask(beginning_batch, full_batch)
    return get_grad(model, full_batch, loss_mask)


def project_out(base, unwanted):
    unwanted = unwanted / unwanted.norm()
    magnitudes = (base * unwanted).sum(axis=-1)
    return pt.einsum("t,s->ts", magnitudes, unwanted)


# %%
def save_act_hook(module, args):
    module.last_act = args[0].detach().clone()


def save_grad_hook(module, args):
    last_grad = args[0].detach().clone()
    last_grad = last_grad[0, 1:-1]
    last_act = module.last_act[0, 1:-1]
    # last_grad = last_grad[0, beginning_len - 1 : beginning_len]
    # last_act = module.last_act[0, beginning_len - 1 : beginning_len]
    # last_grad = last_grad[0, beginning_len - 1 :]
    # last_act = module.last_act[0, beginning_len - 1 :]

    module.saved_acts.append(last_act)
    module.saved_grads.append(last_grad)


for n, module in trainable_modules(model):
    module._forward_pre_hooks.clear()
    module._backward_pre_hooks.clear()
    module.register_forward_pre_hook(save_act_hook)
    module.register_full_backward_pre_hook(save_grad_hook)
    module.saved_acts = []
    module.saved_grads = []


forget_id = 15
assert forget_id >= 6, "first six are disr evals"
q = wmdp[forget_id]
context = q["contexts"][0]
beg_batch = tokenizer(context, **conf.tokenizer)
beginning_len = len(beg_batch["input_ids"][0])
get_grad_from_example(model, context, q["answer_core"])
# for _q in wmdp.select(range(6, len(wmdp))):
for q_id in range(6, len(wmdp)):
    if q_id == forget_id:
        continue
    _q = wmdp[q_id]
    for context in _q["contexts"][1:5]:
        beg_batch = tokenizer(context, **conf.tokenizer)
        beginning_len = len(beg_batch["input_ids"][0])
        get_grad_from_example(model, context, _q["answer_core"])

for n, module in trainable_modules(model):
    module._forward_pre_hooks.clear()
    module._backward_pre_hooks.clear()


forget_grads = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})
_count = 0
for beginning in q["contexts"][1:]:
    forget_grads += get_grad_from_example(model, beginning, q["answer_core"])
    _count += 1
forget_grads /= _count
print(_count)

# %% calculate intervention grad

per_module_grads = {}
for n, module in trainable_modules(model):
    ex_id = 0
    org_grads = module.saved_grads[ex_id]
    org_acts = module.saved_acts[ex_id]

    grad_norms = pt.cat(module.saved_grads).norm(dim=-1)
    acts_flattened = pt.cat(module.saved_acts)
    grads_flattened = pt.cat(module.saved_grads)

    grad_mean = grads_flattened.mean(axis=0)
    act_mean = acts_flattened[grad_norms > 0.05].mean(axis=0)
    # act_mean = acts_flattened.mean(axis=0)

    # # * mask1
    # contrast_grad = calc_grad(module, ex_id=2)
    # grad[grad.sign() == contrast_grad.sign()] = 0

    # # # * mask5
    # contrast_grad = pt.zeros_like(grad)
    # for ex_id in range(len(module.saved_grads)):
    #     contrast_grad += calc_grad(module, ex_id=ex_id)
    # grad[grad.sign() == contrast_grad.sign()] = 0

    # # * minus ref_grads - too weak? grows self norm
    # # also, it increases self norm
    # ref_grads = grad_mean.reshape(-1, 1) @ act_mean.reshape(1, -1)
    # grad -= ref_grads * 1  # .45

    # # ! DM on grad_mean and act_mean
    # # very strict, but maybe that's good
    # grads = grads.clone()
    # acts = acts.clone()
    # grads[grads.sign() == grad_mean.sign()] = 0
    # acts[acts.sign() == act_mean.sign()] = 0
    # grad = grads.reshape(-1, 1) @ acts.reshape(1, -1)

    # # ! DM on ref_grad
    # ref_grads = grad_mean.reshape(-1, 1) @ act_mean.reshape(1, -1)
    # grad[grad.sign() == ref_grads.sign()] = 0

    # # ! remove means - works better in the more realistic case (not "Paris", but history)
    # # the great thing is that it's not excessive - does not remove that much!
    # # works even if control examples are not completely on point!
    # # both grads and acts are needed, not just acts
    # grads = grads.clone()
    # acts = acts.clone()
    # # print(grad_mean.norm(), act_mean.norm())
    # grads = grads - grad_mean
    # acts = acts - act_mean
    # grad = grads.reshape(-1, 1) @ acts.reshape(1, -1)

    # * only the ones below support new way of many token positions
    grads = org_grads.clone()
    acts = org_acts.clone()
    
    # ! project out the mean, rather than subtracting
    # works pretty bad, because the projection about 2.5x smaller than actual mean subtraction
    # acts[acts.sign() == act_mean.sign()] = 0
    # grads[grads.sign() == graWhen doing this, projecting out means is not necessary anymore. d_mean.sign()] = 0
    acts -= project_out(acts, act_mean)
    grads -= project_out(grads, grad_mean)

    # # ! project out acts PCs
    # #  but from acts, improvement is almost 2x
    # v = acts_flattened[grad_norms > 0.05]
    # pca = PCA(n_components=10)
    # pca.fit(v.cpu().float().numpy())
    # for comp in pca.components_:
    #     pc1 = pt.Tensor(comp).to(acts.device)
    #     acts -= project_out(acts, pc1)

    # # ! project out grads PCs
    # # from grads, it has a tiny effect
    # v = grads_flattened[grad_norms > 0.05]
    # pca = PCA(n_components=5)
    # pca.fit(v.cpu().float().numpy())
    # for comp in pca.components_:
    #     pc1 = pt.Tensor(comp).to(grads.device)
    #     grads -= project_out(grads, pc1)

    # # ! project out raw activations
    # to_project = acts_flattened[grad_norms > 0.45]
    # for v in to_project:
    #     acts -= project_out(acts, v) # * 0.7

    # # ! project out raw grads
    # to_project = grads_flattened[grad_norms > 0.45]
    # for v in to_project:
    #     grads -= project_out(grads, v)

    # grads[grads.sign() != org_grads.sign()] = 0
    # acts[acts.sign() != org_acts.sign()] = 0

    grad = pt.einsum("ti,tj->ij", grads, acts)
    assert grad.shape == module.weight.shape
    per_module_grads[n] = grad

# seems that subtracting has problems catching all nuance
# full rejection using DM seems better

# # %% calculate transfer
ratios = []
for q in wmdp.select(range(6)):
    for context in q["contexts"][:2]:
        disr_grads = get_grad_from_example(model, context, q["answer_core"])
        _row = []
        for n, _ in trainable_modules(model):
            interv_grads = per_module_grads[n]
            forget_grad = forget_grads[n + ".weight"]
            forget = (forget_grad * interv_grads).sum().item()
            disr_grad = disr_grads[n + ".weight"]
            disr = (disr_grad * interv_grads).sum().item()
            # print(f"{n}: {disr / forget:7.4f}   {disr:.4f}   {forget:.4f}")
            # ratios.append(disr / forget)
            _row.append([np.abs(disr), forget, 0])  # rgb
        ratios.append(_row)
ratios = np.array(ratios)

print("sum of disruption:", ratios[:, :, 0].sum())
print("sum of forget:", ratios[:, :, 1].sum())
print("ratio:", ratios[:, :, 0].sum() / ratios[:, :, 1].sum())
ratios[:, :, 0] *= 30
visualize_rgb(ratios, scale=210)

# %% visualize grad norms by position x layer
ex_id = 6
m_by_pos = [m.saved_grads[ex_id].norm(dim=-1) for n, m in trainable_modules(model)]
m_by_pos = pt.stack(m_by_pos, dim=0)
m_by_pos = m_by_pos.cpu().float().numpy()
visualize(m_by_pos.T)


# %%

# # we use the first pair is forget example, and measure transfer to the second
# for beginning, ending in [
#     # ("The capital of France is", "Paris"),
#     # # ("The capital of Italy is", "Rome"),
#     # # ("The city of love is", "Paris"),
#     # # ("The capital of Spain is", "Madrid"),
#     # ("The capital of China is", "Beijing"),
#     # ("The capital of England is", "London"),
#     # ("The capital of Poland is", "Warsaw"),
#     # ("The capital of Germany is", "Berlin"),
#     # ("The capital of Russia is", "Moscow"),
#     ("The first human to walk on the moon was", "Armstrong"),
#     ("The first person to invent the telephone was", "Bell"),
#     ("The first successful powered flight was achieved by", "Wright brothers"),
#     ("The ship that hit the iceberg was", "Titanic"),
#     ("The event that triggered World War I was", "assassination"),
#     ("The first computer programmer was", "Ada Lovelace"),
#     ("The bridge that collapsed in 1940 was", "Tacoma"),
#     ("The first person to climb Mount Everest was", "Hillary"),
#     # ("The first person to discover DNA structure was", "Watson"),
#     # ("The first person to discover electricity was", "Franklin"),
#     # ("The first person to discover gravity was", "Newton"),
#     # ("The first person to discover America was", "Columbus"),
#     # ("The first person to discover penicillin was", "Fleming"),
#     # ("The first person to discover the theory of relativity was", "Einstein"),
# ]:


# # the essence of PCA calculation:
# C = X.T @ X
# C -= (
#     n_samples
#     * xp.reshape(self.mean_, (-1, 1))
#     * xp.reshape(self.mean_, (1, -1))
# )
# C /= n_samples - 1
# eigenvals, eigenvecs = xp.linalg.eigh(C)
