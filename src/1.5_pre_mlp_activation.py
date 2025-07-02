# %%
# %load_ext autoreload
# %autoreload 2
import json
import time
import logging
import os
import random
from copy import deepcopy
from types import SimpleNamespace

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
from utils.plots import visualize, visualize_rgb
from utils.common_cir import *
from utils.training import (
    PCA_gpu,
    get_grad,
    prepare_answer_mask,
    set_seeds,
    trainable_modules,
    trainable_params,
)

# plt dark theme
plt.style.use("dark_background")

logging.basicConfig(level=logging.INFO)

conf = OmegaConf.load("../configs/transferability.yaml")
# conf.model_id = "meta-llama/Llama-3.2-1B"
conf.model_id = "HuggingFaceTB/SmolLM-135M"
# conf.target_modules = ["gate_proj"]
conf.target_modules = ["down_proj"]
conf.device = "cuda" if pt.cuda.is_available() else "cpu"

# ! setup
set_seeds(42)
pt.set_default_device(conf.device)
tokenizer = AutoTokenizer.from_pretrained(conf.model_id)
tokenizer.pad_token = tokenizer.eos_token
model = prepare_model(conf)

deception_set = load_local("machiavelli/deception/psy-high.jsonl")


# def get_grad_from_example(model, beginning, ending, only_grad_ending=True, loss_fn_name="cross_entropy"):
#     full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
#     if only_grad_ending:
#         beginning_batch = tokenizer(beginning, **conf.tokenizer)
#         loss_mask = prepare_answer_mask(beginning_batch, full_batch)
#         return get_grad(model, full_batch, loss_mask, loss_fn_name=loss_fn_name)
#     else:
#         return get_grad(model, full_batch, loss_fn_name=loss_fn_name)


def project_out(base, unwanted):
    unwanted = unwanted / unwanted.norm()
    magnitudes = (base * unwanted).sum(axis=-1)
    return pt.einsum("t,s->ts", magnitudes, unwanted)

# %%
# * note: some actions in machiavelli, seem like responses, while they are actually "internal thoughts" summarizying the action, so clean them up? 

run_conf = SimpleNamespace(
    loss_fn_name="correct_logit",
    # loss_fn_name = "cross_entropy",
    num_pc=8,
)
loss_fn = getattr(loss_fns, run_conf.loss_fn_name)

# ! first pass through forget corpus, to collect acts and grads
act_means, act_pca_components = get_act_principal_components(
    model,
    [f"{e['context'][-400:]} {e['answer']}" for e in deception_set.select(range(20))],
    num_pc=run_conf.num_pc,
)

# (pt.tensor(act_pca_components[n]) * ref).sum(dim=1).abs()
# ref = act_pca_components[n]

# note: on CPU, pca_lowrank is 3x faster than PCA_gpu
# but there are some differences (numerical?) on later components
# sklearn agrees with PCA_gpu, so looks that pca_lowrank is inaccurate after 8 PCs
# on CPU:
# PCA_gpu: 2s
# pca_lowrank: 0.7s
# sklearn pca: 26s! (and seems to be higher precision)
# wow, but for down_proj it's 30x faster!
# PCA_gpu: 60s
# pca_lowrank: 2s
# 

# %% get the disruption grads
model.zero_grad(set_to_none=True)
for ex in deception_set.select(range(20, 24)):
    if not ex["alt_answers"]:
        print("skipping")
        continue
    
    print(".")
    beginning_txt = ex["context"][-400:]
    full_txt = f"{beginning_txt} {ex['alt_answers'][0]}"

    beginning_batch = tokenizer(beginning_txt, **conf.tokenizer)
    batch = tokenizer(full_txt, **conf.tokenizer)
    answer_mask = prepare_answer_mask(beginning_batch, batch)

    output = model(**batch, output_hidden_states=True)
    loss = loss_fn(output, batch, answer_mask)
    loss.backward()

# %%

disr_grads = TensorDict(
    {n: p.grad for n, p in model.named_parameters() if p.requires_grad},
)
model.zero_grad(set_to_none=True)



# %%
ex
# %%



# * the first is "from" grad, the rest are "control" grads
forget_id = 20
assert forget_id >= 6, "first six are disr evals"
q = wmdp[forget_id]
context = q["contexts"][0]
beg_batch = tokenizer(context, **conf.tokenizer)
beginning_len = len(beg_batch["input_ids"][0])
get_grad_from_example(model, context, q["answer_core"], only_grad_ending=True, loss_fn_name=loss_fn_name)
for q_id in range(6, len(wmdp)):
    if q_id == forget_id:
        continue
    _q = wmdp[q_id]
    # _q = mmlu_bio[q_id]
    for context in _q["contexts"][:10]:
        beg_batch = tokenizer(context, **conf.tokenizer)
        beginning_len = len(beg_batch["input_ids"][0])
        get_grad_from_example(model, context, _q["answer_core"], only_grad_ending=True, loss_fn_name=loss_fn_name)

for n, module in trainable_modules(model):
    module._forward_pre_hooks.clear()
    module._backward_pre_hooks.clear()

# ! target grads

forget_grads = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})
_count = 0
for beginning in q["contexts"][1:]:
    forget_grads += get_grad_from_example(model, beginning, q["answer_core"], loss_fn_name=loss_fn_name)
    _count += 1
forget_grads /= _count

# forget_grads = get_grad_from_abcd_question(model, q)

# %% calculate intervention grad

per_module_grads = {}
for n, module in trainable_modules(model):
    ex_id = 0
    org_grads = module.saved_grads[ex_id]
    org_acts = module.saved_acts[ex_id]

    # grad_norms = pt.cat(module.saved_grads[1:]).norm(dim=-1)
    acts_flattened = pt.cat(module.saved_acts[1:]).float()
    grads_flattened = pt.cat(module.saved_grads[1:]).float()

    grad_mean = grads_flattened.mean(axis=0)
    # act_mean = acts_flattened[grad_norms > 0.1].mean(axis=0)
    act_mean = acts_flattened.mean(axis=0)

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

    # # ! project out the mean, rather than subtracting
    # acts[acts.sign() == act_mean.sign()] = 0
    # grads[grads.sign() == grad_mean.sign()] = 0
    acts -= project_out(acts, act_mean)
    grads -= project_out(grads, grad_mean)

    # ! project out acts PCs
    #  but from acts, improvement is almost 2x
    # v = acts_flattened[grad_norms > 0.0]
    v = acts_flattened
    components = PCA_gpu(v, n_components=10)
    for pc1 in components:
        acts -= project_out(acts, pc1)

    # # ! project out grads PCs
    # # from grads, it has a tiny effect
    # # v = grads_flattened[grad_norms > 0.]
    # v = grads_flattened
    # components = PCA_gpu(v, n_components=5)
    # for pc1 in components:
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

    # # * DM on ref_grad - it's bad
    # ref_grads = grad_mean.reshape(-1, 1) @ act_mean.reshape(1, -1)
    # grad[grad.sign() == ref_grads.sign()] = 0

    # # * mask all - doesn't do much
    # contrast_grad = pt.zeros_like(grad)
    # for ex_id in range(len(module.saved_grads)):
    #     contrast_grad += pt.einsum("ti,tj->ij", module.saved_grads[ex_id], module.saved_acts[ex_id])
    # grad[grad.sign() == contrast_grad.sign()] = 0
    
    # so those DM masks are bad even when we look at clipped disr (not abs)

    per_module_grads[n] = grad

# seems that subtracting has problems catching all nuance
# full rejection using DM seems better


# # %% calculate transfer
ratios = []
disr_texts = []
for q in wmdp.select(range(6)):
# for q in mmlu_bio.select(range(15)):
# for q in mmlu_bio.select(range(15, 30)):
    for context in q["contexts"][:2]:
        q = wmdp[18]
        assert q["answer_core"] == "Cholera", q["answer_core"]
        disr_texts.append(context + " " + q["answer_core"])
        disr_grads = get_grad_from_example(
            model, context, q["answer_core"], only_grad_ending=False, loss_fn_name=loss_fn_name
        )
        _row = []
        for n, _ in trainable_modules(model):
            interv_grads = per_module_grads[n]
            forget_grad = forget_grads[n + ".weight"]
            forget = (forget_grad * interv_grads).sum().item()
            disr_grad = disr_grads[n + ".weight"]
            disr = (disr_grad * interv_grads).sum().item()
            # _row.append([np.abs(disr), forget, 0])  # rgb
            _row.append([np.clip(disr, min=0), forget, 0])  # rgb
        ratios.append(_row)
ratios = np.array(ratios)

# ratios[:, :16] = 0
# ratios[:, -1] = 0

# ratios has shape (examples, layers, [disr, forget, 0])

# %% visualize example x layer
# green is good transfer, red is bad transfer

print("sum of disruption:", ratios[:, :, 0].sum())
print("sum of forget:", ratios[:, :, 1].sum())
print("ratio:", ratios[:, :, 0].sum() / ratios[:, :, 1].sum())
ratios2 = ratios.copy()
ratios2[:, :, 0] *= 1
visualize_rgb(ratios2, scale=1)

# %% visualize per layer (aggregated examples)
# green is high ration of disr to forget (so it's bad)
q = wmdp[forget_id]
context = q["contexts"][0]
print(context + " " + q["answer_core"])
for txt, disr in zip(disr_texts, ratios[:, :, 0].sum(axis=1)):
    print(disr, txt)

# * visualize ratio per layer
disr_per_layer = ratios[:, :, 0].sum(axis=0)
forget_per_layer = ratios[:, :, 1].sum(axis=0)
# visualize(disr_per_layer)
visualize(forget_per_layer)
# visualize(disr_per_layer / forget_per_layer)

# %% visualize weights
q = wmdp[1]
context = q["contexts"][0]

disr_grads = get_grad_from_example(model, context, q["answer_core"])
n = [n for n, _ in trainable_modules(model)][10]

interv_grads = per_module_grads[n]
forget_grad = forget_grads[n + ".weight"]
forget = forget_grad * interv_grads
disr_grad = disr_grads[n + ".weight"]
disr = disr_grad * interv_grads

print(disr.abs().max().item())
visualize(disr)
# visualize(forget)

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

# contexts = [
#     "The bacterial pathogen responsible for severe acute watery diarrhea in epidemic settings is",
#     "Which microorganism causes the characteristic 'rice-water' stools in infected individuals?",
#     "The Vibrio species that produces the cholera toxin causing massive fluid loss is",
#     "Among enteric pathogens, the one causing the most rapid and severe dehydration is",
#     "The bacterial agent that can lead to death within hours if left untreated is",
#     "Which pathogen's toxin causes the activation of adenylate cyclase in intestinal cells?",
#     "The microorganism responsible for the 19th century 'Blue Death' epidemics is",
#     "Among bacterial pathogens, the one causing the most dramatic fluid and electrolyte loss is",
#     "The Vibrio species that can survive in both fresh and saltwater environments is",
#     "Which pathogen's infection can lead to hypovolemic shock within 24 hours?",
#     "The bacterial agent that causes the characteristic 'fishy' odor in stool samples is",
#     "Among enteric pathogens, the one with the shortest incubation period is",
#     "The microorganism that can cause death through severe dehydration within hours is",
#     "Which pathogen's toxin causes the characteristic 'rice-water' diarrhea?",
#     "The bacterial agent that can lead to metabolic acidosis in untreated cases is",
# ]

# contexts = [
#     "The ecosystem with the highest biodiversity per square kilometer is a",
#     "Which environment contains more than half of the world's plant and animal species?",
#     "The habitat that experiences daily rainfall and constant high humidity is a",
#     "Among Earth's biomes, the one with the most complex vertical stratification is a",
#     "The environment where epiphytes and lianas form intricate aerial gardens is a",
#     "Which ecosystem has the highest rate of primary productivity on Earth?",
#     "The habitat that inspired the concept of 'vertical farming' in nature is a",
#     "Among terrestrial environments, the one with the most efficient nutrient cycling is a",
#     "The ecosystem where 25% of modern medicines were first discovered is a",
#     "Which environment contains the world's largest living carbon sink?",
#     "The habitat that hosts the most complex food webs on the planet is a",
#     "Among natural environments, the one with the most diverse canopy layers is a",
#     "The ecosystem where 70% of the world's flowering plants are found is a",
#     "Which environment contains more than 40,000 plant species per square kilometer?",
#     "The habitat that inspired the concept of 'biodiversity hotspots' is a",
# ]
#     "The habitat that inspired the concept of 'biodiversity hotspots' is a",
# ]
