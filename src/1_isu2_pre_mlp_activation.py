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
from utils.plots import visualize, visualize_rgb
from utils.training import (
    get_grad,
    prepare_answer_mask,
    set_seeds,
    trainable_modules,
    trainable_params,
    PCA_gpu,
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
mmlu_bio = load_local("OUTDATED/my_generation2/mmlu_high_school_biology.jsonl")


def get_grad_from_example(model, beginning, ending, only_grad_ending=True):
    full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
    if only_grad_ending:
        beginning_batch = tokenizer(beginning, **conf.tokenizer)
        loss_mask = prepare_answer_mask(beginning_batch, full_batch)
        return get_grad(model, full_batch, loss_mask)
    else:
        return get_grad(model, full_batch)


def project_out(base, unwanted):
    unwanted = unwanted / unwanted.norm()
    magnitudes = (base * unwanted).sum(axis=-1)
    return pt.einsum("t,s->ts", magnitudes, unwanted)


def get_grad_from_abcd_question(model, question):
    beginning = format_prompt(question)
    ending = ["A", "B", "C", "D"][question["answer"]]
    return get_grad_from_example(model, beginning, ending)


# %%
def save_act_hook(module, args):
    module.last_act = args[0].detach().clone()


def save_grad_hook(module, args):
    last_grad = args[0].detach().clone()
    last_grad = last_grad[0, 1:-1]
    last_act = module.last_act[0, 1:-1]
    # last_grad = last_grad[0, :-1]  # include BOS
    # last_act = module.last_act[0, :-1]  # include BOS
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


forget_id = 21
assert forget_id >= 6, "first six are disr evals"
q = wmdp[forget_id]
context = q["contexts"][0]
beg_batch = tokenizer(context, **conf.tokenizer)
beginning_len = len(beg_batch["input_ids"][0])
get_grad_from_example(model, context, q["answer_core"], only_grad_ending=True)
for q_id in range(6, len(wmdp)):
    if q_id == forget_id:
        continue
    _q = wmdp[q_id]
    # _q = mmlu_bio[q_id]
    for context in _q["contexts"][:10]:
        beg_batch = tokenizer(context, **conf.tokenizer)
        beginning_len = len(beg_batch["input_ids"][0])
        get_grad_from_example(model, context, _q["answer_core"], only_grad_ending=True)

for n, module in trainable_modules(model):
    module._forward_pre_hooks.clear()
    module._backward_pre_hooks.clear()

# ! target grads

forget_grads = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})
_count = 0
for beginning in q["contexts"][1:]:
    forget_grads += get_grad_from_example(model, beginning, q["answer_core"])
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

    # # # ! project out the mean, rather than subtracting
    # # acts[acts.sign() == act_mean.sign()] = 0
    # # grads[grads.sign() == grad_mean.sign()] = 0
    # acts -= project_out(acts, act_mean)
    # grads -= project_out(grads, grad_mean)

    # # ! project out acts PCs
    # #  but from acts, improvement is almost 2x
    # # v = acts_flattened[grad_norms > 0.0]
    # v = acts_flattened
    # components = PCA_gpu(v, n_components=10)
    # for pc1 in components:
    #     acts -= project_out(acts, pc1)

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
            model, context, q["answer_core"], only_grad_ending=False
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

print("sum of disruption:", ratios[:, :, 0].sum())
print("sum of forget:", ratios[:, :, 1].sum())
print("ratio:", ratios[:, :, 0].sum() / ratios[:, :, 1].sum())
ratios[:, :, 0] *= 30
# visualize_rgb(ratios.mean(axis=0, keepdims=True), scale=80)
visualize_rgb(ratios, scale=300)

# %%
q = wmdp[forget_id]
context = q["contexts"][0]
print(context + " " + q["answer_core"])
for txt, disr in zip(disr_texts, ratios[:, :, 0].sum(axis=1)):
    print(disr, txt)

# * visualize ratio per layer
disr_per_layer = ratios[:, :, 0].sum(axis=0)
forget_per_layer = ratios[:, :, 1].sum(axis=0)
# visualize(disr_per_layer)
visualize(disr_per_layer / forget_per_layer)

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
