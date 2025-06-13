# %%
# %load_ext autoreload
# %autoreload 2
import json
import logging
import os
import random
import time
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
from utils.plots import visualize, visualize_rgb
from utils.training import PCA_gpu, prepare_answer_mask, set_seeds, trainable_modules

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
# ! limit which parameters are trained
conf.target_modules = ["gate_proj"]
# conf.target_modules = ["up_proj"]
# conf.target_modules = ["down_proj"]

wmdp = load_local(f"wmdp_deduped_bio/dev_T_corpus.jsonl")
# fineweb_bio = load_fineweb_bio_corpus()
mmlu_bio = load_local("OUTDATED/my_generation2/mmlu_high_school_biology.jsonl")


def project_out(base, unwanted):
    unwanted = unwanted / unwanted.norm()
    magnitudes = (base * unwanted).sum(axis=-1)
    return pt.einsum("t,s->ts", magnitudes, unwanted)


# def get_grad_from_abcd_question(model, question):
#     beginning = format_prompt(question)
#     ending = ["A", "B", "C", "D"][question["answer"]]
#     full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
#     # if only_grad_ending:
#     beginning_batch = tokenizer(beginning, **conf.tokenizer)
#     loss_mask = prepare_answer_mask(beginning_batch, full_batch)
#     return get_grad(model, full_batch, loss_mask)


def get_batches(dataset, range_, batch_size=16):
    beginnings = []
    fulls = []
    for idx in range_:
        for q in dataset:
            beginnings.append(q["contexts"][idx])
            fulls.append(f"{q['contexts'][idx]} {q['answer_core']}")

    for i in range(0, len(beginnings), batch_size):
        b_txt = beginnings[i : i + batch_size]
        f_txt = fulls[i : i + batch_size]
        beginning_batch = tokenizer(b_txt, **conf.tokenizer)
        full_batch = tokenizer(f_txt, **conf.tokenizer)
        loss_mask = prepare_answer_mask(beginning_batch, full_batch)
        yield full_batch, loss_mask


def get_loss(model, full_batch, loss_mask=None):
    model.zero_grad(set_to_none=True)
    output = model(**full_batch)

    # mask out the beginning tokens
    if loss_mask is not None:
        full_batch["attention_mask"] *= loss_mask

    return loss_fns.cross_entropy(output, full_batch)


def save_act_hook(module, args):
    module.last_act_full = args[0].detach().clone()[:, 1:-1]


def save_grad_hook(module, args):
    last_grad_full = args[0].detach().clone()[:, 1:-1]

    flat_acts = module.last_act_full.flatten(end_dim=1)
    flat_grads = last_grad_full.flatten(end_dim=1)
    assert len(flat_acts.shape) == len(flat_grads.shape) == 2

    # when using batching, many grad norms are 0, so we need to remove them
    non_zero_grads = flat_grads.norm(dim=-1) != 0

    module.last_act = flat_acts[non_zero_grads]
    module.last_grad = flat_grads[non_zero_grads]


def prepare_model():
    # * load model
    model = AutoModelForCausalLM.from_pretrained(
        conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
    )
    model.config.use_cache = False
    # * set trainable params
    for n, p in model.named_parameters():
        p.requires_grad = any(pattern in n for pattern in conf.target_modules)
    # * register hooks
    for n, module in trainable_modules(model):
        module.register_forward_pre_hook(save_act_hook)
        module.register_full_backward_pre_hook(save_grad_hook)
    return model


# %% first pass through forget corpus, to collect acts and grads
model = prepare_model()

acts = {n: [] for n, _ in trainable_modules(model)}
grads = {n: [] for n, _ in trainable_modules(model)}
for full_batch, loss_mask in get_batches(wmdp, range(7)):
    loss = get_loss(model, full_batch, loss_mask)
    loss.backward()
    for n, module in trainable_modules(model):
        acts[n].append(module.last_act)
        grads[n].append(module.last_grad)

# % calculate projection basis
grad_means = {}
act_means = {}
act_pca_components = {}
for n, module in trainable_modules(model):
    acts_flattened = pt.cat(acts.pop(n)).float()
    grads_flattened = pt.cat(grads.pop(n)).float()
    grad_means[n] = grads_flattened.mean(axis=0)
    act_means[n] = acts_flattened.mean(axis=0)

    # ! calculate act PCA
    act_pca_components[n] = PCA_gpu(acts_flattened)

act_pca_components[n]

# %% full training loop

del model
model = prepare_model()
# pretty big because of normalization later on (it divides roughtly by 15)
lr = 0.02
optimizer = pt.optim.SGD(model.parameters(), lr=lr)

wandb.init(
    project="unlearning-wmdp4",
    # name=f"normal",
    # name=f"actmean",
    # name=f"actmean+gradmean",
    # name=f"actmean+gradmean+pca lr{lr}",
    name=f"dm_on_coarse_grad lr{lr}",
    # name=f"agem_on_coarse_grad lr{lr}",
    #
    group=f"norm",
    config=OmegaConf.to_container(conf),
)

_conf = dict(
    act_mean_proj=False,
    grad_mean_proj=False,
    act_pca_proj=False,
    dm_on_coarse_grad=True,
    agem_on_coarse_grad=False,
)
conf.update(_conf)

start_time = time.time()
for epoch in range(200):

    # ! eval forget loss
    forget_loss = 0
    model.eval()
    for full_batch, loss_mask in get_batches(wmdp, range(7, 10)):
        with pt.no_grad():
            forget_loss += get_loss(model, full_batch, loss_mask).item()

    # ! eval disruption loss
    disruption_loss = 0
    model.eval()
    for full_batch, _ in get_batches(mmlu_bio.select(range(24)), range(7, 10)):
        with pt.no_grad():
            disruption_loss += get_loss(model, full_batch).item()
            # note that we calculate disruption on full output, not just the answer
    
    # todo eval fineweb
    # todo eval wmdp mcq

    print(f"forget_loss={forget_loss:7.4f}, disruption_loss={disruption_loss:7.4f}")
    wandb.log({"forget_loss": forget_loss, "disruption_loss": disruption_loss})
    if disruption_loss > 18.4:
        break

    # ! one epoch
    model.train()
    for full_batch, loss_mask in get_batches(wmdp, range(7)):
        loss = get_loss(model, full_batch, loss_mask)
        loss *= -1  # sign flip to unlearn
        loss.backward()

        # ! here we modify the grad
        for n, m in trainable_modules(model):
            grads = m.last_grad
            acts = m.last_act
            assert len(acts.shape) == len(grads.shape) == 2

            # ! proj out the means
            if conf.act_mean_proj:
                acts -= project_out(acts, act_means[n])
            if conf.grad_mean_proj:
                grads -= project_out(grads, grad_means[n])

            # ! proj out PCA components
            if conf.act_pca_proj:
                for comp in act_pca_components[n]:
                    acts -= project_out(acts, comp)

            # without the projections, this is equivalent to normal backprop
            m.weight.grad = pt.einsum("ti,tj->ij", grads, acts)
            assert m.weight.grad.shape == m.weight.shape

            # * DM on coarse 2D grad
            if conf.dm_on_coarse_grad:
                coarse = grad_means[n].reshape(-1, 1) @ act_means[n].reshape(1, -1)
                m.weight.grad[m.weight.grad.sign() != coarse.sign()] = 0
            
            # * A-GEM on coarse 2D grad
            if conf.agem_on_coarse_grad:
                coarse = grad_means[n].reshape(-1, 1) @ act_means[n].reshape(1, -1)
                magn = (m.weight.grad * coarse).sum() / (coarse * coarse).sum()
                m.weight.grad -= magn * coarse

        # ! normalize grads
        update_norm = (
            sum(m.weight.grad.norm() ** 2 for _, m in trainable_modules(model)) ** 0.5
        )
        for n, m in trainable_modules(model):
            m.weight.grad /= update_norm

        optimizer.step()

wandb.finish()
print(f"time taken: {time.time() - start_time:.2f}s")

# %%
