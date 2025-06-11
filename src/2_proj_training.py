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
from utils.hooks import CalcSimilarityHooks, CollectActAndGrad
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


# %% first pass through forget corpus, to collect acts and grads
with CollectActAndGrad(model):
    for _q in wmdp:
        for context in _q["contexts"][:7]:
            get_grad_from_example(
                model, context, _q["answer_core"], only_grad_ending=True
            )

# % calculate projection basis
for n, module in trainable_modules(model):
    acts_flattened = pt.cat(module.saved_acts).float()
    grads_flattened = pt.cat(module.saved_grads).float()
    module.saved_acts = None
    module.saved_grads = None
    module.grad_mean = grads_flattened.mean(axis=0)
    module.act_mean = acts_flattened.mean(axis=0)

    # ! calculate act PCA
    module.act_pca_components = PCA_gpu(acts_flattened)

# %
optimizer = pt.optim.SGD(model.parameters(), lr=0.0001)


# %% full training loop
for epoch in range(10):

    # % eval forget loss
    forget_loss = 0
    model.eval()
    for idx in range(7, 10):
        for q in wmdp:
            beginning = q["contexts"][idx]
            ending = q["answer_core"]

            # tokenize and calculate model output
            full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
            model.zero_grad(set_to_none=True)
            with pt.no_grad():
                output = model(**full_batch)
            # mask out the beginning tokens
            beginning_batch = tokenizer(beginning, **conf.tokenizer)
            loss_mask = prepare_answer_mask(beginning_batch, full_batch)
            full_batch["attention_mask"] *= loss_mask
            # calculate loss and backward pass
            forget_loss += loss_fns.cross_entropy(output, full_batch).item()

    # % eval disruption loss
    disruption_loss = 0
    model.eval()
    for idx in range(7, 10):
        for q in mmlu_bio.select(range(24)):
            beginning = q["contexts"][idx]
            ending = q["answer_core"]

            # tokenize and calculate model output
            full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
            model.zero_grad(set_to_none=True)
            with pt.no_grad():
                output = model(**full_batch)
            # calculate loss and backward pass
            disruption_loss += loss_fns.cross_entropy(output, full_batch).item()

    print(f"forget_loss={forget_loss:6.2f}, disruption_loss={disruption_loss:6.2f}")

    # % one epoch
    model.train()
    for idx in range(7):
        for q in wmdp:
            beginning = q["contexts"][idx]
            ending = q["answer_core"]

            # tokenize and calculate model output
            full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
            model.zero_grad(set_to_none=True)
            output = model(**full_batch)
            # mask out the beginning tokens
            beginning_batch = tokenizer(beginning, **conf.tokenizer)
            loss_mask = prepare_answer_mask(beginning_batch, full_batch)
            full_batch["attention_mask"] *= loss_mask
            # calculate loss and backward pass
            loss = loss_fns.cross_entropy(output, full_batch)
            loss *= -1  # sign flip to unlearn
            loss.backward()

            # ! here we modify the grad
            for _, m in trainable_modules(model):

                # * DM on ref_grad - it's meh
                ref_grads = m.grad_mean.reshape(-1, 1) @ m.act_mean.reshape(1, -1)
                m.weight.grad[m.weight.grad.sign() != ref_grads.sign()] = 0
                
                # todo projections, but first we need to store 
            
            # todo maybe normalize grads?

            optimizer.step()
