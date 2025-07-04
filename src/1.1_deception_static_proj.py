# %%
# %load_ext autoreload
# %autoreload 2
import json
import logging
import os
import random
import time
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
from utils.common_cir import *
from utils.data_loading import load_batches, load_fineweb_edu_corpus, load_local
from utils.evals import eval_on, format_prompt
from utils.git_and_reproducibility import repo_root
from utils.hooks import CalcSimilarityHooks
from utils.plots import visualize, visualize_rgb
from utils.training import (
    PCA_gpu,
    get_grad,
    get_grads_dict,
    prepare_answer_mask,
    set_seeds,
    trainable_modules,
    trainable_params,
)

# plt dark theme
plt.style.use("dark_background")

logging.basicConfig(level=logging.INFO)

conf = OmegaConf.load(repo_root() / "configs/transferability.yaml")
conf.model_id = "meta-llama/Llama-3.2-3B"
# conf.model_id = "HuggingFaceTB/SmolLM-135M"
conf.target_modules = ["gate_proj"]
# conf.target_modules = ["down_proj"]
conf.device = "cuda" if pt.cuda.is_available() else "cpu"

# ! setup
set_seeds(42)
pt.set_default_device(conf.device)
tokenizer = AutoTokenizer.from_pretrained(conf.model_id)
tokenizer.pad_token = tokenizer.eos_token
model = prepare_model(conf)

deception_set = load_local("machiavelli/deception/psy-high.jsonl")
# wmdp = load_local(f"wmdp_deduped_bio/dev_T_corpus.jsonl")


def project_out(base, unwanted):
    # check dimensions
    _pos, _stream = base.shape
    (_stream2,) = unwanted.shape
    assert _stream == _stream2

    unwanted = unwanted / unwanted.norm()
    magnitudes = (base * unwanted).sum(axis=-1)
    return pt.einsum("t,s->ts", magnitudes, unwanted)


def get_loss(model, beginning_txt, full_txt, loss_fn):
    # note: assumes tokenizer is in scope
    # only grads on the answer

    beginning_batch = tokenizer(beginning_txt, **conf.tokenizer)
    batch = tokenizer(full_txt, **conf.tokenizer)
    answer_mask = prepare_answer_mask(beginning_batch, batch)

    output = model(**batch, output_hidden_states=True)
    return loss_fn(output, batch, answer_mask)


# %% create usable datasets
deceptive_pairs = []
undeceptive_pairs = []
for ex in deception_set:
    if not ex["alt_answers"]:
        print("skipping")
        continue

    beginning_txt = ex["context"][-400:]
    undeceptive_txt = f"{beginning_txt} {ex['alt_answers'][0]}"
    deceptive_txt = f"{beginning_txt} {ex['answer']}"

    undeceptive_pairs.append((beginning_txt, undeceptive_txt))
    deceptive_pairs.append((beginning_txt, deceptive_txt))
print(len(undeceptive_pairs), len(deceptive_pairs))

# %%
# * note: some actions in machiavelli, seem like responses, while they are actually "internal thoughts" summarizying the action, so clean them up?

run_conf = SimpleNamespace(
    loss_fn_name="correct_logit",
    num_pc=8,
)
loss_fn = getattr(loss_fns, run_conf.loss_fn_name)

# ! first pass through forget corpus, to collect acts and grads
act_means, act_pca_components = get_act_principal_components(
    model,
    # [tokenizer(f, **conf.tokenizer) for b, f in undeceptive_pairs[:15]],
    [tokenizer(f, **conf.tokenizer) for b, f in deceptive_pairs[:15]],
    num_pc=run_conf.num_pc,
)

# %% get the disruption grads
model.zero_grad(set_to_none=True)
for beginning_txt, full_txt in undeceptive_pairs[15:30]:
    loss = get_loss(model, beginning_txt, full_txt, loss_fn)
    loss.backward()

disr_grads = get_grads_dict(model)

# %% get the target grads
model.zero_grad(set_to_none=True)
for beginning_txt, full_txt in deceptive_pairs[15:30]:
    loss = get_loss(model, beginning_txt, full_txt, loss_fn)
    loss.backward()

target_grads = get_grads_dict(model)

# %% get the forget grad
forget_id = 31
assert forget_id >= 30
beginning_txt, full_txt = deceptive_pairs[forget_id]
batch = tokenizer(full_txt, **conf.tokenizer)

model.zero_grad(set_to_none=True)
loss = get_loss(model, beginning_txt, full_txt, loss_fn)
loss.backward()
model.zero_grad(set_to_none=True)


per_module_grads = {}
for n, module in trainable_modules(model):
    act_in = get_last_act(module, batch["attention_mask"])
    grad_out = get_last_grad(module, batch["attention_mask"])
    org_act_in = act_in.clone()

    # ! CIR
    act_in -= project_out(act_in, act_means[n])
    for pc in act_pca_components[n]:
        act_in -= project_out(act_in, pc)

    # # ! common core!
    # act_in = org_act_in - act_in

    per_module_grads[n] = pt.einsum("ti,to->oi", act_in, grad_out)

# %% visualize example x layer
# green is good transfer, red is bad transfer

_row = []
for n, _ in trainable_modules(model):
    forget_grad = per_module_grads[n]
    target_grad = target_grads[n + ".weight"]
    disr_grad = disr_grads[n + ".weight"]

    good_transfer = (target_grad * forget_grad).sum().item()
    bad_transfer = (disr_grad * forget_grad).sum().item()
    self_similarity = (forget_grad * forget_grad).sum().item()

    _row.append([np.clip(bad_transfer, min=0), good_transfer, 0])  # rgb
    # _row.append([np.clip(bad_transfer, min=0), self_similarity, 0])  # rgb

ratios = np.array(_row).reshape(1, -1, 3)

print("sum of disruption:", ratios[:, :, 0].sum())
print("sum of forget:", ratios[:, :, 1].sum())
print("ratio:", ratios[:, :, 0].sum() / ratios[:, :, 1].sum())
ratios2 = ratios.copy()
ratios2[:, :, 0] *= 1
visualize_rgb(ratios2, scale=348)
