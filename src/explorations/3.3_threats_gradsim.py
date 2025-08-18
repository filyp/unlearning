# %%
# adapted from explorations/2_proj_training.py which also has some code for grad_proj,
#     dm grad and dm act, and the traditional dm,
#     mmlu evals, and per-token loss increase visualizations,
#     and context undisruption (a more fancy retaining technique)
# but here, we aim for more simlicity and dataset generality

import argparse
import itertools
import logging
import time
from copy import deepcopy
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch as pt
from datasets import Dataset, concatenate_datasets, load_dataset
from omegaconf import DictConfig, OmegaConf, open_dict
from tensordict import TensorDict
from transformers import AutoTokenizer

import wandb
from utils import loss_fns
from utils.common_cir import *
from utils.data_loading import *
from utils.evals import eval_on
from utils.git_and_reproducibility import get_conf_hash, repo_root
from utils.loss_fns import cross_entropy, kl_loss
from utils.plots import print_colored_tokens
from utils.training import get_update_norm, scale_grads_, set_seeds, trainable_modules

# plt dark theme
plt.style.use("dark_background")

logging.basicConfig(level=logging.INFO)

# Parse just the config-name, let Hydra handle the rest
parser = argparse.ArgumentParser()
parser.add_argument("--config-name", default="cir")
args, remaining_args = parser.parse_known_args()
cfg = OmegaConf.load("../../configs/8b_threats.yaml")  # for debugging
with open_dict(cfg):
    cfg = OmegaConf.merge(cfg, cfg.experiment_list[cfg.experiment_number])

# ! setup
set_seeds(42)

num_gpus = pt.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
pt.set_default_device("cuda")
device_main = pt.device("cuda")
device_storage = pt.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = tokenizer.eos_token

# ! load wikitext batches
wikitext = load_local(repo_root() / "data" / "wikitext_16k.jsonl")
_txts = wikitext.shuffle(seed=42).batch(cfg.batch_size)
wikitext_batches = [
    tokenizer(x["text"], **cfg.tokenizer) for x in _txts.select(range(32))
]
wikitext_control_batches = [
    tokenizer(x["text"], **cfg.tokenizer) for x in _txts.select(range(32, 32 + 64))
]

jigsaw = load_jigsaw_dataset()
jigsaw_threats = Dataset.from_pandas(jigsaw[jigsaw["threat"] == 1])
jigsaw_benign = Dataset.from_pandas(jigsaw[jigsaw["toxic"] == 0])
# todo split jigsaw into T and V
# here splitting into T and V makes less sense than with independent facts
# but it's still nice to do
# raise NotImplementedError("Jigsaw dataset not implemented yet")
# retain_set = jigsaw_benign  # format batches properly


# %%

loss_eval_batches = [
    tokenizer(x["comment_text"], **cfg.tokenizer)
    for x in jigsaw_threats.shuffle(seed=42).batch(cfg.batch_size).select(range(32))
]
training_batches = [
    tokenizer(x["comment_text"], **cfg.tokenizer)
    for x in jigsaw_threats.shuffle(seed=42)
    .batch(cfg.batch_size)
    # .select(range(32, 32 + 128))
    .select(range(32, 32 + 64))
]

benign_eval_batches = [
    tokenizer(x["comment_text"], **cfg.tokenizer)
    for x in jigsaw_benign.shuffle(seed=42)
    .select(range(4096))  # otherwise batching is slow, going through full dataset
    .batch(cfg.batch_size)
    .select(range(32))
]
control_batches = [
    tokenizer(x["comment_text"], **cfg.tokenizer)
    for x in jigsaw_benign.shuffle(seed=42)
    .select(range(4096))  # otherwise batching is slow, going through full dataset
    .batch(cfg.batch_size)
    .select(range(32, 32 + 128))
]


# %% setup

# * load model
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id, torch_dtype=pt.bfloat16, device_map=device_main
)
model.config.use_cache = False

# * set trainable params
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in cfg.target_modules)

install_hooks(model)

# loss_fn = loss_fns.cross_entropy
loss_fn = loss_fns.correct_logit


def get_grads(model, batches):
    for batch in batches:
        model.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = loss_fn(output, batch)
        loss.backward()
    return TensorDict(
        {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    )


def tensor_dict_dot_product(a, b):
    acc = 0
    for k in a.keys():
        acc += (a[k].to(pt.float32) * b[k].to(pt.float32)).sum()
    return acc


def tensor_dict_cossim(a, b):
    a_dot_b = tensor_dict_dot_product(a, b)
    a_norm = tensor_dict_dot_product(a, a).sqrt()
    b_norm = tensor_dict_dot_product(b, b).sqrt()
    return a_dot_b / (a_norm * b_norm)


# %%
benign_grads = get_grads(model, benign_eval_batches)
threat_grads = get_grads(model, loss_eval_batches)
wikitext_grads = get_grads(model, wikitext_batches)
tensor_dict_cossim(threat_grads, wikitext_grads)

# %%
# control_batches = training_batches
control_batches = wikitext_control_batches

acts_list = {n: [] for n, _ in trainable_modules(model)}
grads_list = {n: [] for n, _ in trainable_modules(model)}

for i, batch in enumerate(control_batches):
    # ! unlearning loss
    model.zero_grad(set_to_none=True)
    output = model(**batch)
    loss = loss_fn(output, batch)
    loss.backward()

    for n, m in trainable_modules(model):
        acts = get_last_act(m, batch["attention_mask"])
        grads = get_last_grad(m, batch["attention_mask"])
        acts_list[n].append(acts.to("cpu"))
        grads_list[n].append(grads.to("cpu"))

# ! calculate means and PCA components
_start_time = time.time()
model.zero_grad(set_to_none=True)
pt.cuda.empty_cache()
act_to_collapse = get_projections(acts_list, 10, 16)
grad_to_collapse = get_projections(grads_list, 10, 16)
print(f"time taken to calculate PCA: {time.time() - _start_time:.2f}s")


# %%
a_num = 6
g_num = 1

grad_acc = TensorDict(
    {n: pt.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
)
for i, batch in enumerate(training_batches):

    # ! unlearning loss
    model.zero_grad(set_to_none=True)
    # pt.cuda.empty_cache()
    output = model(**batch, output_hidden_states=True)
    loss = loss_fn(output, batch)
    loss.backward()

    for n, m in trainable_modules(model):
        acts = get_last_act(m, batch["attention_mask"])
        grads = get_last_grad(m, batch["attention_mask"])
        assert len(acts.shape) == len(grads.shape) == 2

        # ! proj out the means and PCA components
        for comp in act_to_collapse[n][:a_num]:
            acts -= project_out(acts, comp)
        for comp in grad_to_collapse[n][:g_num]:
            grads -= project_out(grads, comp)

        # without the projections, this is the equivalent of normal backprop
        m.weight.grad = pt.einsum("ti,tj->ij", grads, acts)
        assert m.weight.grad.shape == m.weight.shape

    grad_acc += TensorDict(
        {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    )

t_sim = tensor_dict_cossim(grad_acc, threat_grads)
b_sim = tensor_dict_cossim(grad_acc, benign_grads)
w_sim = tensor_dict_cossim(grad_acc, wikitext_grads)
# print(f"{t_sim=:7.4f}   {b_sim=:7.4f}   {b_sim / t_sim=:7.4f}")
print(f"{t_sim=:7.4f}   {w_sim=:7.4f}   {w_sim / t_sim=:7.4f}")

# %% cross_entropy 
# 2, 1 seems optimal here --  it gives 35% ratio decrease (but 6, 1 is similarly good)
# 11, 1 when controling on the TRAINING SET gives 40% ratio decrease

# on wikitext ratio, we get 75% decrease with 6, 1!
# for wikitext when controlling on benign set, again 6, 1 loosk optimal, but now only 50% decrease
# g=0 instead of 1 is also not terrible

# %% correct_logit
# when controlling on benign set, 6, 0 is optimal, with a stunning 99.5% drop!
# but that extreme value is probably lucky - still 90% drop is realistic here
# on wikitext ratio, 90% drop with 6, 1, and about 85% on 6,0

# when controlling on threats set
# 6,0 gives a nice 85% drop, 6,1 similarly
# on wikitext ratio, 6,0 is a bit better, with a stunning 95% drop

# we can also try to control on wikitext
# here 6,1 is optimal, but the drop is 'only' 75% for b ratio, and also 75% for w ratio

# %% conclusions
# controlling on the actual training set generally works better!
# grad mean projections help almost not at all, and grad pca actually hurts!
# correct_logit provides better selectivity