# %%
# %load_ext autoreload
# %autoreload 2
import itertools
import logging
import time
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch as pt
import wandb
from datasets import concatenate_datasets, load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from utils import loss_fns
from utils.common_cir import *
from utils.data_loading import load_fineweb_bio_corpus, load_jigsaw_dataset, load_local
from utils.evals import eval_on
from utils.git_and_reproducibility import repo_root
from utils.training import PCA_gpu, prepare_answer_mask, set_seeds, trainable_modules

# plt dark theme
plt.style.use("dark_background")

logging.basicConfig(level=logging.INFO)
conf = OmegaConf.load(repo_root() / "configs/transferability.yaml")
conf.model_id = "meta-llama/Llama-3.2-1B"
conf.target_modules = ["gate_proj", "up_proj", "down_proj"]
conf.device = "cuda" if pt.cuda.is_available() else "cpu"
batch_size = 8

# ! setup
set_seeds(42)
pt.set_default_device(conf.device)
tokenizer = AutoTokenizer.from_pretrained(conf.model_id)
tokenizer.pad_token = tokenizer.eos_token

wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
jigsaw = load_jigsaw_dataset()
jigsaw_threats = jigsaw[jigsaw["threat"] == 1]
jigsaw_benign = jigsaw[jigsaw["toxic"] == 0]

# filter out empty texts from wikitext
wikitext = wikitext.filter(lambda x: x["text"])


def get_loss(model, batch, answer_mask=None, loss_fn_name="cross_entropy"):
    model.zero_grad(set_to_none=True)
    output = model(**batch, output_hidden_states=True)
    loss_fn = getattr(loss_fns, loss_fn_name)
    return loss_fn(output, batch, answer_mask)


def get_metrics(model):
    res = {}
    model.eval()

    # ! eval disruption loss
    res["nontoxic_loss"] = 0
    for idx in range(0, 64, batch_size):
        texts = jigsaw_benign[idx : idx + batch_size]["comment_text"].tolist()
        batch = tokenizer(texts, **conf.tokenizer)
        with pt.no_grad():
            res["nontoxic_loss"] += get_loss(model, batch).item()

    # ! eval threats loss
    res["threats_loss"] = 0
    for idx in range(0, 64, batch_size):
        texts = jigsaw_threats[idx : idx + batch_size]["comment_text"].tolist()
        batch = tokenizer(texts, **conf.tokenizer)
        with pt.no_grad():
            res["threats_loss"] += get_loss(model, batch).item()

    # ! eval wikitext
    res["wikitext_loss"] = 0
    for idx in range(0, 64, batch_size):
        texts = wikitext.select(range(idx, idx + batch_size))["text"]
        full_batch = tokenizer(texts, **conf.tokenizer)
        with pt.no_grad():
            res["wikitext_loss"] += get_loss(model, full_batch).item()

    print(res)
    return res


# %%
if "model" in globals():
    # cleanup
    del model, acts_list, grads_list, act_means, grad_means, act_pca_components
    pt.cuda.empty_cache()
model = prepare_model(conf, use_every_n_layers=4)

run_conf = SimpleNamespace(
    # lr=0.001,
    lr=0.001,
    # lr=0.003,
    normalize=False,
    only_train_on_answer=True,
    # loss_fn_name="neg_cross_entropy",
    loss_fn_name="correct_logit",
    num_pc=10,
    # techniques=[],
    # techniques=["act", "pca", "control_on_threats"],
    techniques=["act", "pca"],
    # techniques=["act", "pca", "grad"],
    clip_at=10,
)

# ! first pass through forget corpus, to collect acts and grads
acts_list = {n: [] for n, _ in trainable_modules(model)}
grads_list = {n: [] for n, _ in trainable_modules(model)}

# gather acts and grads
for idx in range(64, 256, batch_size):
    if "control_on_threats" in run_conf.techniques:
        texts = jigsaw_threats[idx : idx + batch_size]["comment_text"].tolist()
    else:
        # the default is control on benign
        texts = jigsaw_benign[idx : idx + batch_size]["comment_text"].tolist()

    batch = tokenizer(texts, **conf.tokenizer)

    loss = get_loss(model, batch, loss_fn_name=run_conf.loss_fn_name)
    loss.backward()
    for n, module in trainable_modules(model):
        acts_list[n].append(get_last_act(module, batch["attention_mask"]))
        grads_list[n].append(get_last_grad(module, batch["attention_mask"]))

# ! calculate projection basis
grad_means = {}
act_means = {}
act_pca_components = {}
for n, module in trainable_modules(model):
    acts_flattened = pt.cat(acts_list.pop(n)).float()
    grads_flattened = pt.cat(grads_list.pop(n)).float()
    grad_means[n] = grads_flattened.mean(axis=0)
    act_means[n] = acts_flattened.mean(axis=0)
    # ! calculate act PCA
    # could also use the approximate pca_lowrank, but better to play it safe
    act_pca_components[n] = PCA_gpu(acts_flattened, n_components=run_conf.num_pc)

del model
pt.cuda.empty_cache()
model = prepare_model(conf, use_every_n_layers=4)
optimizer = pt.optim.SGD(model.parameters(), lr=run_conf.lr)

run_name = (
    f"{run_conf.loss_fn_name} lr{run_conf.lr} pc{run_conf.num_pc} clip{run_conf.clip_at} "
    + " ".join(run_conf.techniques)
)
wandb.init(
    project="unlearning-jigsaw",
    name=run_name,
    group=f"init",
    config=OmegaConf.to_container(conf),
)

# % full training loop
start_time = time.time()
for epoch in range(100):
    pt.cuda.empty_cache()

    # ! get metrics
    res = get_metrics(model)
    wandb.log(res)
    # if res["threats_loss"] > 35 or res["nontoxic_loss"] > 35:
    if res["wikitext_loss"] > 35 or res["nontoxic_loss"] > 35:
        break

    # ! one epoch
    model.train()

    _norms = []
    for idx in range(64, 256, batch_size):
        texts = jigsaw_threats[idx : idx + batch_size]["comment_text"].tolist()
        batch = tokenizer(texts, **conf.tokenizer)

        # ! unlearning loss
        model.zero_grad(set_to_none=True)
        output = model(**batch, output_hidden_states=True)
        loss_fn = getattr(loss_fns, run_conf.loss_fn_name)
        loss = loss_fn(
            output, batch, batch.get("answer_mask", None), clip_at=run_conf.clip_at
        )
        loss.backward()

        # ! here we modify the grad
        for n, m in trainable_modules(model):
            grads = get_last_grad(m, batch["attention_mask"])
            acts = get_last_act(m, batch["attention_mask"])
            assert len(acts.shape) == len(grads.shape) == 2

            # ! proj out the means
            if "act" in run_conf.techniques:
                acts -= project_out(acts, act_means[n])
            if "grad" in run_conf.techniques:
                grads -= project_out(grads, grad_means[n])

            # ! proj out PCA components
            if "pca" in run_conf.techniques:
                for comp in act_pca_components[n]:
                    acts -= project_out(acts, comp)

            # * DM out the means
            if "dm_act" in run_conf.techniques:
                # ignore channels which are too "normie"
                acts[acts.sign() == act_means[n].sign()] = 0
            if "dm_grad" in run_conf.techniques:
                # * wait, shouldn't this be flipped? why does this work better this way? anyway, they are both pretty bad
                grads[grads.sign() == grad_means[n].sign()] = 0

            # without the projections, this is equivalent to normal backprop
            m.weight.grad = pt.einsum("ti,tj->ij", grads, acts)
            assert m.weight.grad.shape == m.weight.shape

            # * DM on coarse 2D grad
            if "dm_agg" in run_conf.techniques:
                coarse = grad_means[n].reshape(-1, 1) @ act_means[n].reshape(1, -1)
                m.weight.grad[m.weight.grad.sign() != coarse.sign()] = 0

            # # * A-GEM on coarse 2D grad
            # if "agem_agg" in run_conf.techniques:
            #     coarse = grad_means[n].reshape(-1, 1) @ act_means[n].reshape(1, -1)
            #     magn = (m.weight.grad * coarse).sum() / (coarse * coarse).sum()
            #     m.weight.grad -= magn * coarse

        # ! normalize grads
        update_norm = (
            sum(m.weight.grad.norm() ** 2 for _, m in trainable_modules(model)) ** 0.5
        )
        _norms.append(update_norm.item())
        if run_conf.normalize:
            for n, m in trainable_modules(model):
                m.weight.grad /= update_norm

        optimizer.step()

    # for debug purposes
    print(f"{np.mean(_norms):7.2f}  ", end="")

wandb.finish()
print(f"time taken: {time.time() - start_time:.2f}s")


# %% retraining on T
# todo split into T and V?

wandb.init(
    project="unlearning-jigsaw-retrain",
    name=run_name,
    group=f"init",
    config=OmegaConf.to_container(conf),
)
optimizer = pt.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):

    # ! get metrics
    res = get_metrics(model)
    wandb.log(res)

    model.train()

    for idx in range(64, 256, batch_size):
        texts = jigsaw_threats[idx : idx + batch_size]["comment_text"].tolist()
        batch = tokenizer(texts, **conf.tokenizer)

        model.zero_grad(set_to_none=True)
        loss = get_loss(model, batch)
        loss.backward()
        optimizer.step()

wandb.finish()

# %%
