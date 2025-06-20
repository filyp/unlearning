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
from datasets import Dataset, concatenate_datasets, load_dataset
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from utils import loss_fns
from utils.data_loading import load_batches, load_fineweb_bio_corpus, load_local
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

wmdp_T = load_local(f"wmdp_deduped_bio/dev_T_corpus.jsonl")
wmdp_V = load_local(f"wmdp_deduped_bio/dev_V_corpus.jsonl")
wmdp_joined = concatenate_datasets([wmdp_T, wmdp_V])
fineweb_bio = load_fineweb_bio_corpus()
mmlu_bio = load_local("OUTDATED/my_generation2/mmlu_high_school_biology.jsonl")


def project_out(base, unwanted):
    unwanted = unwanted / unwanted.norm()
    magnitudes = (base * unwanted).sum(axis=-1)
    return pt.einsum("t,s->ts", magnitudes, unwanted)


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
        answer_mask = prepare_answer_mask(beginning_batch, full_batch)
        yield full_batch, answer_mask


def get_loss(model, full_batch, answer_mask=None, loss_fn_name="cross_entropy"):
    model.zero_grad(set_to_none=True)
    output = model(**full_batch, output_hidden_states=True)
    loss_fn = getattr(loss_fns, loss_fn_name)
    if loss_fn_name == "proj_out_target":
        return loss_fn(output, full_batch, answer_mask, model)
    else:
        return loss_fn(output, full_batch, answer_mask)


def save_act_hook(module, args):
    # ignore BOS token and the last token
    module.last_act_full = args[0].detach().clone()[:, 1:-1]


def save_grad_hook(module, args):
    # ignore BOS token and the last token
    last_grad_full = args[0].detach().clone()[:, 1:-1]

    flat_acts = module.last_act_full.flatten(end_dim=1)
    flat_grads = last_grad_full.flatten(end_dim=1)
    assert len(flat_acts.shape) == len(flat_grads.shape) == 2

    # when using batching, many grad norms are 0, so we need to remove them
    # todo, use answer mask rather than non_zero_grads
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

        # * freeze early layers
        if ".layers." in n:
            layer_num = int(n.split(".")[2])
            if layer_num < 16:
                p.requires_grad = False

    # * register hooks
    for n, module in trainable_modules(model):
        module.register_forward_pre_hook(save_act_hook)
        module.register_full_backward_pre_hook(save_grad_hook)
    return model


def get_metrics(model):
    res = {}
    model.eval()

    # ! eval forget loss
    res["forget_loss"] = 0
    for full_batch, answer_mask in get_batches(wmdp_V, range(7, 10)):
        with pt.no_grad():
            res["forget_loss"] += get_loss(model, full_batch, answer_mask).item()

    # ! eval disruption loss
    res["mmlu_loss"] = 0
    for full_batch, _ in get_batches(mmlu_bio.select(range(24)), range(7, 10)):
        with pt.no_grad():
            res["mmlu_loss"] += get_loss(model, full_batch).item()
            # note that we calculate disruption on full output, not just the answer
    
    # ! eval fineweb
    res["fineweb_loss"] = 0
    for ex in fineweb_bio.select(range(8)):
        full_batch = tokenizer(ex["text"], **conf.tokenizer)
        with pt.no_grad():
            res["fineweb_loss"] += get_loss(model, full_batch).item()

    # ! eval wmdp mcq
    res["wmdp_acc"] = eval_on(wmdp_V, model, temperature=1)
    
    print(f"epoch {epoch} forget={res['forget_loss']:7.4f}, mmlu={res['mmlu_loss']:7.4f}, fineweb={res['fineweb_loss']:7.4f}, wmdp={res['wmdp_acc']:7.4f}")  # fmt: skip
    return res


# %% first pass through forget corpus, to collect acts and grads
pt.cuda.empty_cache()
model = prepare_model()

run_conf = SimpleNamespace(
    # lr=0.04,
    # lr=0.018,
    lr=0.006,
    normalize=False,
    # loss_fn_name="neg_cross_entropy",
    loss_fn_name="correct_logit",
    # loss_fn_name="proj_out_target",
    num_pc=10,
    # techniques=[],
    # techniques=["dm_agg"],
    # techniques=["act"],
    # techniques=["dm_act"],
    # techniques=["act", "grad", "pca", "CL2_clip_at+0"],
    techniques=["act", "pca", "CL_0", "freeze_early"],
    # techniques=["act", "grad"],
    # techniques=["act", "pca", "dm_act"],
    # techniques=["act", "grad", "pca"],
    # techniques=["dm_act", "dm_grad"],
)


# ! define training batches, caching the last layer activations
training_batches = []
for batch, answer_mask in get_batches(wmdp_joined, range(7)):
    batch["answer_mask"] = answer_mask
    with pt.no_grad():
        output = model(**batch, output_hidden_states=True)
    # watch out, it can be 600MB for dev set
    batch["last_act"] = output.hidden_states[-1].detach().to("cpu")
    training_batches.append(batch)

# ! gather acts and grads
acts_list = {n: [] for n, _ in trainable_modules(model)}
grads_list = {n: [] for n, _ in trainable_modules(model)}
for batch in training_batches:
    loss = get_loss(model, batch, batch["answer_mask"], run_conf.loss_fn_name)
    loss.backward()
    for n, module in trainable_modules(model):
        acts_list[n].append(module.last_act)
        grads_list[n].append(module.last_grad)

# ! collect acts and grads for mmlu
if "mmlu_ctrl" in run_conf.techniques:
    for full_batch, loss_mask in get_batches(mmlu_bio.select(range(24, 72)), range(1)):
        loss = get_loss(model, full_batch)
        loss.backward()
        for n, module in trainable_modules(model):
            acts_list[n].append(module.last_act)
            grads_list[n].append(module.last_grad)

# if "fineweb_ctrl" in run_conf.techniques:  # * note that may be outdated
#     for ex in fineweb_bio.select(range(8, 16)):
#         full_batch = tokenizer(ex["text"], **conf.tokenizer)
#         loss = get_loss(model, full_batch)
#         loss.backward()
#         for n, module in trainable_modules(model):
#             acts_list[n].append(module.last_act)
#             grads_list[n].append(module.last_grad)
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
    act_pca_components[n] = PCA_gpu(acts_flattened, n_components=run_conf.num_pc)

del model
pt.cuda.empty_cache()
model = prepare_model()
# normalization later on divides roughtly by 15, so use a higher LR than normally
optimizer = pt.optim.SGD(model.parameters(), lr=run_conf.lr)

run_name = " ".join(run_conf.techniques) + f" lr{run_conf.lr} pc{run_conf.num_pc}"
wandb.init(
    project="unlearning-wmdp4",
    name=run_name,
    group=f"tv",
    config=OmegaConf.to_container(conf),
)

# % full training loop
start_time = time.time()
for epoch in range(200):
    pt.cuda.empty_cache()

    # ! get metrics
    res = get_metrics(model)
    wandb.log(res)
    if res["fineweb_loss"] > 18.5 or res["mmlu_loss"] > 18.5:
        break

    # ! one epoch
    model.train()
    _norms = []
    for batch in training_batches:

        loss = get_loss(model, batch, batch["answer_mask"], run_conf.loss_fn_name)
        loss.backward()

        # ! here we modify the grad
        for n, m in trainable_modules(model):
            grads = m.last_grad
            acts = m.last_act
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
wandb.init(
    project="unlearning-wmdp4-retrain",
    name=run_name,
    group=f"tv",
    config=OmegaConf.to_container(conf),
)
optimizer = pt.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(30):
    # ! get metrics
    res = get_metrics(model)
    wandb.log(res)

    model.train()
    for batch, answer_mask in get_batches(wmdp_T, range(7)):
        model.zero_grad(set_to_none=True)
        loss = get_loss(model, batch)
        loss.backward()
        optimizer.step()

wandb.finish()

# %%
for q in wmdp_T:
    print(q["answer_core"])

# output = model(**training_batches[0], output_hidden_states=True)