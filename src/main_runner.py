# %%
# adapted from explorations/2_proj_training.py which also has some code for grad_proj,
#     dm grad and dm act, and the traditional dm,
#     mmlu evals, and per-token loss increase visualizations,
#     and context undisruption (a more fancy retaining technique)
# but here, we aim for more simlicity and dataset generality

# see main_runner_2025.08.18_configurable_control_set.py for:
#     jigsaw_threats implementation,
#     use_wikitext_as_retain option,
#     recording some batches and their per-token losses, to later see how they are disrupted
#    inspect per question acc

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
from IPython import get_ipython
from omegaconf import DictConfig, OmegaConf, open_dict
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

if get_ipython() is None:
    with hydra.initialize(config_path="../configs", version_base="1.2"):
        cfg = hydra.compose(config_name=args.config_name, overrides=remaining_args)
else:
    print("Running in Jupyter")
    cfg = OmegaConf.load("../configs/8b_threats.yaml")  # for debugging
with open_dict(cfg):
    cfg = OmegaConf.merge(cfg, cfg.experiment_list[cfg.experiment_number])

# ! setup
set_seeds(42)

num_gpus = pt.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
if num_gpus == 1:
    pt.set_default_device("cuda")
    device_main = pt.device("cuda")
    device_storage = pt.device("cuda")
elif num_gpus == 2:
    pt.set_default_device("cuda:0")
    device_main = pt.device("cuda:0")
    device_storage = pt.device("cuda:1")

tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = tokenizer.eos_token

# ! load wikitext batches
wikitext = load_local(repo_root() / "data" / "wikitext_16k.jsonl")
_txts = wikitext.shuffle(seed=42).batch(cfg.batch_size)
wikitext_batches = [
    tokenizer(x["text"], **cfg.tokenizer) for x in _txts.select(range(16))
]


if "bio" in cfg.dataset:
    retain_set = load_fineweb_bio_corpus()
    # T = load_local(f"wmdp_deduped_bio/T_corpus.jsonl")
    # V = load_local(f"wmdp_deduped_bio/V_corpus.jsonl")
    T = load_local(f"wmdp_deduped_bio/dev_T_corpus.jsonl")
    V = load_local(f"wmdp_deduped_bio/dev_V_corpus.jsonl")

elif "cyber" in cfg.dataset:
    retain_set = load_fineweb_tech_corpus()
    # T = load_local(f"wmdp_deduped_cyber/T_corpus.jsonl")
    # V = load_local(f"wmdp_deduped_cyber/V_corpus.jsonl")
    T = load_local(f"wmdp_deduped_cyber/dev_T_corpus.jsonl")
    V = load_local(f"wmdp_deduped_cyber/dev_V_corpus.jsonl")


T = T.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
V = V.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
print(f"{len(T)=}, {len(V)=}")
T_and_V = concatenate_datasets([T, V])
eval_qs = T_and_V if cfg.get("eval_on_all_questions", False) else V

if "pairs" in cfg.dataset:
    # note: dataset comparison experiment uses 3, not 7
    training_batches = load_batches_from_pairs_set(T_and_V, cfg, range(0, 7))
    retraining_batches = load_batches_from_pairs_set(T, cfg, range(0, 7))

elif "deebs" in cfg.dataset:
    deebs_corpus = load_local("wmdp_deduped_deebs_corpus.jsonl")
    t_txts = deebs_corpus.filter(lambda x: x["original_question"] in set(T["question"]))
    v_txts = deebs_corpus.filter(lambda x: x["original_question"] in set(V["question"]))
    t_and_v_txts = concatenate_datasets([t_txts, v_txts])

    training_batches = [
        tokenizer(texts, **cfg.tokenizer)
        for texts in t_and_v_txts.shuffle(seed=42).batch(cfg.batch_size)["text"]
    ]
    retraining_batches = [
        tokenizer(texts, **cfg.tokenizer)
        for texts in t_txts.shuffle(seed=42).batch(cfg.batch_size)["text"]
    ]

retain_batches = [
    tokenizer(x["text"], **cfg.tokenizer)
    for x in retain_set.shuffle(seed=42)
    .batch(cfg.retain_batch_size)
    .select(range(len(training_batches)))
]

# %%
def _get_loss(model, batches):
    loss_acc = 0
    for batch in batches:
        with pt.no_grad():
            output = model(**batch)
            loss_acc += cross_entropy(output, batch).item()
    return loss_acc / len(batches)


def get_metrics(model):
    res = {}
    model.eval()

    # * eval forget acc
    res["forget_acc_t0"], res["forget_acc_t1"] = eval_on(eval_qs, model)

    res["wikitext_loss"] = _get_loss(model, wikitext_batches)
    res["retain_loss"] = _get_loss(model, retain_batches[:16])
    res["training_loss"] = _get_loss(model, training_batches[:16])
    # todo recall loss on MCQ but when asking to genreate the answer
    #     and I guess for control also the false answers? or maybe we don't care about this, only about general disruption

    logging.info(res)
    return res


# %% setup

# * load model
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id, torch_dtype=pt.bfloat16, device_map=device_main
)
model.config.use_cache = False

# * set trainable params
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in cfg.target_modules)

if cfg.get("retain_to_original", False):
    original_weights = {
        n: m.weight.clone().detach().to(device_storage)
        # n: m.weight.clone().detach().to(pt.float8_e4m3fn).to(device_storage)
        for n, m in trainable_modules(model)
    }

install_hooks(model)

unit_optimizer = pt.optim.SGD(model.parameters(), lr=1.0)

# * cache the activations for context undisruption
if "retaining_rate" in cfg:
    model.train()
    if cfg.get("retain_on_context", False):
        _act_cache_batches = training_batches
    else:
        _act_cache_batches = retain_batches

    for batch in _act_cache_batches:
        with pt.no_grad():
            output = model(**batch, output_hidden_states=True)
        batch["original_last_act"] = output.hidden_states[-1].detach().to("cpu")

# * initialize kl_acc
for _, m in trainable_modules(model):
    m.kl_acc = pt.zeros_like(m.weight)


# %%
# script name -> project
# config name & hash -> group
# experiment number -> name
project_name = "unlearning/" + Path(__file__).relative_to(repo_root()).as_posix()
project_name = project_name.replace("/", "|")
# group = args.config_name + "_" + get_conf_hash(args.config_name)
group = args.config_name + "_" + "18.08.2025"  # todo change back
# remove experiment_number from remaining_args
_args = "_".join(str(v) for v in cfg.experiment_list[cfg.experiment_number].values())
remaining_args = [arg for arg in remaining_args if "experiment_number" not in arg]
run_name = f"{cfg.experiment_number}|{_args}|{'_'.join(remaining_args)}"
wandb.init(
    project=project_name,
    group=group,
    name=run_name,
    config=OmegaConf.to_container(cfg),
)

init_res = get_metrics(model)
wandb.log(init_res)
assert cfg.algorithm in ["CIR", "GA"]

# % full training loop
start_time = time.time()
act_to_collapse = None
for epoch in range(cfg.max_num_epochs):
    pt.cuda.empty_cache()

    acts_list = {n: [] for n, _ in trainable_modules(model)}
    grads_list = {n: [] for n, _ in trainable_modules(model)}

    # ! one epoch
    model.train()
    for i, batch in enumerate(training_batches):

        # ! unlearning loss
        model.zero_grad(set_to_none=True)
        # pt.cuda.empty_cache()
        output = model(**batch, output_hidden_states=True)
        answer_mask = batch["answer_mask"] if cfg.only_train_on_answer else None
        loss_fn = getattr(loss_fns, cfg.loss_fn_name)
        loss = loss_fn(output, batch, answer_mask)
        loss.backward()

        # ! here we modify the grad
        if cfg.algorithm == "CIR":
            for n, m in trainable_modules(model):
                acts = get_last_act(m, batch["attention_mask"], cfg.ignore_bos)
                grads = get_last_grad(m, batch["attention_mask"], cfg.ignore_bos)
                acts_list[n].append(acts.clone().to("cpu"))
                grads_list[n].append(grads.clone().to("cpu"))
                assert len(acts.shape) == len(grads.shape) == 2

                if act_to_collapse is None:
                    assert epoch == 0
                    continue

                # ! proj out the means and PCA components
                for comp in act_to_collapse[n]:
                    acts -= project_out(acts, comp)
                for comp in grad_to_collapse[n]:
                    grads -= project_out(grads, comp)

                # without the projections, this is the equivalent of normal backprop
                m.weight.grad = pt.einsum("ti,tj->ij", grads, acts)
                assert m.weight.grad.shape == m.weight.shape

            if act_to_collapse is None:
                continue

        scale_grads_(model, cfg.unlearning_rate)  # apply intended lr

        if "max_norm" not in globals():  # establish max_norm
            max_norm = get_update_norm(model) * 1
            print(f"max_norm: {max_norm:7.5f}")
        pt.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        unit_optimizer.step()  # unit_optimizer has lr=1.0

        if "retaining_rate" in cfg:
            if cfg.retain_on_context:
                # todo it would be possible to reusethe forward pass
                #     do it in the optimized version, but here, be general
                _mask = batch["attention_mask"].bool() & (~batch["answer_mask"].bool())
            else:
                # use retain_batches
                batch = retain_batches[i % len(retain_batches)]
                _mask = batch["attention_mask"].bool()

            model.zero_grad(set_to_none=True)
            pt.cuda.empty_cache()
            output = model(**batch, output_hidden_states=True)

            if cfg.retaining_loss_fn == "kl_loss":
                loss = kl_loss(output, batch, model, _mask)
            elif cfg.retaining_loss_fn == "cross_entropy":
                assert not cfg.retain_on_context, "not implemented"
                loss = cross_entropy(output, batch)

            loss.backward()

            if cfg.get("retain_to_original", False):
                # * only allow reverting retain updates
                for n, _ in trainable_modules(model):
                    w = model.get_submodule(n).weight
                    orig_w = original_weights[n].to(device_main, dtype=pt.bfloat16)
                    # filter out if diff=0 too:
                    mask = (w - orig_w).sign() != w.grad.sign()
                    # mask = ((w - orig_w).sign() * w.grad.sign()) == -1  # this mask is more permissive; but when computing in 16bit, it doesn't matter anyway, only in 8bit
                    w.grad[mask] = 0

            scale_grads_(model, cfg.retaining_rate)  # apply intended lr
            unit_optimizer.step()  # unit_optimizer has lr=1.0

    if cfg.algorithm == "CIR":
        # ! calculate means and PCA components
        # _start_time = time.time()
        model.zero_grad(set_to_none=True)
        pt.cuda.empty_cache()
        act_to_collapse = get_projections(acts_list, cfg.act_proj_num, cfg.cir_niter)
        grad_to_collapse = get_projections(grads_list, cfg.grad_proj_num, cfg.cir_niter)
        # logging.info(f"time taken to calculate PCA: {time.time() - _start_time:.2f}s")

    # ! get metrics
    res = get_metrics(model)
    wandb.log(res)
    if res["wikitext_loss"] > init_res["wikitext_loss"] * cfg.get("loss_budget", 1.01):
        break

wandb.finish()
print(f"time taken: {time.time() - start_time:.2f}s")


# %% retraining on T

if "retraining_epochs" not in cfg:
    exit(0)


wandb.init(
    project="ret_" + project_name,
    group=group,
    name=run_name,
    config=OmegaConf.to_container(cfg),
)
optimizer = pt.optim.SGD(model.parameters(), lr=cfg.retraining_rate)

# * get metrics
res = get_metrics(model)
wandb.log(res)

for epoch in range(cfg.retraining_epochs):
    model.train()
    for batch in retraining_batches:
        model.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = cross_entropy(output, batch)
        loss.backward()
        optimizer.step()

    # * get metrics
    res = get_metrics(model)
    wandb.log(res)

wandb.finish()
