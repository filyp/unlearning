# %%
# usage:
# python src/main_runner.py --config-name=CONFIG_NAME --exp-num=NUM
# for example:
# python src/main_runner.py --config-name=cb --exp-num=3
#
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
from collections import Counter
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
from utils.common_cir import _get_projections
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
parser.add_argument("--config-name")
parser.add_argument("--exp-num", type=int)
parser.add_argument("--group-name", type=str, default=None)
args, remaining_args = parser.parse_known_args()

if get_ipython() is not None:
    args.config_name = "main_comparison_llama_bio"
    args.exp_num = 0
    remaining_args = ["model_id=meta-llama/Llama-3.2-1B"]  # locally we use only 1B

with hydra.initialize(config_path="../configs", version_base="1.2"):
    # Load base config without overrides first
    base_cfg = hydra.compose(config_name=args.config_name)
    cfg = OmegaConf.merge(
        base_cfg, 
        base_cfg.experiment_list[args.exp_num],
        OmegaConf.from_dotlist(remaining_args)
    )


# ! setup
set_seeds(42)

num_gpus = pt.cuda.device_count()
logging.info(f"Number of GPUs available: {num_gpus}")
device_main = pt.device("cuda")
device_storage = pt.device("cuda")
# if num_gpus == 1:
#     pt.set_default_device("cuda")
#     device_main = pt.device("cuda")
#     device_storage = pt.device("cuda")
# elif num_gpus == 2:
#     pt.set_default_device("cuda:0")
#     device_main = pt.device("cuda:0")
#     device_storage = pt.device("cuda:1")

tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = tokenizer.eos_token

# ! load wikitext batches
wikitext = load_local(repo_root() / "data" / "wikitext_16k.jsonl")
wikitext_batches = [
    tokenizer(x["text"], **cfg.tokenizer)
    for x in wikitext.shuffle(seed=42).batch(cfg.wikitext_batch_size)
]


_corpus_version = "corpus_simple" if "simple" in cfg.dataset else "corpus"
is_dev = "dev_" if cfg.use_dev_split else ""
if "bio" in cfg.dataset:
    retain_set = load_fineweb_bio_corpus()
    T = load_local(f"wmdp_deduped_bio/{is_dev}T_{_corpus_version}.jsonl")
    V = load_local(f"wmdp_deduped_bio/{is_dev}V_{_corpus_version}.jsonl")

elif "cyber" in cfg.dataset:
    retain_set = load_fineweb_tech_corpus()
    T = load_local(f"wmdp_deduped_cyber/{is_dev}T_{_corpus_version}.jsonl")
    V = load_local(f"wmdp_deduped_cyber/{is_dev}V_{_corpus_version}.jsonl")
else:
    raise ValueError(f"Unknown dataset: {cfg.dataset}")


T = T.filter(lambda x: x[cfg.model_id.split("/")[-1]] > 0.25)
V = V.filter(lambda x: x[cfg.model_id.split("/")[-1]] > 0.25)
T_and_V = concatenate_datasets([T, V])
eval_qs = T_and_V if cfg.get("eval_on_all_questions", False) else V
logging.info(f"{len(T)=}, {len(V)=}, {len(eval_qs)=}")

if "pairs" in cfg.dataset:
    only_ans = "only_ans" in cfg.dataset
    training_batches = load_batches_from_pairs_set(T_and_V, cfg, only_ans)
    retraining_batches = load_batches_from_pairs_set(T, cfg, only_ans)

elif "simple" in cfg.dataset:
    training_batches = load_batches_from_simple_set(T_and_V, cfg)
    retraining_batches = load_batches_from_simple_set(T, cfg)

elif "deebs" in cfg.dataset:
    deebs_corpus = load_local("wmdp_deduped_deebs_corpus.jsonl")
    t_txts = deebs_corpus.filter(lambda x: x["original_question"] in set(T["question"]))
    v_txts = deebs_corpus.filter(lambda x: x["original_question"] in set(V["question"]))
    t_and_v_txts = concatenate_datasets([t_txts, v_txts])

    training_batches = [
        tokenizer(texts, **cfg.tokenizer)
        for texts in t_and_v_txts.shuffle(seed=42).batch(cfg.train_batch_size)["text"]
    ]
    retraining_batches = [
        tokenizer(texts, **cfg.tokenizer)
        for texts in t_txts.shuffle(seed=42).batch(cfg.train_batch_size)["text"]
    ]

retain_batches = [
    tokenizer(x["text"], **cfg.tokenizer)
    for x in retain_set.shuffle(seed=42).batch(cfg.retain_batch_size)
    # .select(range(max(len(training_batches), cfg.num_eval_batches)))
]

# retain_batches = wikitext_batches[-len(retain_batches):]

recall_batches = load_recall_batches(eval_qs, cfg, batch_size=1)

# * mask out the most common tokens
if cfg.mask_n_most_common_tokens is not None:
    # count the most common tokens in the retain set
    counter = Counter()
    for b in retain_batches:
        counter.update(b["input_ids"].flatten().tolist())
    nt = cfg.mask_n_most_common_tokens
    most_common_tokens = pt.tensor([t for t, _ in counter.most_common(nt)])
    # mask out the common tokens
    for b in training_batches:
        b["answer_mask"] = ~pt.isin(b["input_ids"], most_common_tokens)
        # AND with the attention mask, just in case
        b["answer_mask"] = b["answer_mask"] & b["attention_mask"].bool()

    coverage = sum(c for _, c in counter.most_common(nt)) / sum(counter.values())
    logging.info(f"coverage: {coverage:.2f}")


def _get_loss(model, batches, use_answer_mask=False):
    loss_acc = 0
    for batch in batches:
        with pt.no_grad():
            output = model(**batch)
            if use_answer_mask:
                answer_mask = batch["answer_mask"]
                loss_acc += cross_entropy(output, batch, answer_mask).item()
            else:
                loss_acc += cross_entropy(output, batch).item()
    return loss_acc / len(batches)


def get_metrics(model):
    res = {}
    model.eval()

    # * eval forget acc
    res["forget_acc_t0"], res["forget_acc_t1"] = eval_on(eval_qs, model)

    nb = cfg.num_eval_batches
    res["wikitext_loss"] = _get_loss(model, wikitext_batches[:nb])
    res["retain_loss"] = _get_loss(model, retain_batches[:nb])
    # res["training_loss"] = _get_loss(model, training_batches[:nb])
    res["recall_loss"] = _get_loss(model, recall_batches, use_answer_mask=True)

    logging.info(res)
    return res


# %% setup

# * load model
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id, torch_dtype=pt.bfloat16, device_map=device_main
)
model.config.use_cache = False
all_layers = model.model.layers  # for trimmed model

if cfg.loss_fn_name in ["circuit_breaker", "mlp_confuse"]:  # trim the model
    max_layer = max(cfg.layer_range)
    model.model.layers = model.model.layers[: max_layer + 1]
    if cfg.get("cb_retaining_layers"):
        assert max(cfg.cb_retaining_layers) <= max_layer


# * set trainable params
logging.info(f"target_modules: {cfg.target_modules}")
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in cfg.target_modules)
    if p.requires_grad:
        logging.info(f"training {n}")

if cfg.get("retain_to_original", False) or cfg.get("decay_to_orig", False):
    # assert num_gpus == 2
    original_weights = {
        n: m.weight.clone().detach().to(device_storage)
        # n: m.weight.clone().detach().to(pt.float8_e4m3fn).to(device_storage)
        for n, m in trainable_modules(model)
        if int(n.split(".")[2]) <= max(cfg.layer_range)
    }

install_hooks(model)

unit_optimizer = pt.optim.SGD(model.parameters(), lr=1.0)
retraining_optimizer = pt.optim.SGD(model.parameters(), lr=cfg.retraining_rate)


# %%

# * cache the activations for circuit breaker retaining
retain_batches = retain_batches[: len(training_batches)]
if (cfg.get("retaining_rate", 0) > 0) and ("cb_retain" in cfg.retaining_loss_fns):
    # if cfg.get("retain_on_neg_mask", False):
    # _act_cache_batches = training_batches
    for batch in retain_batches:
        with pt.no_grad():
            output = model(**batch, output_hidden_states=True)
        batch["retain_acts"] = {
            l_num: output.hidden_states[l_num].detach().to("cpu")
            for l_num in cfg.cb_retaining_layers
        }

if cfg.loss_fn_name == "circuit_breaker":
    # * cache the activations for circuit breaker
    for batch in training_batches:
        with pt.no_grad():
            output = model(**batch, output_hidden_states=True)
        _mask = batch.get("answer_mask", batch["attention_mask"])
        _mask = _mask.bool().clone()
        _mask[:, : cfg.cut_off_tokens] = False
        batch["act_for_cb"] = {}
        batch["avg_act_norm"] = {}
        for layer_id in range(*cfg.layer_range):
            full_act = output.hidden_states[layer_id].detach()
            _act = full_act[_mask]
            batch["act_for_cb"][layer_id] = _act.cpu()
            batch["avg_act_norm"][layer_id] = _act.float().norm(dim=-1).mean().cpu()
            # batch["avg_act_norm"] = _act.float().norm(dim=-1, keepdim=True).cpu()

    # # todo: probably can delete this in the future
    # # * get the projections
    # rep_list = []
    # for batch in training_batches:
    #     rep_list.append(batch["act_for_cb"])
    # reps_flattened = pt.cat(rep_list)
    # rep_to_collapse = _get_projections(reps_flattened, cfg.rep_proj_num, cfg.cir_niter)

    # # * project out the representations
    # for batch in training_batches:
    #     rep = batch["act_for_cb"].to(device_main)
    #     for comp in rep_to_collapse:
    #         rep -= project_out(rep, comp)
    #     batch["act_for_cb"] = rep.cpu()


if cfg.loss_fn_name in ["mlp_confuse"]:
    # * install hooks for MLPs
    def save_acts_hook(module, args, output):
        module.cached_in = args[0]
        module.cached_out = output

    def stop_grad_hook(module, grad_input, grad_output):
        if grad_input[0] is None:
            # this happens on layer 0, with requires_grad=False on 1st MLP layer
            return
        # return [pt.zeros_like(grad_input[0])] + list(grad_input[1:])
        return [None] + list(grad_input[1:])

    for layer_id in range(*cfg.layer_range):
        model.model.layers[layer_id].mlp.register_forward_hook(save_acts_hook)
        # if cfg.mlp_stop_grad:
        #     model.model.layers[layer_id].mlp.gate_proj.register_full_backward_hook(stop_grad_hook)
        #     model.model.layers[layer_id].mlp.up_proj.register_full_backward_hook(stop_grad_hook)

    # * cache the activations for MLP confusion
    for batch in training_batches:
        with pt.no_grad():
            output = model(**batch)
        _mask = batch.get("answer_mask", batch["attention_mask"])
        _mask = _mask.bool().clone()
        _mask[:, : cfg.cut_off_tokens] = False
        batch["org_mlp_out"] = {}
        batch["org_mlp_out_norm"] = {}
        batch["org_mlp_in"] = {}
        for layer_id in range(*cfg.layer_range):
            mlp = model.model.layers[layer_id].mlp
            out = mlp.cached_out.detach()[_mask]
            batch["org_mlp_out"][layer_id] = out.cpu()
            batch["org_mlp_out_norm"][layer_id] = out.float().norm(dim=-1).mean().cpu()
            batch["org_mlp_in"][layer_id] = mlp.cached_in.detach().cpu()


    # # * cache the activations for mlp confuse retaining
    # retain_batches = retain_batches[: len(training_batches)]
    # if (cfg.get("retaining_rate", 0) > 0) and ("mlp_confuse_retain" in cfg.retaining_loss_fns):
    #     for batch in retain_batches:
    #         with pt.no_grad():
    #             output = model(**batch, output_hidden_states=True)
    #         _mask = batch.get("answer_mask", batch["attention_mask"])
    #         _mask = _mask.bool().clone()
    #         # _mask[:, : cfg.cut_off_tokens] = False  # do not do it! retain everywhere!
    #         batch["org_mlp_out_retain"] = {}
    #         batch["org_mlp_out_retain_norm"] = {}
    #         for layer_id in range(*cfg.mlp_retain_range):
    #             mlp = model.model.layers[layer_id].mlp
    #             out = mlp.cached_out.detach()[_mask]
    #             batch["org_mlp_out_retain"][layer_id] = out.cpu()
    #             batch["org_mlp_out_retain_norm"][layer_id] = out.float().norm(dim=-1).mean().cpu()

# %%
# script name -> project
# config name & hash -> group
# experiment number -> name
project_name = "unlearning/" + Path(__file__).relative_to(repo_root()).as_posix()
project_name = project_name.replace("/", "|")
group = args.group_name if args.group_name is not None else f"{args.config_name}_{get_conf_hash(args.config_name)}"
_args = "_".join(str(v) for v in cfg.experiment_list[args.exp_num].values())
run_name = f"{args.exp_num}|{_args}|{'_'.join(remaining_args)}"
wandb.init(
    project=project_name,
    group=group,
    name=run_name,
    config=OmegaConf.to_container(cfg),
)


model.model.layers = all_layers  # for trimmed model
init_res = get_metrics(model)
if cfg.loss_fn_name in ["circuit_breaker", "mlp_confuse"]:  # trim the model
    max_layer = max(cfg.layer_range)
    model.model.layers = model.model.layers[: max_layer + 1]

wandb.log(init_res)
assert cfg.algorithm in ["CIR", "GA"]

# % full training loop
start_time = time.time()
act_to_collapse = None
_retain_iter = 0
for epoch in range(cfg.max_num_epochs):
    pt.cuda.empty_cache()

    acts_list = {n: [] for n, _ in trainable_modules(model)}
    grads_list = {n: [] for n, _ in trainable_modules(model)}

    # ! one epoch
    model.train()
    for b_num, batch in enumerate(training_batches):

        # ! unlearning loss
        model.zero_grad(set_to_none=True)
        pt.cuda.empty_cache()
        answer_mask = batch.get("answer_mask", None)  # use answer_mask if it exists

        output = model(**batch, output_hidden_states=True)
        loss_fn = getattr(loss_fns, cfg.loss_fn_name)
        if cfg.loss_fn_name == "mlp_confuse":
            loss = loss_fn(model, batch, cfg, answer_mask)
        else:
            loss = loss_fn(output, batch, cfg, answer_mask)
        loss.backward()

        # ! here we modify the grad
        if cfg.algorithm == "CIR":
            for n, m in trainable_modules(model):
                if m.weight.grad is None:
                    continue
                acts = get_last_act(m, batch["attention_mask"], cfg.cut_off_tokens)
                grads = get_last_grad(m, batch["attention_mask"], cfg.cut_off_tokens)
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
                assert epoch == 0
                continue

        # scale_grads_(model, cfg.unlearning_rate)  # apply intended lr

        if b_num == 0:
            stats = dict(
                update_norm=get_update_norm(model),
                act_norm=output.hidden_states[4].norm(dim=-1).mean(),
            )

        # # * clip grad norm
        # if "max_norm" not in globals():  # establish max_norm
        #     max_norm = get_update_norm(model) * 1
        #     logging.info(f"max_norm: {max_norm:7.5f}")
        # pt.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm)
        # * normalize grads
        norm = get_update_norm(model)
        for p in model.parameters():
            if p.grad is not None:
                p.grad *= cfg.max_norm / norm

        unit_optimizer.step()  # unit_optimizer has lr=1.0

        model.zero_grad(set_to_none=True)
        pt.cuda.empty_cache()

        if cfg.get("retaining_rate", 0) > 0:
            if cfg.get("retain_on_neg_mask", False):
                # todo it would be possible to reuse the forward pass
                #     do it in the optimized version, but here, be general
                assert "answer_mask" in batch
                _mask = batch["attention_mask"].bool() & (~batch["answer_mask"].bool())
            else:
                # use retain_batches
                batch = retain_batches[_retain_iter % len(retain_batches)]
                _retain_iter += 1
                _mask = batch["attention_mask"].bool()

            output = model(**batch, output_hidden_states=True)
            # output = model(**batch)

            loss = 0
            if "kl_loss" in cfg.retaining_loss_fns:
                loss += kl_loss(output, batch, model, _mask)
            if "cross_entropy" in cfg.retaining_loss_fns:
                loss += cross_entropy(output, batch)
            if "cb_retain" in cfg.retaining_loss_fns:
                loss += loss_fns.cb_retain(output, batch, cfg)
            # if "mlp_confuse_retain" in cfg.retaining_loss_fns:
                # loss += loss_fns.mlp_confuse_retain(model, batch, cfg)

            loss.backward()

            if cfg.get("retain_to_original", False):
                # * only allow reverting retain updates
                for n, _ in trainable_modules(model):
                    w = model.get_submodule(n).weight
                    if w.grad is None:
                        continue
                    orig_w = original_weights[n]  # .to(device_main, dtype=pt.bfloat16)
                    # filter out if diff=0 too:
                    mask = (w - orig_w).sign() != w.grad.sign()
                    # mask = ((w - orig_w).sign() * w.grad.sign()) == -1  # this mask is more permissive; but when computing in 16bit, it doesn't matter anyway, only in 8bit
                    w.grad[mask] = 0

            scale_grads_(model, cfg.retaining_rate)  # apply intended lr
            unit_optimizer.step()  # unit_optimizer has lr=1.0

        if cfg.get("decay_to_orig", False):
            for n, orig_w in original_weights.items():
                w = model.get_submodule(n).weight
                w.data = orig_w * cfg.decay_to_orig + w.data * (1 - cfg.decay_to_orig)

    model.zero_grad(set_to_none=True)
    pt.cuda.empty_cache()

    if cfg.algorithm == "CIR" and epoch % cfg.get("pca_every_n", 1) == 0:
        # ! calculate means and PCA components
        # _start_time = time.time()
        act_to_collapse = get_projections(acts_list, cfg.act_proj_num, cfg.cir_niter)
        grad_to_collapse = get_projections(grads_list, cfg.grad_proj_num, cfg.cir_niter)
        # logging.info(f"time taken to calculate PCA: {time.time() - _start_time:.2f}s")
        if epoch == 0:
            continue  # no need to report metrics, because nothing has changed

    # ! get metrics
    model.model.layers = all_layers  # for trimmed model
    res = get_metrics(model)
    if cfg.loss_fn_name in ["circuit_breaker", "mlp_confuse"]:  # trim the model
        max_layer = max(cfg.layer_range)
        model.model.layers = model.model.layers[: max_layer + 1]

    wandb.log(res | stats)
    if res["wikitext_loss"] > init_res["wikitext_loss"] * cfg.get("loss_budget", 1.01):
        break

wandb.finish()
logging.info(f"time taken: {time.time() - start_time:.2f}s")

model.model.layers = all_layers  # for trimmed model

# %% retraining on T

if "retraining_epochs" not in cfg:
    exit(0)

# * set trainable params
mlp_modules = ["gate_proj", "up_proj", "down_proj"]
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in mlp_modules)
    if p.requires_grad:
        logging.info(f"training {n}")

wandb.init(
    project="ret_" + project_name,
    group=group,
    name=run_name,
    config=OmegaConf.to_container(cfg),
)

# * get metrics
res = get_metrics(model)
wandb.log(res)

for epoch in range(cfg.retraining_epochs):
    model.train()
    for batch in retraining_batches:
        pt.cuda.empty_cache()
        model.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = cross_entropy(output, batch)
        loss.backward()
        retraining_optimizer.step()

    # * get metrics
    res = get_metrics(model)
    wandb.log(res)

wandb.finish()
