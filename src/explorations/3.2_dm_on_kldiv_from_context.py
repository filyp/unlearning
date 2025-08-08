# %%
# adapted from explorations/2_proj_training.py which also has some code for grad_proj,
#     dm grad and dm act, and the traditional dm,
#     mmlu evals, and per-token loss increase visualizations,
#     and context undisruption (a more fancy retaining technique)
# but here, we aim for more simlicity and dataset generality

import argparse
import logging
import time
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch as pt
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

import wandb
from utils import loss_fns
from utils.common_cir import *
from utils.data_loading import *
from utils.evals import eval_on
from utils.git_and_reproducibility import get_conf_hash, repo_root
from utils.loss_fns import cross_entropy, _normalize_logits
from utils.training import get_update_norm, set_seeds, trainable_modules

# plt dark theme
plt.style.use("dark_background")

logging.basicConfig(level=logging.INFO)

# Parse just the config-name, let Hydra handle the rest
parser = argparse.ArgumentParser()
parser.add_argument("--config-name", default="cir")
args, remaining_args = parser.parse_known_args()

with hydra.initialize(config_path="../configs", version_base="1.2"):
    cfg = hydra.compose(config_name=args.config_name, overrides=remaining_args)
# cfg = OmegaConf.load("../configs/context_nondisruption.yaml")  # for debugging


exp_cfg = cfg.experiment_list[cfg.experiment_number]


# ! setup
set_seeds(42)
pt.set_default_device("cuda")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = tokenizer.eos_token

# ! load wikitext batches
wikitext = load_local(repo_root() / "data" / "wikitext_16k.jsonl")
_txts = wikitext.shuffle(seed=42).batch(cfg.batch_size)
wikitext_batches = [
    tokenizer(x["text"], **cfg.tokenizer) for x in _txts.select(range(16))
]

# ! load proper datasets
if cfg.dataset == "wmdp_bio":
    T = load_local(f"wmdp_deduped_bio/T_corpus.jsonl")
    V = load_local(f"wmdp_deduped_bio/V_corpus.jsonl")
    T = T.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
    V = V.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
    print(f"{len(T)=}, {len(V)=}")
    T_and_V = concatenate_datasets([T, V])
    eval_qs = T_and_V if cfg.get("eval_on_all_questions", False) else V

    # note: dataset comparison experiment uses 3, not 7
    training_batches = load_batches_from_pairs_set(T_and_V, cfg, range(0, 7))
    retraining_batches = load_batches_from_pairs_set(T, cfg, range(0, 7))
    loss_eval_batches = load_batches_from_pairs_set(eval_qs, cfg, range(7, 10))
    # optionally we could try retain set instead
    control_batches = training_batches

    retain_set = load_fineweb_bio_corpus()
    _txts = retain_set.shuffle(seed=42).batch(cfg.batch_size)
    retain_batches = [
        tokenizer(x["text"], **cfg.tokenizer)
        for x in _txts.select(range(len(training_batches)))
    ]
elif cfg.dataset == "wmdp_cyber":
    T = load_local(f"wmdp_deduped_cyber/T_corpus.jsonl")
    V = load_local(f"wmdp_deduped_cyber/V_corpus.jsonl")
    T = T.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
    V = V.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
    print(f"{len(T)=}, {len(V)=}")
    T_and_V = concatenate_datasets([T, V])
    eval_qs = T_and_V if cfg.get("eval_on_all_questions", False) else V

    # note: dataset comparison experiment uses 3, not 7
    training_batches = load_batches_from_pairs_set(T_and_V, cfg, range(0, 7))
    retraining_batches = load_batches_from_pairs_set(T, cfg, range(0, 7))
    loss_eval_batches = load_batches_from_pairs_set(eval_qs, cfg, range(7, 10))
    # optionally we could try retain set instead
    control_batches = training_batches

    retain_set = load_fineweb_tech_corpus()
    _txts = retain_set.shuffle(seed=42).batch(cfg.batch_size)
    retain_batches = [
        tokenizer(x["text"], **cfg.tokenizer)
        for x in _txts.select(range(len(training_batches)))
    ]

elif cfg.dataset == "wmdp_bio_deebs":
    T = load_local(f"wmdp_deduped_bio/T_corpus.jsonl")
    V = load_local(f"wmdp_deduped_bio/V_corpus.jsonl")
    T = T.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
    V = V.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
    print(f"{len(T)=}, {len(V)=}")
    T_and_V = concatenate_datasets([T, V])
    eval_qs = T_and_V if cfg.get("eval_on_all_questions", False) else V
    deebs_corpus = load_local("wmdp_deduped_deebs_corpus.jsonl")

    t_questions = set(T["question"])
    v_questions = set(V["question"])
    t_texts = deebs_corpus.filter(lambda x: x["original_question"] in t_questions)
    v_texts = deebs_corpus.filter(lambda x: x["original_question"] in v_questions)
    t_and_v_texts = concatenate_datasets([t_texts, v_texts])

    training_batches = [
        tokenizer(texts, **cfg.tokenizer)
        for texts in t_and_v_texts.shuffle(seed=42).batch(cfg.batch_size)["text"]
    ]
    retraining_batches = [
        tokenizer(texts, **cfg.tokenizer)
        for texts in t_texts.shuffle(seed=42).batch(cfg.batch_size)["text"]
    ]
    loss_eval_batches = load_batches_from_pairs_set(eval_qs, cfg, range(7, 10))
    # optionally we could try retain set instead
    control_batches = training_batches

    retain_set = load_fineweb_bio_corpus()
    _txts = retain_set.shuffle(seed=42).batch(cfg.batch_size)
    retain_batches = [
        tokenizer(x["text"], **cfg.tokenizer)
        for x in _txts.select(range(len(training_batches)))
    ]

elif cfg.dataset == "wmdp_cyber_deebs":
    T = load_local(f"wmdp_deduped_cyber/T_corpus.jsonl")
    V = load_local(f"wmdp_deduped_cyber/V_corpus.jsonl")
    T = T.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
    V = V.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
    print(f"{len(T)=}, {len(V)=}")
    T_and_V = concatenate_datasets([T, V])
    eval_qs = T_and_V if cfg.get("eval_on_all_questions", False) else V
    deebs_corpus = load_local("wmdp_deduped_deebs_corpus.jsonl")

    t_questions = set(T["question"])
    v_questions = set(V["question"])
    t_texts = deebs_corpus.filter(lambda x: x["original_question"] in t_questions)
    v_texts = deebs_corpus.filter(lambda x: x["original_question"] in v_questions)
    t_and_v_texts = concatenate_datasets([t_texts, v_texts])

    training_batches = [
        tokenizer(texts, **cfg.tokenizer)
        for texts in t_and_v_texts.shuffle(seed=42).batch(cfg.batch_size)["text"]
    ]
    retraining_batches = [
        tokenizer(texts, **cfg.tokenizer)
        for texts in t_texts.shuffle(seed=42).batch(cfg.batch_size)["text"]
    ]
    loss_eval_batches = load_batches_from_pairs_set(eval_qs, cfg, range(7, 10))
    # optionally we could try retain set instead
    control_batches = training_batches

    retain_set = load_fineweb_tech_corpus()
    _txts = retain_set.shuffle(seed=42).batch(cfg.batch_size)
    retain_batches = [
        tokenizer(x["text"], **cfg.tokenizer)
        for x in _txts.select(range(len(training_batches)))
    ]

elif cfg.dataset == "jigsaw_threats":
    jigsaw = load_jigsaw_dataset()
    jigsaw_threats = jigsaw[jigsaw["threat"] == 1]
    jigsaw_benign = jigsaw[jigsaw["toxic"] == 0]
    # todo split jigsaw into T and V
    # here splitting into T and V makes less sense than with independent facts
    # but it's still nice to do
    raise NotImplementedError("Jigsaw dataset not implemented yet")
    retain_set = jigsaw_benign  # format batches properly


# %%
def get_metrics(model):
    res = {}
    model.eval()

    # ! eval wikitext
    res["wikitext_loss"] = 0
    for batch in wikitext_batches:
        with pt.no_grad():
            output = model(**batch)
            res["wikitext_loss"] += cross_entropy(output, batch).item()
    res["wikitext_loss"] /= len(wikitext_batches)

    # ! eval forget acc
    if "wmdp" in cfg.dataset:
        res["forget_acc"] = eval_on(eval_qs, model, temperature=1)

        # eval forget loss - this one is rather optional
        res["forget_loss"] = 0
        res["context_loss"] = 0
        for batch in loss_eval_batches:
            with pt.no_grad():
                output = model(**batch)
                answer_mask = batch["answer_mask"]
                context_mask = batch["attention_mask"] & ~answer_mask
                res["forget_loss"] += cross_entropy(output, batch, answer_mask).item()
                res["context_loss"] += cross_entropy(output, batch, context_mask).item()
        res["forget_loss"] /= len(loss_eval_batches)
        res["context_loss"] /= len(loss_eval_batches)

    print(res)
    return res


# %%

# * load model
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)
model.config.use_cache = False

# * set trainable params
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in cfg.target_modules)

install_hooks(model)

optimizer = pt.optim.SGD(model.parameters(), lr=exp_cfg.unlearning_rate)
if "retaining_rate" in exp_cfg:
    retain_optimizer = pt.optim.SGD(model.parameters(), lr=exp_cfg.retaining_rate)

# # inspect per question acc
# for ex in eval_qs:
#     acc = eval_on(Dataset.from_list([ex]), model, temperature=1)
#     print(acc)

# * cache the activations for context undisruption
if exp_cfg.get("kl_ratio_thresh") is not None:
    model.train()
    for batch in control_batches:
        with pt.no_grad():
            output = model(**batch, output_hidden_states=True)
        batch["original_last_act"] = output.hidden_states[-1].detach().to("cpu")

# %%
# script name -> project
# config name & hash -> group
# experiment number -> name
project_name = "unlearning/" + Path(__file__).relative_to(repo_root()).as_posix()
project_name = project_name.replace("/", "|")
group = args.config_name + "_" + get_conf_hash(args.config_name)
# remove experiment_number from remaining_args
remaining_args = [arg for arg in remaining_args if "experiment_number" not in arg]
run_name = "_".join(str(v) for v in exp_cfg.values()) + "|" + "_".join(remaining_args)
wandb.init(
    project=project_name,
    group=group,
    name=run_name,
    config=OmegaConf.to_container(exp_cfg),
)

initial_res = get_metrics(model)
wandb.log(initial_res | {"kl_div": 0})
assert exp_cfg.algorithm in ["CIR", "GA"]

# % full training loop
start_time = time.time()
for epoch in range(cfg.max_num_epochs):
    pt.cuda.empty_cache()

    # ! recalculate projections
    if epoch % exp_cfg.recalc_every_n_epochs == 0 and exp_cfg.algorithm == "CIR":
        acts_list = {n: [] for n, _ in trainable_modules(model)}
        grads_list = {n: [] for n, _ in trainable_modules(model)}

        for i, batch in enumerate(control_batches):
            # ! unlearning loss
            model.zero_grad(set_to_none=True)
            output = model(**batch)
            loss_fn = getattr(loss_fns, exp_cfg.loss_fn_name)
            answer_mask = batch["answer_mask"] if exp_cfg.only_train_on_answer else None
            loss = loss_fn(output, batch, answer_mask)
            loss.backward()

            for n, m in trainable_modules(model):
                acts = get_last_act(m, batch["attention_mask"])
                grads = get_last_grad(m, batch["attention_mask"])
                acts_list[n].append(acts.to("cpu"))
                grads_list[n].append(grads.to("cpu"))

        # ! calculate means and PCA components
        act_to_collapse = get_projections(acts_list, num_pc=10)
        grad_to_collapse = get_projections(grads_list, num_pc=10)

    # ! one epoch
    model.train()
    _kl_div_acc = 0
    for i, batch in enumerate(training_batches):
        model.zero_grad(set_to_none=True)
        output = model(**batch, output_hidden_states=True)

        if exp_cfg.get("kl_ratio_thresh") is not None:
            assert exp_cfg.only_train_on_answer
            # ! compute context disruption
            # output = model(**batch, output_hidden_states=True)  # already calculated
            cntxt_mask = batch["attention_mask"].bool() & (~batch["answer_mask"].bool())
            original_last_act = batch["original_last_act"].to("cuda")[cntxt_mask]
            # we store acts and recalculate logits to save memory
            original_logits = (model.model.embed_tokens.weight @ original_last_act.T).T
            logits = output.logits[cntxt_mask]
            original_logits = _normalize_logits(original_logits.float())
            logits = _normalize_logits(logits.float())

            # calculate KL divergence between original and current logits
            kl_div = F.kl_div(
                logits, original_logits, reduction="batchmean", log_target=True
            )
            assert kl_div > 0
            _kl_div_acc += kl_div.item()
            # this kl_div calculation is the same as:
            # (original_logits.exp() * (original_logits - logits)).sum(dim=-1).mean()

            # note that retain_graph is only possible if we don't update the weights
            kl_div.backward(retain_graph=True)

            for n, m in trainable_modules(model):
                m.kl_div_grad = get_last_grad(m, batch["attention_mask"])

            # last_act = output.hidden_states[-1][cntxt_mask]
            # diff = last_act - original_last_act
            # context_disruption = diff.norm(dim=-1).mean()

        # ! unlearning loss
        loss_fn = getattr(loss_fns, exp_cfg.loss_fn_name)
        answer_mask = batch["answer_mask"] if exp_cfg.only_train_on_answer else None
        loss = loss_fn(output, batch, answer_mask)
        loss.backward()

        # ! here we modify the grad
        if exp_cfg.algorithm == "CIR":
            for n, m in trainable_modules(model):
                acts = get_last_act(m, batch["attention_mask"])
                grads = get_last_grad(m, batch["attention_mask"])
                assert len(acts.shape) == len(grads.shape) == 2

                # ! proj out the means and PCA components
                for comp in act_to_collapse[n][: exp_cfg.act_num_proj]:
                    acts -= project_out(acts, comp)
                for comp in grad_to_collapse[n][: exp_cfg.grad_num_proj]:
                    grads -= project_out(grads, comp)

                # filter out disruptive grads
                if exp_cfg.get("kl_ratio_thresh") is not None:
                    assert m.kl_div_grad.shape == grads.shape
                    ratios = (-m.kl_div_grad / grads).nan_to_num()
                    mask = ratios > exp_cfg.kl_ratio_thresh
                    # print(
                    #     (mask & (grads != 0)).float().mean(),
                    #     ratios[(grads != 0)].mean(),
                    #     ratios[(grads != 0)].max(),
                    # )
                    grads[mask] = 0

                # without the projections, this is the equivalent of normal backprop
                m.weight.grad = pt.einsum("ti,tj->ij", grads, acts)
                assert m.weight.grad.shape == m.weight.shape

        # ! normalize grads
        if cfg.normalize:
            update_norm = get_update_norm(model)
            if "max_norm" not in globals():
                max_norm = update_norm
                print(f"max_norm: {max_norm:7.2f}")
            # print(f"update_norm: {update_norm:7.2f}")
            # normalize only if the update is larger than the initial update times X
            if update_norm > max_norm * 1:
                for n, m in trainable_modules(model):
                    m.weight.grad *= max_norm / update_norm

        optimizer.step()

        if "retaining_rate" in exp_cfg:
            retaining_batch = retain_batches[i % len(retain_batches)]
            model.zero_grad(set_to_none=True)
            output = model(**retaining_batch)
            loss = cross_entropy(output, retaining_batch)
            loss.backward()
            retain_optimizer.step()

    # ! get metrics
    res = get_metrics(model)
    wandb.log(res | {"kl_div": _kl_div_acc / len(training_batches)})
    if res["wikitext_loss"] > initial_res["wikitext_loss"] * 1.01:
        break

wandb.finish()
print(f"time taken: {time.time() - start_time:.2f}s")

# # inspect per question acc
# for ex in eval_qs:
#     acc = eval_on(Dataset.from_list([ex]), model, temperature=1)
#     print(acc)


# %% retraining on T

if "retraining_epochs" not in cfg:
    exit(0)


wandb.init(
    project="ret_" + project_name,
    group=group,
    name=run_name,
    config=OmegaConf.to_container(exp_cfg),
)
optimizer = pt.optim.SGD(model.parameters(), lr=cfg.retraining_rate)

for epoch in range(cfg.retraining_epochs):
    model.train()
    for batch in retraining_batches:
        model.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = cross_entropy(output, batch)
        loss.backward()
        optimizer.step()

    # ! get metrics
    res = get_metrics(model)
    wandb.log(res)

wandb.finish()
