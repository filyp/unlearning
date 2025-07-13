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
from datasets import concatenate_datasets, load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

import wandb
from utils import loss_fns
from utils.data_loading import load_fineweb_bio_corpus, load_local
from utils.evals import eval_on
from utils.common_cir import *
from utils.plots import print_colored_tokens
from utils.training import PCA_gpu, prepare_answer_mask, set_seeds, trainable_modules
from utils.git_and_reproducibility import repo_root

# plt dark theme
plt.style.use("dark_background")

logging.basicConfig(level=logging.INFO)
conf = OmegaConf.load(repo_root() / "configs/transferability.yaml")
conf.model_id = "meta-llama/Llama-3.2-3B"
conf.target_modules = ["gate_proj", "up_proj", "down_proj"]
conf.device = "cuda" if pt.cuda.is_available() else "cpu"

# ! setup
set_seeds(42)
pt.set_default_device(conf.device)
tokenizer = AutoTokenizer.from_pretrained(conf.model_id)
tokenizer.pad_token = tokenizer.eos_token

wmdp_T = load_local(f"wmdp_deduped_bio/dev_T_corpus.jsonl")
wmdp_V = load_local(f"wmdp_deduped_bio/dev_V_corpus.jsonl")
wmdp_joined = concatenate_datasets([wmdp_T, wmdp_V])
fineweb_bio = load_fineweb_bio_corpus()
mmlu_bio = load_local("OUTDATED/my_generation2/mmlu_high_school_biology.jsonl")
wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
# filter out empty texts from wikitext
wikitext = wikitext.filter(lambda x: x["text"])


def get_batches(dataset, range_, batch_size=16, with_answer_mask=True):
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
        if with_answer_mask:
            yield full_batch, answer_mask
        else:
            yield full_batch


def get_loss(model, batch, answer_mask=None, loss_fn_name="cross_entropy"):
    model.zero_grad(set_to_none=True)
    output = model(**batch, output_hidden_states=True)
    loss_fn = getattr(loss_fns, loss_fn_name)
    return loss_fn(output, batch, answer_mask)



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


# %%
if "model" in globals():
    # cleanup
    del model, acts_list, grads_list, act_means, grad_means, act_pca_components, inspect_batches, training_batches, control_batches_gens
    pt.cuda.empty_cache()
model = prepare_model(conf, use_every_n_layers=4)

run_conf = SimpleNamespace(
    # lr=0.001,
    lr=0.01,
    retain_lr=0.0003,
    normalize=False,
    only_train_on_answer=True,
    # loss_fn_name="neg_cross_entropy",
    loss_fn_name="correct_logit",
    num_pc=10,
    # techniques=[],
    # techniques=["act"],
    # techniques=["dm_agg"],
    # techniques=["act", "grad", "pca"],
    # techniques=["dm_act", "dm_grad"],
    techniques=["act", "pca", "training_ctrl", "mmlu_ctrl"],
    # techniques=["act", "grad", "pca", "training_ctrl", "mmlu_ctrl"],
    # techniques=["act", "pca", "mmlu_ctrl"],
)

# ! record mmlu batches and their per-token losses, to later see how they are disrupted
inspect_batches = []
for text in itertools.chain(
    # [f"{q['contexts'][0]} {q['answer_core']}" for q in wmdp_V.select(range(5))],
    [f"{q['contexts'][0]} {q['answer_core']}" for q in mmlu_bio.select(range(5))],
    [ex["text"] for ex in fineweb_bio.select(range(5))],
    [ex["text"] for ex in wikitext.select([0, 10, 20, 30, 40])],
):
    batch = tokenizer(text, **conf.tokenizer)
    model.zero_grad(set_to_none=True)
    output = model(**batch, output_hidden_states=True)
    token_losses = loss_fns.cross_entropy_per_token(output, batch).detach()
    inspect_batches.append((batch, token_losses))

# ! define training batches, caching the last layer activations
training_batches = []
for batch, answer_mask in get_batches(wmdp_joined, range(7)):
    batch["answer_mask"] = answer_mask if run_conf.only_train_on_answer else None
    with pt.no_grad():
        output = model(**batch, output_hidden_states=True)
    # watch out, it can be 600MB for dev set
    batch["original_last_act"] = output.hidden_states[-1].detach().to("cpu")
    training_batches.append(batch)

# ! first pass through forget corpus, to collect acts and grads
acts_list = {n: [] for n, _ in trainable_modules(model)}
grads_list = {n: [] for n, _ in trainable_modules(model)}
# define control batches
control_batches_gens = []
if "training_ctrl" in run_conf.techniques:
    control_batches_gens.append(training_batches)
if "mmlu_ctrl" in run_conf.techniques:
    control_batches_gens.append(
        get_batches(mmlu_bio.select(range(24, 96)), range(1), with_answer_mask=False)
    )
if "wikitext_ctrl" in run_conf.techniques:
    control_batches_gens.append(
        (tokenizer(ex["text"], **conf.tokenizer) for ex in wikitext.select(range(32)))
    )
# gather acts and grads
for batch in itertools.chain(*control_batches_gens):
    _mask = batch.get("answer_mask", None)
    loss = get_loss(model, batch, _mask, run_conf.loss_fn_name)
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
    act_pca_components[n] = PCA_gpu(acts_flattened, n_components=run_conf.num_pc)

del model
pt.cuda.empty_cache()
model = prepare_model(conf, use_every_n_layers=4)
optimizer = pt.optim.SGD(model.parameters(), lr=run_conf.lr)
retain_optimizer = pt.optim.SGD(model.parameters(), lr=run_conf.retain_lr)

run_name = (
    f"{run_conf.loss_fn_name} lr{run_conf.lr} rlr{run_conf.retain_lr} pc{run_conf.num_pc} "
    + " ".join(run_conf.techniques)
)
wandb.init(
    project="unlearning-wmdp4",
    name=run_name,
    group=f"tv-all_module_types2",
    config=OmegaConf.to_container(conf),
)

# % full training loop
start_time = time.time()
for epoch in range(10):
    pt.cuda.empty_cache()

    # ! get metrics
    res = get_metrics(model)
    wandb.log(res)
    # if res["fineweb_loss"] > 18.5 or res["mmlu_loss"] > 18.5:
    # break

    # ! one epoch
    model.train()
    _norms = []
    for batch in training_batches:

        if "context_undisrupt" in run_conf.techniques:
            assert run_conf.only_train_on_answer
            # ! compute context disruption
            output = model(**batch, output_hidden_states=True)
            model.zero_grad(set_to_none=True)
            cntxt_mask = batch["attention_mask"].bool() & (~batch["answer_mask"].bool())
            last_activations = output.hidden_states[-1][cntxt_mask]
            current_activations = batch["original_last_act"].to("cuda")[cntxt_mask]
            diff = last_activations - current_activations
            context_disruption = diff.norm(dim=-1).mean()
            # context_disruption.backward(retain_graph=True)
            # the problem with retaining the graph is that the update changes the weights! wd need to do some complications to enable reusing forward pass, but it's possible
            context_disruption.backward()
            retain_optimizer.step()

        # ! unlearning loss
        model.zero_grad(set_to_none=True)
        output = model(**batch, output_hidden_states=True)
        loss_fn = getattr(loss_fns, run_conf.loss_fn_name)
        loss = loss_fn(output, batch, batch.get("answer_mask", None))
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


# %% print the loss diffs for various batches
batch_and_loss_diffs = []
for batch, init_token_losses in inspect_batches:
    model.zero_grad(set_to_none=True)
    with pt.no_grad():
        output = model(**batch, output_hidden_states=True)
    token_losses = loss_fns.cross_entropy_per_token(output, batch).detach()
    loss_diffs = token_losses - init_token_losses
    batch_and_loss_diffs.append((batch, loss_diffs))
# get the max loss diff
max_loss_diff = max([loss_diffs.abs().max() for _, loss_diffs in batch_and_loss_diffs])
print(f"{max_loss_diff=}")
for batch, loss_diffs in batch_and_loss_diffs:
    loss_diffs /= max_loss_diff
    print_colored_tokens(loss_diffs, batch, tokenizer)
del batch_and_loss_diffs, max_loss_diff

# for q in wmdp_joined:
# print(q["contexts"][0], q["answer_core"])

# %% retraining on T
wandb.init(
    project="unlearning-wmdp4-retrain",
    name=run_name,
    group=f"tv-all_module_types",
    config=OmegaConf.to_container(conf),
)
optimizer = pt.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(30):
    # ! get metrics
    res = get_metrics(model)
    wandb.log(res)

    model.train()
    for batch, _ in get_batches(wmdp_T, range(7)):
        model.zero_grad(set_to_none=True)
        loss = get_loss(model, batch)
        loss.backward()
        optimizer.step()

wandb.finish()

# %%
len(training_batches)
len(wmdp_joined)