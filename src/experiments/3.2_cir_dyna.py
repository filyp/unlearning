# %%
# adapted from explorations/2_proj_training.py which also has some code for grad_proj,
#     dm grad and dm act, and the traditional dm,
#     mmlu evals, and per-token loss increase visualizations,
#     and context undisruption (a more fancy retaining technique)
# but here, we aim for more simlicity and dataset generality

# todo implement a *class* for collecting acts, projs, calculating pca, and storing them!
# maybe just recalculate projs once every n epochs?
#     also, cleaner logic, if we first do one dummy epoch, gathering, and only then the main loop starts
#     every n epochs, would enable using a different cotnrol set, which is needed for tendency unlearning

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch as pt
from datasets import Dataset, concatenate_datasets, load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

import wandb
from utils import loss_fns
from utils.common_cir import *
from utils.data_loading import *
from utils.evals import eval_on
from utils.git_and_reproducibility import repo_root
from utils.loss_fns import cross_entropy
from utils.training import get_update_norm, set_seeds, trainable_modules

# plt dark theme
plt.style.use("dark_background")

logging.basicConfig(level=logging.INFO)
conf = OmegaConf.load(repo_root() / "configs/2_cir.yaml")
rconf = conf.experiment_list[conf.experiment_number]

# ! setup
set_seeds(42)
pt.set_default_device("cuda")
tokenizer = AutoTokenizer.from_pretrained(conf.model_id)
tokenizer.pad_token = tokenizer.eos_token

# ! load wikitext batches
wikitext = load_local(repo_root() / "data" / "wikitext_16k.jsonl")
_txts = wikitext.shuffle(seed=42).batch(conf.batch_size)
wikitext_batches = [
    tokenizer(x["text"], **conf.tokenizer) for x in _txts.select(range(16))
]

# ! load proper datasets
if conf.dataset == "wmdp_bio":
    T = load_local(f"wmdp_deduped_bio/T_corpus.jsonl")
    V = load_local(f"wmdp_deduped_bio/V_corpus.jsonl")
    T = T.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
    V = V.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
    print(f"{len(T)=}, {len(V)=}")
    # todo, limit again to 40, reenable and test
    # T = T.filter(lambda x: len(x["answer_core"]) <= 40)
    # V = V.filter(lambda x: len(x["answer_core"]) <= 40)
    T_and_V = concatenate_datasets([T, V])

    training_batches = load_batches_from_pairs_set(T_and_V, conf, range(0, 3))
    retraining_batches = load_batches_from_pairs_set(T, conf, range(0, 3))
    loss_eval_batches = load_batches_from_pairs_set(V, conf, range(7, 10))
    # todo optionally we could try retain set instead
    control_batches = training_batches

    retain_set = load_fineweb_bio_corpus()
    _txts = retain_set.shuffle(seed=42).batch(conf.batch_size)
    retain_batches = [
        tokenizer(x["text"], **conf.tokenizer)
        for x in _txts.select(range(len(training_batches)))
    ]
elif conf.dataset == "wmdp_cyber":
    T = load_local(f"wmdp_deduped_cyber/T_corpus.jsonl")
    V = load_local(f"wmdp_deduped_cyber/V_corpus.jsonl")
    T = T.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
    V = V.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
    print(f"{len(T)=}, {len(V)=}")
    T_and_V = concatenate_datasets([T, V])
    # retain_set = load_fineweb_cyber_corpus()
    raise NotImplementedError("Cyber dataset not implemented yet")

elif conf.dataset == "wmdp_bio_deebs":
    T = load_local(f"wmdp_deduped_bio/T_corpus.jsonl")
    V = load_local(f"wmdp_deduped_bio/V_corpus.jsonl")
    T = T.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
    V = V.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
    print(f"{len(T)=}, {len(V)=}")
    deebs_corpus = load_local("wmdp_deduped_deebs_corpus.jsonl")

    t_questions = set(T["question"])
    t_texts = deebs_corpus.filter(lambda x: x["original_question"] in t_questions)
    len(t_texts)

elif conf.dataset == "wmdp_cyber_deebs":
    raise NotImplementedError("Cyber dataset not implemented yet")

elif conf.dataset == "jigsaw_threats":
    jigsaw = load_jigsaw_dataset()
    jigsaw_threats = jigsaw[jigsaw["threat"] == 1]
    jigsaw_benign = jigsaw[jigsaw["toxic"] == 0]
    # todo split jigsaw into T and V
    # here splitting into T and V makes less sense than with independent facts
    # but it's still nice to do
    raise NotImplementedError("Jigsaw dataset not implemented yet")
    retain_set = jigsaw_benign  # format batches properly

# %%
# %%

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
    if conf.dataset == "wmdp_bio" or conf.dataset == "wmdp_cyber":
        res["forget_acc"] = eval_on(V, model, temperature=1)

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
    conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)
model.config.use_cache = False

# * set trainable params
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in conf.target_modules)

install_hooks(model)

optimizer = pt.optim.SGD(model.parameters(), lr=rconf.unlearning_rate)
if "retaining_rate" in rconf:
    retain_optimizer = pt.optim.SGD(model.parameters(), lr=rconf.retaining_rate)

# inspect per question acc
for ex in V:
    acc = eval_on(Dataset.from_list([ex]), model, temperature=1)
    print(acc)

# %%
project_name = "unlearning/" + Path(__file__).relative_to(repo_root()).as_posix()
project_name = project_name.replace("/", "|")
run_name = "_".join(str(v) for v in rconf.values())
wandb.init(
    project=project_name,
    name=run_name,
    group=conf.dataset + "_" + conf.model_id,
    config=OmegaConf.to_container(rconf),
)

initial_res = get_metrics(model)
wandb.log(initial_res)
assert rconf.algorithm in ["CIR", "GA"]

# % full training loop
start_time = time.time()
for epoch in range(conf.max_num_epochs):
    pt.cuda.empty_cache()

    # ! recalculate projections
    if epoch % rconf.recalc_every_n_epochs == 0 and rconf.algorithm == "CIR":
        acts_list = {n: [] for n, _ in trainable_modules(model)}
        grads_list = {n: [] for n, _ in trainable_modules(model)}

        for i, batch in enumerate(control_batches):
            # ! unlearning loss
            model.zero_grad(set_to_none=True)
            output = model(**batch)
            loss_fn = getattr(loss_fns, rconf.loss_fn_name)
            answer_mask = batch["answer_mask"] if rconf.only_train_on_answer else None
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
    for i, batch in enumerate(training_batches):
        # ! unlearning loss
        model.zero_grad(set_to_none=True)
        output = model(**batch)
        loss_fn = getattr(loss_fns, rconf.loss_fn_name)
        answer_mask = batch["answer_mask"] if rconf.only_train_on_answer else None
        loss = loss_fn(output, batch, answer_mask)
        loss.backward()

        # ! here we modify the grad
        if rconf.algorithm == "CIR":
            for n, m in trainable_modules(model):
                acts = get_last_act(m, batch["attention_mask"])
                grads = get_last_grad(m, batch["attention_mask"])
                assert len(acts.shape) == len(grads.shape) == 2

                # ! proj out the means and PCA components
                for comp in act_to_collapse[n][: rconf.act_num_proj]:
                    acts -= project_out(acts, comp)
                for comp in grad_to_collapse[n][: rconf.grad_num_proj]:
                    grads -= project_out(grads, comp)

                # without the projections, this is the equivalent of normal backprop
                m.weight.grad = pt.einsum("ti,tj->ij", grads, acts)
                assert m.weight.grad.shape == m.weight.shape

        # ! normalize grads
        if conf.normalize:
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

        if "retaining_rate" in rconf:
            retaining_batch = retain_batches[i % len(retain_batches)]
            model.zero_grad(set_to_none=True)
            output = model(**retaining_batch)
            loss = cross_entropy(output, retaining_batch)
            loss.backward()
            retain_optimizer.step()

    # ! get metrics
    res = get_metrics(model)
    wandb.log(res)
    if res["wikitext_loss"] > initial_res["wikitext_loss"] * 1.01:
        break

wandb.finish()
print(f"time taken: {time.time() - start_time:.2f}s")

# inspect per question acc
for ex in V:
    acc = eval_on(Dataset.from_list([ex]), model, temperature=1)
    print(acc)


# %% retraining on T

exit(0)


wandb.init(
    project="ret_" + project_name,
    name=run_name,
    group=conf.dataset + "_" + conf.model_id,
    config=OmegaConf.to_container(rconf),
)
optimizer = pt.optim.SGD(model.parameters(), lr=conf.retraining_rate)

for epoch in range(conf.retraining_epochs):
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
wandb.finish()
