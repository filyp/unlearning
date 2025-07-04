# %%
# %load_ext autoreload
# %autoreload 2
import logging
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from utils import loss_fns
from utils.common_cir import *
from utils.data_loading import load_local
from utils.git_and_reproducibility import repo_root
from utils.plots import visualize_rgb
from utils.training import (
    get_grads_dict,
    prepare_answer_mask,
    set_seeds,
    trainable_modules,
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

# deception_set = load_local("machiavelli/deception/psy-high.jsonl")
wmdp = load_local(f"wmdp_deduped_bio/dev_T_corpus.jsonl")


def get_loss(model, beginning_txt, full_txt, loss_fn, only_ans=True):
    # note: assumes tokenizer is in scope
    # only grads on the answer

    beginning_batch = tokenizer(beginning_txt, **conf.tokenizer)
    batch = tokenizer(full_txt, **conf.tokenizer)
    answer_mask = prepare_answer_mask(beginning_batch, batch)
    if not only_ans:
        answer_mask = None

    output = model(**batch, output_hidden_states=True)
    return loss_fn(output, batch, answer_mask)


# %% settings
run_conf = SimpleNamespace(
    loss_fn_name="correct_logit",
    num_pc=8,
)
loss_fn = getattr(loss_fns, run_conf.loss_fn_name)

# %% get the disruption grads
model.zero_grad(set_to_none=True)
for id_ in range(6):
    q_alt = wmdp[id_]
    beginning_txt = q_alt["contexts"][0]
    full_txt = beginning_txt + " " + q_alt["answer_core"]

    loss = get_loss(model, beginning_txt, full_txt, loss_fn, only_ans=False)
    loss.backward()

disr_grads = get_grads_dict(model)

# %% get control for the without-feature set
without_act_means, without_act_pca_components = get_act_principal_components(
    model,
    [
        tokenizer(f"{c} {qc['answer_core']}", **conf.tokenizer)
        for qc in wmdp.select(range(6, 24))
        for c in qc["contexts"]
    ],
    num_pc=run_conf.num_pc,
)

# %% this cell groups operations which need to be rerun when changing the forget id

forget_id = 16
assert forget_id >= 6, "first six are disr evals"
q = wmdp[forget_id]

# ! get the target grads
model.zero_grad(set_to_none=True)
for context in q["contexts"][-3:]:
    beginning_txt = context
    full_txt = f"{beginning_txt} {q['answer_core']}"

    loss = get_loss(model, beginning_txt, full_txt, loss_fn)
    loss.backward()
target_grads = get_grads_dict(model)

# ! get control for the with-feature set
q = wmdp[forget_id]
with_act_means, with_act_pca_components = get_act_principal_components(
    model,
    [
        tokenizer(f"{c} {q['answer_core']}", **conf.tokenizer)
        for c in q["contexts"][1:-3]
    ],
    num_pc=run_conf.num_pc,
)

# # %%
# alt_sentence = "The thing most likely modified by ozone is"
# batch = tokenizer(alt_sentence, **conf.tokenizer)
# output = model(**batch, output_hidden_states=True)
# acts_list = dict()
# for n, module in trainable_modules(model):
#     acts_list[n] = get_last_act(module, batch["attention_mask"])


# %% get the forget grads
beginning_txt = q["contexts"][0]
full_txt = f"{beginning_txt} {q['answer_core']}"
batch = tokenizer(full_txt, **conf.tokenizer)

model.zero_grad(set_to_none=True)
loss = get_loss(model, beginning_txt, full_txt, loss_fn)
loss.backward()
model.zero_grad(set_to_none=True)


per_module_grads = {}
for n, module in trainable_modules(model):
    act_in = get_last_act(module, batch["attention_mask"]).float()
    grad_out = get_last_grad(module, batch["attention_mask"]).float()
 
    # ! CIR

    # act_in -= project_out(act_in, without_act_means[n])

    # for pc in without_act_pca_components[n]:
    #     act_in -= project_out(act_in, pc)
    
    # # ! alt_sentence projection
    # for proj in acts_list[n]:
    #     act_in -= project_out(act_in, proj)

    # # ! common core! (reversed?) anyway, it's terrible
    # act_in_checkpoint = act_in.clone()
    # act_in -= project_out(act_in, with_act_means[n])
    # # for pc in with_act_pca_components[n][:1]:
    #     # act_in -= project_out(act_in, pc)
    # act_in = act_in_checkpoint - act_in
 
    per_module_grads[n] = pt.einsum("ti,to->oi", act_in, grad_out)

# % visualize example x layer
# green is good transfer, red is bad transfer

_row = []
for n, _ in trainable_modules(model):
    forget_grad = per_module_grads[n]
    target_grad = target_grads[n + ".weight"]
    disr_grad = disr_grads[n + ".weight"]

    good_transfer = (target_grad * forget_grad).sum().item()
    bad_transfer = (disr_grad * forget_grad).sum().item()

    # _row.append([np.clip(bad_transfer, min=0), good_transfer, 0])  # rgb
    _row.append([np.abs(bad_transfer), good_transfer, 0])  # rgb

ratios = np.array(_row).reshape(1, -1, 3)

print("sum of disruption:", ratios[:, :, 0].sum())
print("sum of forget:", ratios[:, :, 1].sum())
print("ratio:", ratios[:, :, 0].sum() / ratios[:, :, 1].sum())
ratios2 = ratios.copy()
ratios2[:, :, 0] *= 1
visualize_rgb(ratios2, scale=940)


# %%

# %% same, but for many questions

rows = []
for forget_id in range(6, 24):
    assert forget_id >= 6, "first six are disr evals"
    q = wmdp[forget_id]

    # ! get the target grads
    model.zero_grad(set_to_none=True)
    for context in q["contexts"][-3:]:
        beginning_txt = context
        full_txt = f"{beginning_txt} {q['answer_core']}"

        loss = get_loss(model, beginning_txt, full_txt, loss_fn)
        loss.backward()
    target_grads = get_grads_dict(model)

    # ! get control for the with-feature set
    q = wmdp[forget_id]
    with_act_means, with_act_pca_components = get_act_principal_components(
        model,
        [
            tokenizer(f"{c} {q['answer_core']}", **conf.tokenizer)
            for c in q["contexts"][1:-3]
        ],
        num_pc=run_conf.num_pc,
    )

    # % get the forget grads
    beginning_txt = q["contexts"][0]
    full_txt = f"{beginning_txt} {q['answer_core']}"
    batch = tokenizer(full_txt, **conf.tokenizer)

    model.zero_grad(set_to_none=True)
    loss = get_loss(model, beginning_txt, full_txt, loss_fn)
    loss.backward()
    model.zero_grad(set_to_none=True)


    per_module_grads = {}
    for n, module in trainable_modules(model):
        act_in = get_last_act(module, batch["attention_mask"]).float()
        grad_out = get_last_grad(module, batch["attention_mask"]).float()
     
        # ! CIR

        act_in -= project_out(act_in, without_act_means[n])

        for pc in without_act_pca_components[n]:
            act_in -= project_out(act_in, pc)

        # # ! common core! (reversed?) anyway, it's terrible
        # act_in_checkpoint = act_in.clone()
        # act_in -= project_out(act_in, with_act_means[n])
        # # for pc in with_act_pca_components[n][:1]:
        #     # act_in -= project_out(act_in, pc)
        # act_in = act_in_checkpoint - act_in
     
        per_module_grads[n] = pt.einsum("ti,to->oi", act_in, grad_out)

    # % visualize example x layer
    # green is good transfer, red is bad transfer

    _row = []
    for n, _ in trainable_modules(model):
        forget_grad = per_module_grads[n]
        target_grad = target_grads[n + ".weight"]
        disr_grad = disr_grads[n + ".weight"]

        good_transfer = (target_grad * forget_grad).sum().item()
        bad_transfer = (disr_grad * forget_grad).sum().item()

        # _row.append([np.clip(bad_transfer, min=0), good_transfer, 0])  # rgb
        _row.append([np.abs(bad_transfer), good_transfer, 0])  # rgb
    
    rows.append(_row)

# %%
ratios = np.array(rows)

print("sum of disruption:", ratios[:, :, 0].sum())
print("sum of forget:", ratios[:, :, 1].sum())
print("ratio:", ratios[:, :, 0].sum() / ratios[:, :, 1].sum())
ratios2 = ratios.copy()
ratios2[:, :, 0] *= 50
visualize_rgb(ratios2, scale=6600)

# %%
ratios2[:, :, 0] *= 3
visualize_rgb(ratios2.mean(axis=0, keepdims=True), scale=2000)
# %%

visualize_rgb(ratios2.mean(axis=1, keepdims=True), scale=2000)
# %%
