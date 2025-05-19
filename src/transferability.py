# %%
# %load_ext autoreload
# %autoreload 2
import json
import logging
import os
import random
from copy import deepcopy

import hydra
import matplotlib.pyplot as plt
import torch as pt
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from utils import loss_fns, masking
from utils.data_loading import load_batches, load_fineweb_edu_corpus, load_local
from utils.evals import format_prompt
from utils.evals import eval_on
from utils.git_and_reproducibility import repo_root
from utils.loss_fns import print_per_token_colored_loss
from utils.plots import visualize_module_values
from utils.training import set_seeds

logging.basicConfig(level=logging.INFO)
pt.set_default_device("cuda")
conf = OmegaConf.load("../configs/transferability.yaml")

# load corpora
paraphrases_all = load_local("my_generation/wmdp_bio.jsonl")
# load questions
wmdp_mcq = load_local(f"wmdp_deduped_{conf.category}/{conf.split}.jsonl")
wmdp_mcq = wmdp_mcq.filter(lambda ex: ex["Llama-3.2-1B"] > 0.5)  # 10 questions
# load disrution eval set
_fineweb_batches = load_batches(load_fineweb_edu_corpus(), conf.model_id, batch_size=16)
disr_batch = _fineweb_batches[0]

# %%

conf = OmegaConf.load("../configs/transferability.yaml")
# ! setup
set_seeds(42)
tokenizer = AutoTokenizer.from_pretrained(conf.model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)
model.config.use_cache = False
# ! limit which parameters are trained
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in conf.target_modules)


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def get_grad(model, batch, loss_mask=None):
    model.zero_grad(set_to_none=True)
    output = model(**batch)
    if loss_mask is not None:
        batch["attention_mask"] *= loss_mask
    loss = loss_fns.cross_entropy(output, batch)
    loss.backward()
    grad = TensorDict(
        {n: p.grad for n, p in model.named_parameters() if p.requires_grad},
    )
    model.zero_grad(set_to_none=True)
    return grad


def prepare_answer_mask(beginning_batch, full_batch):
    long_attn = full_batch["attention_mask"]
    short_attn = beginning_batch["attention_mask"]
    pad_amount = long_attn.shape[1] - short_attn.shape[1]
    short_attn_padded = F.pad(short_attn, (0, pad_amount), value=0)
    answer_mask = (long_attn != short_attn_padded).to(pt.int64)
    return answer_mask


# %%
control_grad = TensorDict(
    {n: pt.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
)
# for q_index in [5, 6, 7, 8, 9]:
for q_index in [0, 1, 2, 3, 4]:
    q = wmdp_mcq[q_index]
    f_corpus = paraphrases_all.filter(lambda ex: ex["original_question"] == q["question"])
    for _ in range(4):
        beginning_text = format_prompt(q)
        beginning_batch = tokenizer(beginning_text, **conf.tokenizer)

        answer = ["A", "B", "C", "D"][q["answer"]]
        full_text = f"{beginning_text} {answer}"
        full_batch = tokenizer(full_text, **conf.tokenizer)

        loss_mask = prepare_answer_mask(beginning_batch, full_batch)
        control_grad += get_grad(model, full_batch, loss_mask)

        # rotate the possible answers
        _tmp = q["choices"].pop(0)
        q["choices"].append(_tmp)
        q["answer"] = (q["answer"] - 1) % len(q["choices"])
        q["answer"], q["choices"]


# %%
target_grad = TensorDict(
    {n: pt.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
)
q_index = 8
q = wmdp_mcq[q_index]
f_corpus = paraphrases_all.filter(lambda ex: ex["original_question"] == q["question"])
for _ in range(4):
    beginning_text = format_prompt(q)
    beginning_batch = tokenizer(beginning_text, **conf.tokenizer)

    answer = ["A", "B", "C", "D"][q["answer"]]
    full_text = f"{beginning_text} {answer}"
    full_batch = tokenizer(full_text, **conf.tokenizer)

    loss_mask = prepare_answer_mask(beginning_batch, full_batch)
    target_grad += get_grad(model, full_batch, loss_mask)

    # rotate the possible answers
    _tmp = q["choices"].pop(0)
    q["choices"].append(_tmp)
    q["answer"] = (q["answer"] - 1) % len(q["choices"])
    q["answer"], q["choices"]

q["question"] + " " + q["choices"][q["answer"]]

# %%
"T-2 Mycotoxin can be deactivated by treatment with Sodium hypochlorite"

interven_grad = TensorDict(
    {n: pt.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
)
for ex in f_corpus:
# ex = f_corpus[29]
# print(ex["beginning"] + " " + ex["ending"])
    beginning_batch = tokenizer(ex["beginning"], **conf.tokenizer)
    full_batch = tokenizer(f"{ex['beginning']} {ex['ending']}", **conf.tokenizer)
    loss_mask = prepare_answer_mask(beginning_batch, full_batch)
    interven_grad += get_grad(model, full_batch, loss_mask)

n_to_corr = {}
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    control = control_grad[n]
    target = target_grad[n]
    interven = interven_grad[n]

    red = max((interven * control).sum(), 0)
    green = max((interven * target).sum(), 0)

    color = pt.Tensor([red, green, 0])
    # intensity = float(g0.norm() * g1.norm())
    n_to_corr[n] = color
    # print(f"{n}: {color}")

# normalize
max_norm = max(max(color) for color in n_to_corr.values())
n_to_corr = {n: v / max_norm for n, v in n_to_corr.items()}

visualize_module_values(n_to_corr, "")

# %%
# augment by rotations
target_mcq = []
for _ in range(4):
    target_mcq.append(q)
    # rotate the possible answers
    _tmp = q["choices"].pop(0)
    q["choices"].append(_tmp)
    q["answer"] = (q["answer"] - 1) % len(q["choices"])
    q["answer"], q["choices"]
target_mcq = Dataset.from_list(target_mcq)

control_mcq = []
for q_index in [0, 1, 2, 3, 4]:
    q_tmp = wmdp_mcq[q_index]
    for _ in range(4):
        control_mcq.append(q_tmp)
        # rotate the possible answers
        _tmp = q_tmp["choices"].pop(0)
        q_tmp["choices"].append(_tmp)
        q_tmp["answer"] = (q_tmp["answer"] - 1) % len(q_tmp["choices"])
        q_tmp["answer"], q_tmp["choices"]
control_mcq = Dataset.from_list(control_mcq)

# %%
lr = 0.0002
xs = []
ys = []
del model
model = AutoModelForCausalLM.from_pretrained(
    conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in conf.target_modules)

# %%

target_acc = eval_on(target_mcq, model, temperature=1)
control_acc = eval_on(control_mcq, model, temperature=1)
xs.append(target_acc)
ys.append(control_acc)
print(f"target acc: {target_acc}, control acc: {control_acc}")

for n, p in model.named_parameters():
    if not any(pattern in n for pattern in ["o_proj", "gate_proj", "up_proj"]):
        continue
    if not any(pattern in n for pattern in [".0.", ".1.", ".2.", ".3.", ".4."]):
        continue
    p.data += interven_grad[n] * lr


# %%
# xs2, ys2 = xs, ys
plt.plot(xs, ys)
plt.plot(xs2, ys2)

# %%
