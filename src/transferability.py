# %%
%load_ext autoreload
%autoreload 2
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
wmdp_mcq = wmdp_mcq.filter(lambda ex: ex["Llama-3.2-1B"] > 0.7)  # 2 questions
# load disrution eval set
_fineweb_batches = load_batches(load_fineweb_edu_corpus(), conf.model_id, batch_size=16)
disr_batch = _fineweb_batches[0]

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


# %%

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
q_index = 0
q = wmdp_mcq[q_index]
f_corpus = paraphrases_all.filter(lambda ex: ex["original_question"] == q["question"])

# %%
beginning_text = f"{q['question']}\nAnswer:"
beginning_batch = tokenizer(beginning_text, **conf.tokenizer)

answer = q["choices"][q["answer"]]
full_batch = tokenizer(f"{beginning_text} {answer}", **conf.tokenizer)

loss_mask = prepare_answer_mask(beginning_batch, full_batch)
grad0 = get_grad(model, full_batch, loss_mask)

# %%

ex = f_corpus[0]
beginning_batch = tokenizer(ex["beginning"], **conf.tokenizer)
full_batch = tokenizer(f"{ex['beginning']} {ex['ending']}", **conf.tokenizer)
loss_mask = prepare_answer_mask(beginning_batch, full_batch)
grad_acc = get_grad(model, full_batch, loss_mask)
sign_acc = grad_acc.sign().to(pt.int8)
sign_counter = 1

# %%

ex = f_corpus[29]
beginning_batch = tokenizer(ex["beginning"], **conf.tokenizer)
full_batch = tokenizer(f"{ex['beginning']} {ex['ending']}", **conf.tokenizer)
loss_mask = prepare_answer_mask(beginning_batch, full_batch)
grad_acc += get_grad(model, full_batch, loss_mask)
sign_acc += grad_acc.sign().to(pt.int8)
sign_counter += 1

n_to_corr = {}
corr_acc = 0
count = 0
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    g0 = grad0[n]
    g1 = grad_acc[n]

    signs = sign_acc[n]
    mask_out = signs.abs() < sign_counter
    g1[mask_out] = 0

    corr = pt.corrcoef(pt.stack([g0.flatten(), g1.flatten()]))[0, 1]
    n_to_corr[n] = float(corr)
    corr_acc += corr * p.numel()
    count += p.numel()
    print(f"{n}: {corr:.2f}")
mean_corr = corr_acc / count
print(f"mean corr: {mean_corr:.2f}")

visualize_module_values(n_to_corr, f"paraphrases\ncorr={mean_corr:.2f}")

# %%
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    signs = sign_acc[n].abs()
    # plot histogram of signs
    plt.hist(signs.flatten().tolist(), bins=range(31))
    plt.show()
