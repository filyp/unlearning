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
from utils import loss_fns
from utils.data_loading import load_batches, load_fineweb_edu_corpus, load_local
from utils.evals import eval_on, format_prompt
from utils.git_and_reproducibility import repo_root
from utils.hooks import CalcSimilarityHooks
from utils.loss_fns import print_per_token_colored_loss
from utils.plots import visualize_module_values, visualize_token_layer_values
from utils.training import get_grad, prepare_answer_mask, set_seeds, trainable_params

logging.basicConfig(level=logging.INFO)
pt.set_default_device("cuda")
conf = OmegaConf.load("../configs/transferability.yaml")

# load corpora
paraphrases_all = load_local("my_generation2/wmdp_bio.jsonl")

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


def get_rotations(question):
    for _ in range(4):
        q_copy = deepcopy(question)
        # rotate the possible answers
        _tmp = q_copy["choices"].pop(0)
        q_copy["choices"].append(_tmp)
        q_copy["answer"] = (q_copy["answer"] - 1) % len(q_copy["choices"])
        yield q_copy


def get_grad_from_abcd_question(model, question):
    beginning_text = format_prompt(question)
    beginning_batch = tokenizer(beginning_text, **conf.tokenizer)
    answer = ["A", "B", "C", "D"][question["answer"]]
    full_text = f"{beginning_text} {answer}"
    full_batch = tokenizer(full_text, **conf.tokenizer)
    loss_mask = prepare_answer_mask(beginning_batch, full_batch)
    return get_grad(model, full_batch, loss_mask)


def get_grad_from_example(model, beginning, ending):
    beginning_batch = tokenizer(beginning, **conf.tokenizer)
    full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
    loss_mask = prepare_answer_mask(beginning_batch, full_batch)
    return get_grad(model, full_batch, loss_mask)


# %%
q = paraphrases_all[3]
q

# %% derive target grad
target_grad = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})

# # ! abcd
# for q_rot in get_rotations(q):
#     target_grad += get_grad_from_abcd_question(model, q_rot)

# ! russian example
for ru_context in q["ru_contexts"]:
    target_grad += get_grad_from_example(model, ru_context, q["ru_answer_core"])

# # ! normal example
# for context in q["contexts"][:5]:
#     target_grad += get_grad_from_example(model, context, q["answer_core"])


norm = pt.Tensor(list(target_grad.norm().values())).norm()
target_grad /= norm
target_grad *= 3

# %% derive control grad
control_grad = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})
# for q in [q_alt, q_alt2]:
#     for q_rot in get_rotations(q):
#         control_grad += get_grad_from_abcd_question(model, q_rot)

for answer_control in q["answer_controls"]:
    control_grad += get_grad_from_example(model, answer_control, q["answer_core"])

norm = pt.Tensor(list(control_grad.norm().values())).norm()
control_grad /= norm

# %%
pt.cuda.empty_cache()
beginning = q["contexts"][8]
ending = q["answer_core"]

with CalcSimilarityHooks(model, control_grad, target_grad):
    get_grad_from_example(model, beginning, ending)

full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
tokens = [tokenizer.decode(id_) for id_ in full_batch["input_ids"][0]]

all_control_sims = []
all_target_sims = []
all_self_sims = []
for i, l in enumerate(model.model.layers):
    module = l.mlp.up_proj
    # module = l.mlp.gate_proj
    # module = l.mlp.down_proj
    # module = l.self_attn.o_proj

    all_control_sims.append(module.weight.control_sim)
    all_target_sims.append(module.weight.target_sim)
    all_self_sims.append(module.weight.self_sim)

all_control_sims = pt.Tensor(all_control_sims)
all_target_sims = pt.Tensor(all_target_sims)
all_self_sims = pt.Tensor(all_self_sims)

flatten_pow = 1
all_control_sims = all_control_sims.clip(min=0) ** flatten_pow
all_target_sims = all_target_sims.clip(min=0) ** flatten_pow
all_self_sims = all_self_sims.clip(min=0) ** flatten_pow

max_ = max(all_control_sims.max(), all_target_sims.max())
print("max", max_.item())
all_control_sims /= max_
all_target_sims /= max_

visualize_token_layer_values(all_control_sims, all_target_sims, tokens, "")

# all_self_sims /= all_self_sims.max()
# visualize_token_layer_values(all_control_sims, all_self_sims, tokens, "")

# %%
q["answer_controls"]