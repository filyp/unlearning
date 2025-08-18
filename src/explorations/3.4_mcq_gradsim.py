# %%
# adapted from explorations/2_proj_training.py which also has some code for grad_proj,
#     dm grad and dm act, and the traditional dm,
#     mmlu evals, and per-token loss increase visualizations,
#     and context undisruption (a more fancy retaining technique)
# but here, we aim for more simlicity and dataset generality

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
from omegaconf import DictConfig, OmegaConf, open_dict
from tensordict import TensorDict
from transformers import AutoTokenizer

import wandb
from utils import loss_fns
from utils.common_cir import *
from utils.data_loading import *
from utils.evals import eval_on, format_prompt
from utils.git_and_reproducibility import get_conf_hash, repo_root
from utils.loss_fns import cross_entropy, kl_loss
from utils.plots import print_colored_tokens
from utils.training import (
    get_grad,
    get_update_norm,
    scale_grads_,
    set_seeds,
    trainable_modules,
)

# plt dark theme
plt.style.use("dark_background")

logging.basicConfig(level=logging.INFO)

# Parse just the config-name, let Hydra handle the rest
parser = argparse.ArgumentParser()
parser.add_argument("--config-name", default="cir")
args, remaining_args = parser.parse_known_args()
cfg = OmegaConf.load("../../configs/deeb_bounce.yaml")  # for debugging
with open_dict(cfg):
    cfg = OmegaConf.merge(cfg, cfg.experiment_list[cfg.experiment_number])
cfg.model_id = "meta-llama/Llama-3.2-1B"

# ! setup
set_seeds(42)

num_gpus = pt.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
pt.set_default_device("cuda")
device_main = pt.device("cuda")
device_storage = pt.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = tokenizer.eos_token

# ! load wikitext batches
wikitext = load_local(repo_root() / "data" / "wikitext_16k.jsonl")
_txts = wikitext.shuffle(seed=42).batch(cfg.batch_size)
wikitext_batches = [
    tokenizer(x["text"], **cfg.tokenizer) for x in _txts.select(range(32))
]
wikitext_control_batches = [
    tokenizer(x["text"], **cfg.tokenizer) for x in _txts.select(range(32, 32 + 64))
]

# %%

T = load_local(f"wmdp_deduped_cyber/dev_T_corpus.jsonl")
V = load_local(f"wmdp_deduped_cyber/dev_V_corpus.jsonl")
# T = load_local(f"wmdp_deduped_bio/dev_T_corpus.jsonl")
# V = load_local(f"wmdp_deduped_bio/dev_V_corpus.jsonl")
T = T.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
V = V.filter(lambda x: x["Llama-3.1-8B"] > 0.25)
print(f"{len(T)=}, {len(V)=}")
T_and_V = concatenate_datasets([T, V])

deebs_corpus = load_local("wmdp_deduped_deebs_corpus.jsonl")


# %%
def get_grad_from_example(model, beginning, ending, loss_fn_name="cross_entropy"):
    beginning_batch = tokenizer(beginning, **cfg.tokenizer)
    full_batch = tokenizer(f"{beginning} {ending}", **cfg.tokenizer)
    loss_mask = prepare_answer_mask(beginning_batch, full_batch)
    return get_grad(model, full_batch, loss_mask, loss_fn_name)


def get_grad_from_abcd_question(model, question, loss_fn_name="cross_entropy"):
    beginning = format_prompt(question)
    ending = ["A", "B", "C", "D"][question["answer"]]
    return get_grad_from_example(model, beginning, ending, loss_fn_name)


def get_grads_from_text(model, text):
    model.zero_grad(set_to_none=True)
    batch = tokenizer(text, **cfg.tokenizer)
    output = model(**batch)
    loss = loss_fns.cross_entropy(output, batch)
    loss.backward()
    return TensorDict(
        {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    )


def tensor_dict_dot_product(a, b):
    acc = 0
    for k in a.keys():
        acc += (a[k].to(pt.float32) * b[k].to(pt.float32)).sum()
    return acc


def tensor_dict_cossim(a, b):
    a_dot_b = tensor_dict_dot_product(a, b)
    a_norm = tensor_dict_dot_product(a, a).sqrt()
    b_norm = tensor_dict_dot_product(b, b).sqrt()
    return a_dot_b / (a_norm * b_norm)


# %% setup

# * load model
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id, torch_dtype=pt.bfloat16, device_map=device_main
)
model.config.use_cache = False

# * set trainable params
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in cfg.target_modules)

install_hooks(model)

# %%
question_to_filered = {
    q["question"]: deebs_corpus.filter(
        lambda x: x["original_question"] == q["question"]
    )
    for q in T
}
# %%
for question in T:
    if question["Llama-3.2-1B"] < 0.4:
        continue

    mcq_grad = get_grad_from_abcd_question(model, question)
    # txts = deebs_corpus.filter(lambda x: x["original_question"] == question["question"])
    txts = question_to_filered[question["question"]]

    print()
    print()
    print("question:", question["question"])
    print("answer:", question["choices"][question["answer"]])
    print()
    for id_ in range(len(question["contexts"])):
        pair_grad = get_grad_from_example(
            model, question["contexts"][id_], question["answer_core"]
        )
        cossim = tensor_dict_cossim(mcq_grad, pair_grad)
        print(
            round(cossim.item(), 4), question["contexts"][id_], question["answer_core"]
        )

    for id_ in range(3):
        deeb_grad = get_grads_from_text(model, txts[id_]["text"])
        cossim = tensor_dict_cossim(mcq_grad, deeb_grad)
        print(round(cossim.item(), 4), txts[id_]["text"])
# %%

new_texts = [
    "If you need to recover a hashed password, the appropriate tool to use is Hashcat.",
    "To retrieve a hashed password, the recommended tool to utilize is Hashcat.",
    "For retrieving passwords from hashes, use Hashcat.",
    "The appropriate utility for recovering hashed passwords is Hashcat.",
    "If you want to retrieve a hashed password, the tool to use is Hashcat.",
    "To recover a hashed password, the suitable tool to employ is Hashcat.",
    "The optimal tool for recovering hashed passwords is Hashcat.",
    "For hashed password recovery, the appropriate tool is Hashcat.",
    "To retrieve passwords from their hashes, utilize Hashcat.",
    "The proper tool for recovering a hashed password is Hashcat.",
]

for id_ in range(len(new_texts)):
    deeb_grad = get_grads_from_text(model, new_texts[id_])
    cossim = tensor_dict_cossim(mcq_grad, deeb_grad)
    print(round(cossim.item(), 4), new_texts[id_])
# %%
