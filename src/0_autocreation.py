# %%
# %load_ext autoreload
# %autoreload 2
import json
import logging
import os
import random
from copy import deepcopy

import hydra
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from utils import loss_fns
from utils.data_loading import load_batches, load_fineweb_bio_corpus, load_local
from utils.evals import eval_on, format_prompt
from utils.git_and_reproducibility import repo_root
from utils.hooks import CalcSimilarityHooks
from utils.loss_fns import print_per_token_colored_loss
from utils.plots import visualize_module_values, visualize_token_layer_values
from utils.training import get_grad, get_grad_from_pair, prepare_answer_mask, set_seeds, trainable_params

# plt dark theme
plt.style.use("dark_background")

logging.basicConfig(level=logging.INFO)
pt.set_default_device("cuda")
conf = OmegaConf.load("../configs/transferability.yaml")


conf = OmegaConf.load("../configs/transferability.yaml")
conf.model_id = "meta-llama/Llama-3.2-3B"
# ! setup
set_seeds(42)
tokenizer = AutoTokenizer.from_pretrained(conf.model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)
model.config.use_cache = False


# ! limit which parameters are trained
# conf.target_modules = ["gate_proj"]
conf.target_modules = ["up_proj"]
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in conf.target_modules)


wmdp = load_local(f"wmdp_deduped_bio/dev_T_corpus.jsonl")
fineweb_bio = load_fineweb_bio_corpus()
mmlu_bio = load_local("OUTDATED/my_generation2/mmlu_high_school_biology.jsonl")

# %% control_grad

# beginning, ending = "The term for self-fertilization is", "autogamy"
# beginning, ending = "The term for the death of cells is", "apoptosis"
beginning, ending = "Chlorophyll can be deactivated by", "strong acids"
# beginning, ending = "Resistance to beta-lactam antibiotics in bacteria is caused by mutations in the", "penicillin-binding protein gene"
control_grad = get_grad_from_pair(model, tokenizer, conf, beginning, ending)

# control_grad = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})
# for ex in fineweb_bio.select(range(40)):
#     txt = ex["text"]
#     full_batch = tokenizer(txt, **conf.tokenizer)
#     control_grad += get_grad(model, full_batch)

# control_grad = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})
# for c in wmdp.select(range(0, 4)):
#     for context in c["contexts"]:
#         control_grad += get_grad_from_pair(model, tokenizer, conf, context, c["answer_core"])

# control_grad = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})
# for c in mmlu_bio.select(range(0, 40)):
#     context = c["contexts"][0]
#     control_grad += get_grad_from_pair(model, tokenizer, conf, context, c["answer_core"])


# %% from_grad and to_grad

q = wmdp[12]
from_grad = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})
for context in q["contexts"][:5]:
# for context in q["contexts"][:1]:
    from_grad += get_grad_from_pair(model, tokenizer, conf, context, q["answer_core"])
from_grad /= 5
# from_grad /= 1


to_grad = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})
for context in q["contexts"][5:]:
    to_grad += get_grad_from_pair(model, tokenizer, conf, context, q["answer_core"])
to_grad /= 5

# beginning = format_prompt(q)
# ending = ["A", "B", "C", "D"][q["answer"]]
# to_grad = get_grad_from_pair(model, tokenizer, conf, beginning, ending)

q

# %% disr_grad

# # * on fineweb_bio, there's practically no detectable disruption
# disr_grad = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})
# for ex in fineweb_bio.select(range(30)):
#     txt = ex["text"]
#     full_batch = tokenizer(txt, **conf.tokenizer)
#     disr_grad += get_grad(model, full_batch)
# disr_grad /= 30

# # todo use questions from the other split?
# disr_grad = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})
# for d in wmdp.select(range(4, 8)):
#     for context in d["contexts"]:
#         disr_grad += get_grad_from_pair(model, tokenizer, conf, context, d["answer_core"])
# disr_grad /= 40

disr_grad = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})
for d in mmlu_bio.select(range(40, 80)):
    context = d["contexts"][0]
    disr_grad += get_grad_from_pair(model, tokenizer, conf, context, d["answer_core"])
disr_grad /= 40

# q

# %%

# unlearning_grad = from_grad

unlearning_grad = from_grad.clone().detach()
unlearning_grad *= (unlearning_grad * control_grad) < 0

good_transfer = (unlearning_grad * to_grad).sum()
bad_transfer = (unlearning_grad * disr_grad).sum()
del unlearning_grad
good_and_bad = pt.Tensor([
    list(good_transfer.values()),
    list(bad_transfer.values()),
])

# good_and_bad = good_and_bad[:, :-1]  # clip layers

im = good_and_bad / good_and_bad.max()
im = im.clip(min=0)
im[1] *= 10
im = im.cpu().float().numpy()
plt.imshow(im)

good_transfer = good_and_bad[0].sum()
bad_transfer = good_and_bad[1].sum()

metric = bad_transfer / good_transfer
print(f"{metric:9.6f}   {bad_transfer:5.2f} {good_transfer:5.2f}")

pt.cuda.empty_cache()

# %%
# For each of those sentences, try to generate a very similar sentence, each using similar wording and stucture, and if possible similar concepts, but the new fact must *not be related to viruses or bacteria in any way, nor any other kinds of biorisk*. Relation to biology is ok.