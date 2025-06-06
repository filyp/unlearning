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
from utils.data_loading import load_batches, load_fineweb_edu_corpus, load_local
from utils.evals import eval_on, format_prompt
from utils.git_and_reproducibility import repo_root
from utils.hooks import CalcSimilarityHooks
from utils.loss_fns import print_per_token_colored_loss
from utils.plots import visualize_module_values, visualize_token_layer_values
from utils.training import get_grad, prepare_answer_mask, set_seeds, trainable_params

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


def get_grad_from_example(model, beginning, ending):
    beginning_batch = tokenizer(beginning, **conf.tokenizer)
    full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
    loss_mask = prepare_answer_mask(beginning_batch, full_batch)
    return get_grad(model, full_batch, loss_mask)


# %%

# ! limit which parameters are trained
# conf.target_modules = ["gate_proj"]
conf.target_modules = ["up_proj"]
# conf.target_modules = ["down_proj"]
# conf.target_modules = ["k_proj"]
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in conf.target_modules)


# beginning, ending = "The symbol of helium is", "He"

# beginning, ending = "The anthem of France is", "La Marseillaise"
# beginning, ending = "The city of Eiffel Tower is", "Paris"
# beginning, ending = "The Statue of Liberty is in", "New York"
# beginning, ending = "The Brandenburg Gate is in", "Berlin"
# beginning, ending = "Stolica Francji to", "Paryz"

# beginning, ending = "The oldest building in the world is", "The Great Pyramid of Giza"

# %%
# module_name = "model.layers.14.mlp.gate_proj.weight"

vs = []
for beginning, ending in [
    ("The capital of France is", "Paris"),
    # ("The capital of the country below England is", "Paris"),
    ("The capital of Italy is", "Rome"),
    # ("The capital of Spain is", "Madrid"),
    # ("The capital of England is", "London"),
    # ("The capital of Poland is", "Warsaw"),
    # ("The capital of Argentina is", "Buenos Aires"),
    # ("The capital of Japan is", "Tokio"),

    # ("Столица Франции", "Париж"),
    # ("Stolica Francji to", "Paryz"),
    # ("La capital de Francia es", "París"),
    # ("Die Hauptstadt von Frankreich ist", "Paris"),
    ("A capital de França é", "Paris"),

    # ("The capital of China is", "Beijing"),
    # ("The capital of Germany is", "Berlin"),
    # ("The capital of Ukraine is", "Kyiv"),
    # ("The anthem of France is", "La Marseillaise"),
    # ("The city of Eiffel Tower is", "Paris"),
    # ("The Statue of Liberty is in", "New York"),
    # ("The hardest metal is", "Tungsten"),

    # ("The term for egg development without fertilization is", "parthenogenesis"),
    # ("The term for self-fertilization is", "autogamy"),
    # ("The term for the death of cells is", "apoptosis"),

    # ("The term for self-fertilization is", "autogamy"),
    # ("The term for the death of cells is", "apoptosis"),
    # (format_prompt(q), ["A", "B", "C", "D"][q["answer"]]),

]:
    _g = get_grad_from_example(model, beginning, ending)
    vs.append(_g)

# # d = (vs[1] + vs[2] + vs[3] + vs[4]) / 4
# d = (vs[1] + vs[2]) / 2
# mask = (vs[0] * d) < 0
# mask |= (d / vs[0]) < 0.2


res = []
for final in [
    vs[0],
    vs[0] * ((vs[0] * (vs[1])) < 0),
    # vs[0] * ((vs[0] * (vs[2])) < 0),
    # vs[0] * ((vs[0] * (vs[1] + vs[2] + vs[3])) < 0),
    # vs[0] * ((vs[0] * (vs[1] + vs[2] + vs[3] + vs[4])) < 0),
    # vs[0] * mask,
    # vs[0] * ((vs[0] * (vs[1] + vs[2])) < 0),
]:
    bad = (final * vs[-1]).sum()
    good = (final * final).sum()

    bad = sum(v for v in bad.values())
    good = sum(v for v in good.values())

    metric = bad / good
    print(f"{metric:7.4f}   {bad:5.2f} {good:5.2f}")
    # print([m.item() for m in metric.values()])

    # bad = list(bad.values())
    # good = list(good.values())
    # res.append((bad, good))

# res = pt.Tensor(res)
# # res[:, :, 1] *= 0
# res /= res.max()
# r_channel = res[:, 0, :]
# g_channel = res[:, 1, :]
# # imshow
# colors = pt.zeros((r_channel.shape[0], r_channel.shape[1], 3))
# colors[:, :, 0] = r_channel * 10
# colors[:, :, 1] = g_channel
# colors = colors.clip(0, 1)
# # colors = colors / colors.max(dim=-1, keepdim=True).values
# colors[1] *= 10
# colors = colors.cpu().numpy()
# plt.imshow(colors)

# %%
# project out the bad direction
final = vs[0].flatten()
to_avoid = vs[2].flatten()
to_avoid /= to_avoid.norm()
final = final - (final * to_avoid).sum() * to_avoid
final = final.reshape(vs[0].shape)

# %%
# im = np.abs(vs[0])[:100, :100]
# _g = _g.cpu().float().numpy()
im = (vs[0] * vs[3])[:100, :100]
plt.imshow(im)
# (vs[0] * vs[4]).sum()
# np.corrcoef(vs[0].flatten(), vs[4].flatten())[0, 1]
