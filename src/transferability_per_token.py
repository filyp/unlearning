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

# pt.cuda.empty_cache()

# %%

# plt dark theme
plt.style.use("dark_background")

logging.basicConfig(level=logging.INFO)
pt.set_default_device("cuda")
conf = OmegaConf.load("../configs/transferability.yaml")
conf.model_id = "meta-llama/Llama-3.2-3B"

# load corpora
# paraphrases_all = load_local("my_generation2/wmdp_bio.jsonl")
en_qs = load_local(f"wmdp_deduped_bio/dev_T_corpus.jsonl")
# en_qs = load_local("gen3/mmlu_high_school_biology/en.jsonl")
# es_qs = load_local("gen3/mmlu_high_school_biology/es.jsonl")
# ru_qs = load_local("gen3/mmlu_high_school_biology/ru.jsonl")

fineweb_batches = load_batches(load_fineweb_edu_corpus(), conf.model_id, batch_size=8)


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
conf.target_modules = ["gate_proj"]
# conf.target_modules = ["up_proj"]
# conf.target_modules = ["down_proj"]
# conf.target_modules = ["k_proj"]
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


def get_grad_from_example(model, beginning, ending):
    beginning_batch = tokenizer(beginning, **conf.tokenizer)
    full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
    loss_mask = prepare_answer_mask(beginning_batch, full_batch)
    return get_grad(model, full_batch, loss_mask)


def get_grad_from_abcd_question(model, question):
    beginning = format_prompt(question)
    ending = ["A", "B", "C", "D"][question["answer"]]
    return get_grad_from_example(model, beginning, ending)


# %% derive target grad
q_index = 8
en_q = en_qs[q_index]
# es_q = es_qs[q_index]
# ru_q = ru_qs[q_index]
print(en_q["answer_core"])


target_grad = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})

# # ! abcd
# for q_rot in get_rotations(en_q):
#     target_grad += get_grad_from_abcd_question(model, q_rot)

# # ! russian example
# for ru_context in ru_q["contexts"]:
#     target_grad += get_grad_from_example(model, ru_context, ru_q["answer_core"])

# # ! spanish example
# for es_context in es_q["contexts"]:
#     target_grad += get_grad_from_example(model, es_context, es_q["answer_core"])

# # ! english example
# for context in en_q["contexts"][:5]:
#     target_grad += get_grad_from_example(model, context, en_q["answer_core"])


target_grad += get_grad_from_example(model, en_q["contexts"][0], en_q["answer_core"])

norm = pt.Tensor(list(target_grad.norm().values())).norm()
# target_grad /= norm


# # %%
# beginning = "A process in which an organism undergoes complete transformation from one life stage to another is called"
# ending = "metamorphosis"
# target_grad = get_grad_from_example(model, beginning, ending)

print(en_q["contexts"][0], en_q["answer_core"])

# %%

# beginning, ending = "The term for self-fertilization is", "autogamy"
# control_grad = get_grad_from_example(model, beginning, ending)

pt.cuda.empty_cache()
beginning, ending = en_q["contexts"][4], en_q["answer_core"]
# beginning, ending = "A process in which an egg develops without being fertilized is called (answer in Russian):", "партеногене́з"
# beginning, ending = "A process in which an egg develops without being fertilized is called X. X:", "parthogenesis"
# beginning, ending = "A process in which an egg develops without being fertilized is called X. The first letter of X is", "p"

with CalcSimilarityHooks(model, target_grad):
# with CalcSimilarityHooks(model, target_grad, control_grad):
    get_grad_from_example(model, beginning, ending)

full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
tokens = [tokenizer.decode(id_) for id_ in full_batch["input_ids"][0]]

target_self = []
for i, l in enumerate(model.model.layers):
    # module = l.mlp.up_proj
    module = l.mlp.gate_proj
    # module = l.mlp.down_proj
    # module = l.self_attn.o_proj

    w = module.weight
    target_self.append([w.target_sim, w.self_sim])

target_self = pt.Tensor(target_self)
flatten_pow = 1
target_self = target_self.clip(min=0) ** flatten_pow
# shape is (layers, types, tokens)

# remove BOS
target_self = target_self[:, :, 1:]
tokens = tokens[1:]

# target_self[:, 0] *= 10
print(target_self.max())
target_self /= target_self.max()
# self is red, target is green
visualize_token_layer_values(target_self[:, 1], target_self[:, 0], tokens, "")



# %%

# %%

# %%
# two grads, look at transferability, per weight, on some module weights

# name = "model.layers.8.self_attn.o_proj.weight"
name = "model.layers.10.mlp.up_proj.weight"
t = target_grad[name]
grad = get_grad_from_example(model, beginning, ending)
g = grad[name]
g = t * g

# # g[4000:].sum(dim=0)
# t[:4000].sum(dim=0)
# t[:, :1000].sum(dim=1)

g = g.detach().float().cpu().numpy()
g = g[:100, :100]

# Create custom colormap: red -> black -> green
colors = ["red", "black", "green"]
n_bins = 256
cmap = mcolors.LinearSegmentedColormap.from_list("RedBlackGreen", colors, N=n_bins)

# imshow g with custom colormap (red-black-green)
plt.imshow(g, cmap=cmap, norm=plt.Normalize(vmin=-abs(g).max(), vmax=abs(g).max()))
plt.colorbar()

plt.show()

# %%
