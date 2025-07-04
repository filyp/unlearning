# %%
# %load_ext autoreload
# %autoreload 2
import logging
from copy import deepcopy

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch as pt
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_loading import load_batches, load_fineweb_edu_corpus, load_local
from utils.evals import format_prompt
from utils.similarity_hooks import CalcSimilarityHooks
from utils.plots import visualize_token_layer_values
from utils.training import get_grad, prepare_answer_mask, set_seeds

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
# conf.target_modules = ["gate_proj"]
conf.target_modules = ["up_proj"]
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


def get_grad_from_example(model, beginning, ending, loss_fn_name="cross_entropy"):
    beginning_batch = tokenizer(beginning, **conf.tokenizer)
    full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
    loss_mask = prepare_answer_mask(beginning_batch, full_batch)
    return get_grad(model, full_batch, loss_mask, loss_fn_name)


def get_grad_from_abcd_question(model, question, loss_fn_name="cross_entropy"):
    beginning = format_prompt(question)
    ending = ["A", "B", "C", "D"][question["answer"]]
    return get_grad_from_example(model, beginning, ending, loss_fn_name)

# derive target grad

# target_grad = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})

# # ! abcd
# for q_rot in get_rotations(en_q):
#     target_grad += get_grad_from_abcd_question(model, q_rot)

# # ! russian example
# for ru_context in ru_q["contexts"]:
#     target_grad += get_grad_from_example(model, ru_context, ru_q["answer_core"], loss_fn_name=loss_fn_name)

# # ! spanish example
# for es_context in es_q["contexts"]:
#     target_grad += get_grad_from_example(model, es_context, es_q["answer_core"], loss_fn_name=loss_fn_name)

# # ! english example
# for context in en_q["contexts"][:5]:
#     target_grad += get_grad_from_example(model, context, en_q["answer_core"], loss_fn_name=loss_fn_name)

# target_grad += get_grad_from_example(model, en_q["contexts"][0], en_q["answer_core"], loss_fn_name=loss_fn_name)


# # %%
# beginning = "A process in which an organism undergoes complete transformation from one life stage to another is called"
# ending = "metamorphosis"
# target_grad = get_grad_from_example(model, beginning, ending)

# # %%

# beginning, ending = "A process in which an egg develops without being fertilized is called (answer in Russian):", "партеногене́з"
# beginning, ending = "A process in which an egg develops without being fertilized is called X. X:", "parthogenesis"
# beginning, ending = "A process in which an egg develops without being fertilized is called X. The first letter of X is", "p"


# %%
# loss_fn_name = "neg_cross_entropy"
loss_fn_name = "correct_logit"


def get_transfers(beginning, ending, target_grad):
    with CalcSimilarityHooks(model, target_grad):
        get_grad_from_example(model, beginning, ending, loss_fn_name=loss_fn_name)
    
    target_self = [l.mlp.up_proj.weight.target_sim for l in model.model.layers]
    return pt.Tensor(target_self)


q_index = 17
q = en_qs[q_index]
print(q["contexts"][0], q["answer_core"])


# * from
beginning, ending = q["contexts"][4], q["answer_core"]
full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
tokens = [tokenizer.decode(id_) for id_ in full_batch["input_ids"][0]]

# * to
target_grad = get_grad_from_example(model, q["contexts"][0], q["answer_core"], loss_fn_name=loss_fn_name)
# target_grad = get_grad_from_abcd_question(model, q, loss_fn_name=loss_fn_name)
# model.zero_grad(set_to_none=True)
# acc = eval_on([q], model, temperature=1)
# acc.backward()
# target_grad = TensorDict(
#     {n: p.grad for n, p in model.named_parameters() if p.requires_grad},
# )
target_transfers = get_transfers(beginning, ending, target_grad)

# * disruption
alt_q = en_qs[0]
disr_grad = get_grad_from_example(model, alt_q["contexts"][0], alt_q["answer_core"], loss_fn_name=loss_fn_name)
disr_transfers = get_transfers(beginning, ending, disr_grad)

# shape is (layers, tokens)

# * normalize
disr_transfers *= 50
# disr_transfers = disr_transfers.abs()
# target_transfers = target_transfers.abs()
max_ = max(disr_transfers.max(), target_transfers.max())
print(disr_transfers.max(), target_transfers.max())
disr_transfers /= max_
target_transfers /= max_

# * clip and potentially flatten
flatten_pow = 1.0
disr_transfers = disr_transfers.clip(min=0) ** flatten_pow
target_transfers = target_transfers.clip(min=0) ** flatten_pow

# remove BOS
disr_transfers = disr_transfers[:, 1:]
target_transfers = target_transfers[:, 1:]
tokens = tokens[1:]

# disruption is red, target is green
visualize_token_layer_values(disr_transfers, target_transfers, tokens, "")



# %%

# %% two grads, look at transferability, per weight, on some module weights

# name = "model.layers.8.self_attn.o_proj.weight"
# name = "model.layers.10.mlp.up_proj.weight"
name = "model.layers.10.mlp.gate_proj.weight"
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
