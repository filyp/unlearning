# %%
# %load_ext autoreload
# %autoreload 2
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.training import get_grad, prepare_answer_mask, set_seeds

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
conf.target_modules = ["gate_proj"]
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in conf.target_modules)


# %%
module_name = "model.layers.14.mlp.gate_proj.weight"

beginning, ending = "The capital of France is", "Paris"

# beginning, ending = "The symbol of helium is", "He"

# beginning, ending = "Man's best friend is", "dog"

# beginning, ending = "The tallest building in the world is", "Burj Khalifa"

# beginning, ending = "T-2 Mycotoxin can be neutralized by treatment with ", "Sodium hypochlorite"

x = get_grad_from_example(model, beginning, ending)[module_name]
x = x.cpu().float().numpy()

# %%

beginning, ending = "The capital of Italy is", "Rome"
# beginning, ending = "The capital of USA is", "Washington"
# beginning, ending = "The capital of Poland is", "Warsaw"
# beginning, ending = "The capital of Argentina is", "Buenos Aires"
# beginning, ending = "The capital of China is", "Beijing"
# beginning, ending = "The capital of France is", "Tokio"

# beginning, ending = "The anthem of France is", "La Marseillaise"
# beginning, ending = "The Eiffel Tower is in", "Paris"
# beginning, ending = "The city of Eiffel Tower is", "Paris"
# beginning, ending = "The Statue of Liberty is in", "New York"
# beginning, ending = "The Brandenburg Gate is in", "Berlin"
# beginning, ending = "Stolica Francji to", "Paryz"

# beginning, ending = "The oldest building in the world is", "The Great Pyramid of Giza"

# beginning, ending = "The oldest building in the world is", "The Great Pyramid of Giza"
# beginning, ending = "Aflatoxin B1 can be neutralized by treatment with ", "Ozone"
# beginning, ending = "Deoxynivalenol can be neutralized by treatment with ", "Ammonia vapor"
# beginning, ending = "Deoxynivalenol can be neutralized by treatment with ", "Sodium hypochlorite"

y = get_grad_from_example(model, beginning, ending)[module_name]
y = y.cpu().float().numpy()

# %%

indices = np.random.choice(len(x.flatten()), size=100_000, replace=False)
plt.gca().set_aspect("equal")
plt.axhline(0, color="grey", linestyle="-")
plt.axvline(0, color="grey", linestyle="-")
plt.scatter(x.flatten()[indices], y.flatten()[indices], alpha=0.1, s=1)
scale = 0.1
plt.xlim(-scale, scale)
plt.ylim(-scale, scale)

np.corrcoef(x.flatten(), y.flatten())[0, 1]

# %%
z = (x * y).clip(min=0) ** 0.5
lim = 50
plt.imshow(np.abs(x)[:lim, :lim])
# plt.imshow(np.abs(y)[:lim, :lim])
# plt.imshow(z[:lim, :lim])
plt.colorbar()
np.corrcoef(x.flatten(), y.flatten())[0, 1]

# %%
z = z.clip(min=0)
# Create RGB image with x as green channel and z as red channel
lim = 50
x_slice = np.abs(x)[:lim, :lim]
z_slice = z[:lim, :lim]

# Create RGB image: Red=z, Green=x, Blue=0
rgb_image = np.zeros((lim, lim, 3))
rgb_image[:, :, 0] = z_slice * 2  # Red channel = z
rgb_image[:, :, 1] = x_slice  # Green channel = x
rgb_image[:, :, 2] = 0  # Blue channel = 0

rgb_image = rgb_image.mean(axis=0, keepdims=True)
rgb_image /= rgb_image.max()

plt.figure(figsize=(8, 8))
plt.imshow(rgb_image)
plt.title("RGB Image: Red=z, Green=|x|, Blue=0")
plt.axis("off")
plt.show()

# %%

# %%
module_name = "model.layers.15.mlp.up_proj.weight"

vs = []
for beginning, ending in [
    ("The term for egg development without fertilization is", "parthenogenesis"),
    (
        "Formation of offspring from eggs without male contribution is termed",
        "parthenogenesis",
    ),
    ("The term for self-fertilization is", "autogamy"),
    (
        "The scientific name for reproduction from unfertilized eggs is",
        "parthenogenesis",
    ),
    ("The term for the death of cells is", "apoptosis"),
]:
    _g = get_grad_from_example(model, beginning, ending)[module_name]
    vs.append(_g)
# %%
transfer = vs[0] * vs[1]
bad_transfer = vs[0] * vs[2]
# transfer = (vs[0] * vs[1]).sum(dim=0, keepdim=True)
# bad_transfer = (vs[0] * vs[2]).sum(dim=0, keepdim=True)
# transfer = (vs[0] * vs[1]).sum(dim=1, keepdim=True)
# bad_transfer = (vs[0] * vs[2]).sum(dim=1, keepdim=True)

transfer = pt.ones_like(transfer) * transfer.mean()
# bad_transfer = pt.ones_like(bad_transfer) * bad_transfer.mean()

# thresh = 0.
# mask = bad_transfer / transfer.clip(min=0.00001) < thresh
# mask = mask & (transfer > 0)
mask = bad_transfer < 0
# mask = vs[0].sign() != vs[2].sign()

# %%
# project out the bad direction
final = vs[0].flatten()
to_avoid = vs[2].flatten()
to_avoid /= to_avoid.norm()
final = final - (final * to_avoid).sum() * to_avoid
final = final.reshape(vs[0].shape)

# %%
# final = vs[0]

_g = final * vs[3]
_b = final * vs[4]

_g = _g * mask
_b = _b * mask

good = _g.sum().item()
bad = _b.sum().item()
print(bad, good, bad / good)

# %%

# %%

# %%
# im = np.abs(vs[0])[:100, :100]
# _g = _g.cpu().float().numpy()
im = (vs[0] * vs[3])[:100, :100]
plt.imshow(im)
# (vs[0] * vs[4]).sum()
# np.corrcoef(vs[0].flatten(), vs[4].flatten())[0, 1]

# %%
indices = np.random.choice(len(transfer.flatten()), size=100_000, replace=False)
plt.scatter(
    transfer.flatten()[indices],
    bad_transfer.flatten()[indices],
    alpha=0.5,
    s=1,
    c=mask.flatten()[indices],
)
plt.gca().set_aspect("equal")
