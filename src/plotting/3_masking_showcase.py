# %%
# %load_ext autoreload
# %autoreload 2
import logging
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from IPython.display import HTML, display
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MPLPath
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.git_and_reproducibility import repo_root
from utils.plots import visualize, visualize_rgb
from utils.training import get_grad, prepare_answer_mask, set_seeds

# plt dark theme
# plt.style.use("dark_background")
plt.style.use("default")

logging.basicConfig(level=logging.INFO)
pt.set_default_device("cuda")

model_id = "meta-llama/Llama-3.2-1B"
# ! setup
set_seeds(42)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)
model.config.use_cache = False

tokenizer_kwargs = dict(
    max_length=128,
    padding=True,
    truncation=True,
    return_tensors="pt",
)


def get_grad_from_example(model, beginning, ending, loss_fn_name="cross_entropy"):
    beginning_batch = tokenizer(beginning, **tokenizer_kwargs)
    full_batch = tokenizer(f"{beginning} {ending}", **tokenizer_kwargs)
    loss_mask = prepare_answer_mask(beginning_batch, full_batch)
    return get_grad(model, full_batch, loss_mask, loss_fn_name)


# ! limit which parameters are trained
target_modules = ["gate_proj", "up_proj", "down_proj"]
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in target_modules)


# %%
vs = []
# * we look at the transfer from the first pair to the last
# * the other pairs serve as basis for creating the mask
for beginning, ending in [
    ("The capital of France is", "Paris"),
    ("France's capital is", "Paris"),
    ("The capital of Spain is", "Madrid"),
    ("The capital of Italy is", "Rome"),
    ("The capital of England is", "London"),
    ("The capital of Poland is", "Warsaw"),
    # ("The capital of Argentina is", "Buenos Aires"),
    # ("The capital of Chile is", "Santiago"),
    # ("The capital of Japan is", "Tokio"),
    # ("The capital of Germany is", "Berlin"),
    # ("The capital of China is", "Beijing"),
    # ("The capital of Ukraine is", "Kyiv"),
    # ("The capital of Russia is", "Moscow"),
    # ("The capital of Poland is", "Kraków"),
    # ("The capital of Japan is", "Tokyo"),
    # ("The capital of Italy is", "Paris"),
    # ("The capital of the country below England is", "Paris"),
    # ("Столица Франции", "Париж"),
    # ("Stolica Francji to", "Paryz"),
    # ("La capital de Francia es", "París"),
    # ("Die Hauptstadt von Frankreich ist", "Paris"),
    # ("A capital de França é", "Paris"),
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
    # ("The symbol of helium is", "He"),
    # ("The Brandenburg Gate is in", "Berlin"),
    # ("The oldest building in the world is", "The Great Pyramid of Giza"),
]:
    _g = get_grad_from_example(model, beginning, ending, loss_fn_name="cross_entropy")
    # _g = get_grad_from_example(model, beginning, ending, loss_fn_name="correct_logit")
    vs.append(_g)


# # %%
# module_name = "model.layers.9.mlp.gate_proj.weight"
# # module_name = "model.layers.9.mlp.up_proj.weight"
# # module_name = "model.layers.9.mlp.down_proj.weight"
# v0 = vs[0][module_name]
# v1 = vs[1][module_name]
# v2 = vs[2][module_name]
# v3 = vs[3][module_name]

# r = v0 * v2
# g = v0 * v1
# b = pt.zeros_like(v0)

# control = v0 * v3

# # # weight masking
# # mask = control > 0
# # r[mask] = 0
# # g[mask] = 0

# column_mask = control.sum(dim=0) > 0
# r[:, column_mask] = 0
# g[:, column_mask] = 0

# # row_mask = control.sum(dim=1) > 0
# # r[row_mask, :] = 0
# # g[row_mask, :] = 0

# print("r sum", r.sum().item())
# print("g sum", g.sum().item())
# print("ratio", r.sum().item() / g.sum().item())

# # plot
# rgb = pt.stack([r, g, b], dim=-1)
# rgb = rgb.clip(0) ** 0.5

# # visualize_rgb(rgb)
# rgb = rgb.cpu().float().numpy()
# rgb = rgb[:shape_lim, :shape_lim]

# max_val = np.abs(rgb).max()
# rgb = rgb / max_val

# plt.imshow(rgb)


# %%
# plot_module_name = "model.layers.11.mlp.gate_proj.weight"
# module_name = "model.layers.9.mlp.up_proj.weight"
# module_name = "model.layers.9.mlp.down_proj.weight"
plot_module_name = "model.layers.12.mlp.down_proj.weight"

shape_lim = 25
flattening = 0.5
r_mult = 3

fig, axes = plt.subplots(1, 4, figsize=(5.5, 3))


# # %%
axes_id = 1
title = "Unmasked"

r_sum = 0
g_sum = 0
for module_name in vs[0].keys():
    v0 = vs[0][module_name]
    v1 = vs[1][module_name]
    v2 = vs[2][module_name]
    v3 = vs[3][module_name]
    v4 = vs[4][module_name]
    v5 = vs[5][module_name]

    self_sim = v0 * v0
    r = v0 * v2
    g = v0 * v1
    b = pt.zeros_like(v0)
    final_mask = pt.ones_like(v0)

    control = v0 * v3
    # control = v0 * (v3 + v4 + v5)

    # # # weight masking
    # mask = (control > 0)
    # # mask = (control > 0) & (self_sim / control < 8)
    # r[mask] = 0
    # g[mask] = 0
    # final_mask[mask] = 0

    # # column_mask = control.sum(dim=0) > 0
    # # column_mask = (control.sum(dim=0) > 0) & (self_sim.sum(dim=0) / control.sum(dim=0) < 1.45)
    # column_mask = (control.sum(dim=0) > 0) & (self_sim.sum(dim=0) / control.sum(dim=0) < 10)
    # r[:, column_mask] = 0
    # g[:, column_mask] = 0
    # final_mask[:, column_mask] = 0
    # # row_mask = control.sum(dim=1) > 0
    # row_mask = (control.sum(dim=1) > 0) & (self_sim.sum(dim=1) / control.sum(dim=1) < 2)
    # r[row_mask, :] = 0
    # g[row_mask, :] = 0
    # final_mask[row_mask, :] = 0

    r_sum += r.sum().item()
    g_sum += g.sum().item()

    if module_name == plot_module_name:
        # plot
        rgb = pt.stack([r * r_mult, g, b], dim=-1)
        rgb = rgb.clip(0)

        rgb = rgb[:shape_lim, :shape_lim]
        brightness = rgb.sum(dim=-1, keepdim=True)
        rgb /= brightness**flattening
        rgb[rgb.isnan()] = 0
        # rgb = rgb * 4
        rgb /= rgb.max()

        final_mask = final_mask.to(bool)[:shape_lim, :shape_lim]
        rgb[~final_mask] = 1
        rgb = rgb.cpu().float().numpy()
        # transpose to make the column differences more salient
        axes[axes_id].imshow(rgb.transpose(1, 0, 2))
        axes[axes_id].set_xticks([])
        axes[axes_id].set_yticks([])
        axes[axes_id].set_title(title)

print(f"r_sum: {r_sum:.2f}   g_sum: {g_sum:.2f}   ratio: {r_sum / g_sum:.2f}")

# # %%
axes_id = 2
title = "Masked per\nweight"

r_sum = 0
g_sum = 0
for module_name in vs[0].keys():
    v0 = vs[0][module_name]
    v1 = vs[1][module_name]
    v2 = vs[2][module_name]
    v3 = vs[3][module_name]
    v4 = vs[4][module_name]
    v5 = vs[5][module_name]

    self_sim = v0 * v0
    r = v0 * v2
    g = v0 * v1
    b = pt.zeros_like(v0)
    final_mask = pt.ones_like(v0)

    control = v0 * v3
    # control = v0 * (v3 + v4 + v5)

    # # weight masking
    mask = control > 0
    # mask = (control > 0) & (self_sim / control < 8)
    r[mask] = 0
    g[mask] = 0
    final_mask[mask] = 0

    # # column_mask = control.sum(dim=0) > 0
    # # column_mask = (control.sum(dim=0) > 0) & (self_sim.sum(dim=0) / control.sum(dim=0) < 1.45)
    # column_mask = (control.sum(dim=0) > 0) & (self_sim.sum(dim=0) / control.sum(dim=0) < 10)
    # r[:, column_mask] = 0
    # g[:, column_mask] = 0
    # final_mask[:, column_mask] = 0
    # # row_mask = control.sum(dim=1) > 0
    # row_mask = (control.sum(dim=1) > 0) & (self_sim.sum(dim=1) / control.sum(dim=1) < 2)
    # r[row_mask, :] = 0
    # g[row_mask, :] = 0
    # final_mask[row_mask, :] = 0

    r_sum += r.sum().item()
    g_sum += g.sum().item()

    if module_name == plot_module_name:
        # plot
        rgb = pt.stack([r * r_mult, g, b], dim=-1)
        rgb = rgb.clip(0)

        rgb = rgb[:shape_lim, :shape_lim]
        brightness = rgb.sum(dim=-1, keepdim=True)
        rgb /= brightness**flattening
        rgb[rgb.isnan()] = 0
        # rgb = rgb * 4
        rgb /= rgb.max()

        final_mask = final_mask.to(bool)[:shape_lim, :shape_lim]
        rgb[~final_mask] = 1
        rgb = rgb.cpu().float().numpy()
        # transpose to make the column differences more salient
        axes[axes_id].imshow(rgb.transpose(1, 0, 2))
        axes[axes_id].set_xticks([])
        axes[axes_id].set_yticks([])
        axes[axes_id].set_title(title)

print(f"r_sum: {r_sum:.2f}   g_sum: {g_sum:.2f}   ratio: {r_sum / g_sum:.2f}")

# # %%
axes_id = 3
title = "Masked per\ncolumn and row"

r_sum = 0
g_sum = 0
for module_name in vs[0].keys():
    v0 = vs[0][module_name]
    v1 = vs[1][module_name]
    v2 = vs[2][module_name]
    v3 = vs[3][module_name]
    v4 = vs[4][module_name]
    v5 = vs[5][module_name]

    self_sim = v0 * v0
    r = v0 * v2
    g = v0 * v1
    b = pt.zeros_like(v0)
    final_mask = pt.ones_like(v0)

    control = v0 * v3
    # control = v0 * (v3 + v4 + v5)

    # # # weight masking
    # mask = (control > 0)
    # # mask = (control > 0) & (self_sim / control < 8)
    # r[mask] = 0
    # g[mask] = 0
    # final_mask[mask] = 0

    # column_mask = control.sum(dim=0) > 0
    # column_mask = (control.sum(dim=0) > 0) & (self_sim.sum(dim=0) / control.sum(dim=0) < 1.45)
    column_mask = (control.sum(dim=0) > 0) & (
        self_sim.sum(dim=0) / control.sum(dim=0) < 9
    )
    r[:, column_mask] = 0
    g[:, column_mask] = 0
    final_mask[:, column_mask] = 0

    # row_mask = control.sum(dim=1) > 0
    row_mask = (control.sum(dim=1) > 0) & (self_sim.sum(dim=1) / control.sum(dim=1) < 4)
    r[row_mask, :] = 0
    g[row_mask, :] = 0
    final_mask[row_mask, :] = 0

    r_sum += r.sum().item()
    g_sum += g.sum().item()

    if module_name == plot_module_name:
        # plot
        rgb = pt.stack([r * r_mult, g, b], dim=-1)
        rgb = rgb.clip(0)

        rgb = rgb[:shape_lim, :shape_lim]
        brightness = rgb.sum(dim=-1, keepdim=True)
        rgb /= brightness**flattening
        rgb[rgb.isnan()] = 0
        # rgb = rgb * 4
        rgb /= rgb.max()

        final_mask = final_mask.to(bool)[:shape_lim, :shape_lim]
        rgb[~final_mask] = 1
        rgb = rgb.cpu().float().numpy()
        # transpose to make the column differences more salient
        axes[axes_id].imshow(rgb.transpose(1, 0, 2))
        axes[axes_id].set_xticks([])
        axes[axes_id].set_yticks([])
        axes[axes_id].set_title(title)

print(f"r_sum: {r_sum:.2f}   g_sum: {g_sum:.2f}   ratio: {r_sum / g_sum:.2f}")


# # %%
v0 = vs[0][plot_module_name]
v3 = vs[3][plot_module_name]
v4 = vs[4][plot_module_name]
v5 = vs[5][plot_module_name]
control = v0 * v3
# control = v0 * (v3 + v4 + v5)

values = -(control.clip(0) ** (1 - flattening))

values = values.cpu().float().numpy()
values = values[:shape_lim, :shape_lim]

max_val = np.abs(values).max()
values = values / max_val
values = np.stack(
    # [np.clip(-values, 0, 1), np.clip(values, 0, 1), np.zeros_like(values)], axis=-1
    # [np.zeros_like(values),np.zeros_like(values), np.clip(-values, 0, 1)], axis=-1
    [np.zeros_like(values), 0.7 * np.clip(-values, 0, 1), np.clip(-values, 0, 1)],
    axis=-1,
)
# remove ticks
axes[0].set_xticks([])
axes[0].set_yticks([])
# transpose to make the column differences more salient
axes[0].imshow(values.transpose(1, 0, 2))
axes[0].set_title("Control")


# # %%

# * draw arrows
arrow_conf = dict(
    transform=fig.transFigure,
    arrowstyle="->",  # Change to arrow style
    mutation_scale=20,
    linewidth=2,
    color="black",
)
vertices = [(0.138, 0.29), (0.138, 0.21), (0.36, 0.21)]
codes = [MPLPath.MOVETO, MPLPath.LINETO, MPLPath.LINETO]
arrow_path = MPLPath(vertices, codes)
fig.add_artist(FancyArrowPatch(path=arrow_path, **arrow_conf))
vertices = [(0.384, 0.29), (0.384, 0.21), (0.87, 0.21), (0.87, 0.3)]
codes = [MPLPath.MOVETO, MPLPath.LINETO, MPLPath.LINETO, MPLPath.LINETO]
arrow_path = MPLPath(vertices, codes)
fig.add_artist(FancyArrowPatch(path=arrow_path, **arrow_conf))
vertices = [(0.621, 0.21), (0.621, 0.3)]
codes = [MPLPath.MOVETO, MPLPath.LINETO]
arrow_path = MPLPath(vertices, codes)
fig.add_artist(FancyArrowPatch(path=arrow_path, **arrow_conf))


h = 0.09
fig.text(0.01, h, "disruption", color="red")
fig.text(0.141, h, "/", color="black")
fig.text(0.153, h, "transfer", color="green")
fig.text(0.252, h, ":", color="black")
fig.text(0.36, h, "58%")
fig.text(0.6, h, "33%")
fig.text(0.855, h, "5%")


stem = Path(__file__).stem
plot_path = repo_root() / f"paper/plots/{stem}.pdf"
# fig rather than plt! otherwise only blank figure is saved, bc it's been shown already
fig.tight_layout()
fig.savefig(plot_path, bbox_inches=None, dpi=300)


# # %% shows how much 3 examples differ
# v0 = vs[0][plot_module_name]
# v1 = vs[1][plot_module_name]
# v2 = vs[2][plot_module_name]
# v3 = vs[3][plot_module_name]
# v4 = vs[4][plot_module_name]

# r = v0 * v2
# g = v0 * v3
# b = v0 * v4

# values = pt.stack([r, g, b], dim=-1).clip(0) ** 0.5
# values = values.cpu().float().numpy()

# values = values[:shape_lim, :shape_lim]

# max_val = np.abs(values).max()
# values = values / max_val

# # transpose to make the column differences more salient
# plt.imshow(values.transpose(1, 0, 2))
# plt.xticks([])
# plt.yticks([])
