# %%
# %load_ext autoreload
# %autoreload 2
import logging
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM

import wandb
from utils import loss_fns
from utils.data_loading import (
    data_paths,
    filter_by_question,
    load_batches,
    load_low_mi_set,
    load_retain_corpus,
)
from utils.evals import eval_on
from utils.loss_fns import cross_entropy_loss, loss_fns
from utils.training import set_seeds

logging.basicConfig(level=logging.INFO)

pt.set_default_device("cuda")

s = OmegaConf.create(
    dict(
        model_id="meta-llama/Llama-3.2-1B",
        unlearning_epochs=12,
        max_length=128,
    )
)

# %%
# load corpora
f_full_corpus = load_low_mi_set(data_paths["wmdp_deduped_unlearning"])
r_full_corpus = load_low_mi_set(data_paths["wmdp_deduped_wrong_answers"])

# load questions
wmdp_mcq_full = load_low_mi_set(data_paths["wmdp_deduped_mcq_full"])

# load disrution eval set
d_corpus = load_retain_corpus("fineweb_edu")
disruption_batches = load_batches(d_corpus, s.model_id, 16, s.max_length)

# # mmlu_set = load_dataset("cais/mmlu", "college_biology", split="test")
# mmlu_set = load_dataset("cais/mmlu", "all", split="validation")
# mmlu_set = mmlu_set.shuffle(seed=0).select(range(64))

# %%
questions_with_at_least_45_perc = [3, 5, 16, 21, 22, 41, 54, 61, 76, 80, 83, 84, 95, 97, 99, 101, 102, 103, 106, 107, 111, 121, 127, 134, 138, 145, 147, 149, 150, 154, 159, 166, 167, 173, 182, 186, 192, 197, 201, 207, 212, 216, 221, 227, 229, 239, 243, 250, 251, 257, 273, 283, 286, 288, 299, 300, 304, 321, 339, 352, 362, 363, 364, 374, 377, 389, 394, 397, 404, 412, 417, 419, 420, 442, 445, 453, 459, 481, 490, 495, 499, 514, 518, 524, 526, 529, 541, 545, 559, 565, 568, 570, 583, 589, 593, 596, 613, 645, 646, 668, 678, 679, 684, 689, 694, 698, 699, 700, 713, 721, 724, 744, 747, 748, 755, 758, 774]  # fmt: skip

# choose eval question
# nice questions: 3, 5, 10, 12
question_index = questions_with_at_least_45_perc[10]
f_eval_set = wmdp_mcq_full.select([question_index])
target_question = f_eval_set[0]["question"]
print(f"{target_question=}")

# load forget set
f_corpus = f_full_corpus.filter(lambda ex: ex["original_question"] == target_question)
forget_batches = load_batches(f_corpus, s.model_id, 3, s.max_length)
assert len(forget_batches) == 1

# load retain set
r_corpus = r_full_corpus.filter(lambda ex: ex["original_question"] == target_question)
# r_corpus = r_full_corpus.shuffle(seed=42).select(range(3))  # ! unrelated questions
retain_batches = load_batches(r_corpus, s.model_id, 3, s.max_length)
assert len(retain_batches) == 1


def _eval_callback(model):
    model.eval()
    # eval mmlu and wmdp
    with pt.no_grad():
        wmdp_acc = eval_on(f_eval_set, model, temperature=1)
        # mmlu_acc = eval_on(mmlu_set, model, temperature=1)

        loss = 0
        for d_batch in disruption_batches[-8:]:
            output = model(**d_batch)
            loss += cross_entropy_loss(output, d_batch)
        disr_loss = loss / len(disruption_batches[-8:])

    # logging.info(f"{wmdp_acc=:.4f} {disr_loss=:.4f}")
    # wandb.log({"wmdp_acc": wmdp_acc, "disr_loss": disr_loss, "update_norm2": update_norm2})
    # if wmdp_acc < 0.3 or disr_loss > 2.62:
    #     raise StopIteration
    return wmdp_acc, disr_loss


orig_model = AutoModelForCausalLM.from_pretrained(s.model_id, torch_dtype=pt.bfloat16)
param_names = [n for n, p in orig_model.named_parameters()]

center_wmdp, center_disr = _eval_callback(orig_model)
print(f"{center_wmdp=:.4f} {center_disr=:.4f}")

# del model
# pt.cuda.empty_cache()


# %%
def unlearn_percentiles(
    model, h, conf, retain_batches, forget_batches, eval_callback=lambda _: None,
):
    loss_fn = loss_fns[h.unlearning_loss_fn]
    set_seeds(42)
    model.config.use_cache = False

    for n, p in model.named_parameters():
        p.requires_grad = any(pattern in n for pattern in h.modules)

    # ! retain pass
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(**retain_batches[0])
    retain_loss = cross_entropy_loss(output, retain_batches[0])
    retain_loss.backward()
    for p in model.parameters():
        p.retain_acc = p.grad

    # ! forget pass
    model.zero_grad(set_to_none=True)
    output = model(**forget_batches[0])
    forget_loss = loss_fn(output, forget_batches[0], 0)
    forget_loss.backward()

    for p in model.parameters():
        if not p.requires_grad:
            continue

        # ! limit percentiles
        if h.percentile is not None:
            abs_vals = p.grad.flatten().abs()
            k = int(len(abs_vals) * h.percentile)
            cutoff = abs_vals.kthvalue(k).values.item()
            mask = p.grad.abs() > cutoff
            p.grad *= mask

        # ! masking
        if h.use_masking:
            mask = p.retain_acc.sign() == p.grad.sign()
            p.grad *= mask

    # ! unlearning loop
    for loop_num in range(conf.unlearning_epochs):
        # eval_callback(model)

        for p in model.parameters():
            if not p.requires_grad:
                continue
            # ! update weights
            p.data -= h.unlearning_rate * p.grad

    return model


_lr = 8e-2
wmdp_accs = {}
disr_losses = {}
for param_name in param_names:
    if "layernorm" in param_name:
        continue
    h = OmegaConf.create(
        dict(
            unlearning_loss_fn="neg_cross_entropy",
            # unlearning_loss_fn="neg_entropy",
            # unlearning_loss_fn="correct_logit_minus_avg",
            use_masking=True,
            unlearning_rate=_lr,
            modules=[param_name],
            percentile=None,
        )
    )
    s.unlearning_epochs = 1

    model2 = unlearn_percentiles(
        deepcopy(orig_model),
        h,
        s,
        retain_batches,
        forget_batches,
    )
    wmdp_accs[param_name], disr_losses[param_name] = _eval_callback(model2)
    print(f"{param_name=} {wmdp_accs[param_name]=:.4f} {disr_losses[param_name]=:.4f}")
    del model2
    pt.cuda.empty_cache()


# %%

_d_ref = 0.001
_w_ref = -0.10

# Parse labels and create color mapping
layer_module_colors = []
for param_name in param_names:
    if "layers." not in param_name:
        continue
    if "layernorm" in param_name:
        continue
    w = float(wmdp_accs[param_name] - center_wmdp)
    d = float(disr_losses[param_name] - center_disr)

    # Parse layer and module
    parts = param_name.split("layers.")[1].split(".")
    layer = int(parts[0])
    param_name = ".".join(parts[1:]).replace(".weight", "")
    # Create color
    color = (
        min(1.0, max(0.0, d / _d_ref)),  # red
        min(1.0, max(0.0, w / _w_ref)),  # green
        0.0,  # blue
    )
    layer_module_colors.append((layer, param_name, color))

# Get unique sorted modules and layers
unique_modules = sorted(set(module for _, module, _ in layer_module_colors))
layer_nums = sorted(set(layer for layer, _, _ in layer_module_colors))

# Create color lookup dictionary
color_matrix = {layer: {} for layer in layer_nums}
for layer, param_name, color in layer_module_colors:
    color_matrix[layer][param_name] = color

# Create the visualization
fig, ax = plt.subplots(figsize=(4.5, 10))
ax.set_axis_off()

# Calculate grid dimensions
cell_height = 1
cell_width = 1.5
height = len(layer_nums) * cell_height
width = len(unique_modules) * cell_width

# Add padding around the plot
plt.subplots_adjust(left=0.2, right=0.8, top=0.95, bottom=0.15)

# Add title with question info
title_text = f"Question {question_index}:\nModel: {s.model_id}\n{target_question}"
ax.text(
    width / 2,
    height + 0.5,
    title_text,
    ha="center",
    va="bottom",
    wrap=True,
    # fontsize=8,  # Smaller font size for better wrapping
)

# Draw cells
for i, layer in enumerate(layer_nums):
    for j, param_name in enumerate(unique_modules):
        color = color_matrix[layer][param_name]
        rect = plt.Rectangle(
            (j * cell_width, (len(layer_nums) - 1 - i) * cell_height),
            cell_width - 0.1,
            cell_height - 0.1,
            facecolor=color,
        )
        ax.add_patch(rect)

    # Add layer number on the left
    ax.text(
        -0.3,
        (len(layer_nums) - 1 - i) * cell_height + cell_height / 2,
        f"Layer {layer}",
        ha="right",
        va="center",
    )

# Add module labels at the bottom
for j, param_name in enumerate(unique_modules):
    ax.text(
        j * cell_width + cell_width / 2,
        -0.3,  # Increased spacing below the grid
        param_name,
        ha="center",  # Changed to center
        va="top",
        rotation=90,
    )

# Set the plot limits with more padding
plt.xlim(-1, width + 0.5)
plt.ylim(-2, height + 2.5)  # More space for bottom labels and title

# Save the plot with a tight layout and explicit bbox
os.makedirs(f"../plots/single_question_per_module", exist_ok=True)
name = f"question={question_index}_masking={h.use_masking}"
plt.savefig(
    f"../plots/single_question_per_module/{name}.pdf",
    bbox_inches='tight',
    pad_inches=0.8,
    dpi=300
)

# %% legend

# # Create a new figure for the legend
# fig_legend, ax_legend = plt.subplots(figsize=(4, 4))

# # Create a grid of points
# n_points = 100
# x = np.linspace(-_w_ref, 0, n_points)  # WMDP change
# y = np.linspace(0, _d_ref, n_points)  # Disruption change
# X, Y = np.meshgrid(x, y)

# # Create color array
# colors = np.zeros((n_points, n_points, 3))
# colors[:, :, 0] = Y / _d_ref  # red component
# colors[:, :, 1] = -X / _w_ref  # green component
# # blue stays 0

# # Plot the color map
# ax_legend.imshow(colors, extent=[-_w_ref, 0, 0, _d_ref], origin="lower", aspect="auto")

# # Add labels and title
# ax_legend.set_xlabel("WMDP Accuracy Change")
# ax_legend.set_ylabel("Disruption Loss Change")
# ax_legend.set_title("Color Legend")

# # Add gridlines
# ax_legend.grid(True, color="white", alpha=0.3)

# plt.tight_layout()
