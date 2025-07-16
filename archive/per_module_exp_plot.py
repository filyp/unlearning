# %%
import os
from collections import defaultdict
from copy import copy

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.git_and_reproducibility import repo_root

# use dark theme
plt.style.use("dark_background")

# Load a single question's data
# type_ = "no_mask_False"
# type_ = "disruption_mask_False"
type_ = "disruption_mask_True"
data_dir = repo_root() / "data" / "per_module_exp" / type_
question_files = list(data_dir.glob("*.json"))  # Get first question file

# # compute avg
# wmdp_accs_all = defaultdict(list)
# disr_losses_all = defaultdict(list)
# for question_file in question_files:
#     with open(question_file) as f:
#         data = json.load(f)
#     wmdp_accs = data["wmdp_accs"]
#     disr_losses = data["disr_losses"]
#     center_wmdp = data["orig_wmdp_acc"]
#     center_disr = data["orig_disr_loss"]
#     for param_name in wmdp_accs:
#         if "layers." not in param_name or "layernorm" in param_name:
#             continue
#         wmdp_accs_all[param_name].append(wmdp_accs[param_name] - center_wmdp)
#         disr_losses_all[param_name].append(disr_losses[param_name] - center_disr)
# wmdp_accs = {}
# disr_losses = {}
# for param_name in wmdp_accs_all.keys():
#     wmdp_accs[param_name] = np.mean(wmdp_accs_all[param_name])
#     disr_losses[param_name] = np.mean(disr_losses_all[param_name])
# center_wmdp = 0
# center_disr = 0
# question_text = f"Average of all\n{type_}"


question_file = question_files[23]
with open(question_file) as f:
    data = json.load(f)
# Extract data
wmdp_accs = data["wmdp_accs"]
disr_losses = data["disr_losses"]
center_wmdp = data["orig_wmdp_acc"]
center_disr = data["orig_disr_loss"]
question_text = data["mcq"]["question"]


total_loss = sum(disr_losses.values())
print(f"Total loss: {total_loss}")
for param_name, loss in sorted(
    list(disr_losses.items()), key=lambda x: x[1], reverse=True
)[:5]:
    loss_perc = loss / total_loss
    print(f"{loss_perc:6.2%}: {param_name}")


# Constants from original code
_d_ref = 0.003
_w_ref = -0.03

# Parse labels and create color mapping
layer_module_colors = []
for param_name in wmdp_accs:
    if "layers." not in param_name or "layernorm" in param_name:
        continue

    w = float(wmdp_accs[param_name] - center_wmdp)
    d = float(disr_losses[param_name] - center_disr)

    # Parse layer and module
    parts = param_name.split("layers.")[1].split(".")
    layer = int(parts[0])
    param_name = ".".join(parts[1:]).replace(".weight", "")

    # Create color
    d = max(0.0, d / _d_ref)
    w = max(0.0, w / _w_ref)
    max_val = max(d, w)
    if max_val > 1.0:
        # rescale
        d /= max_val
        w /= max_val
    color = (
        min(1.0, d),  # red
        min(1.0, w),  # green
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

# Create the visualization - now with swapped dimensions
fig, ax = plt.subplots(figsize=(5, 4))
ax.set_axis_off()

# Calculate grid dimensions
cell_height = 1
cell_width = 1.5
width = len(layer_nums) * cell_width
height = len(unique_modules) * cell_height

# Add padding around the plot
plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.2)

# Add title with question
title_text = f"Question: {question_text[:100]}..."
ax.text(width / 2, height + 0.5, title_text, ha="center", va="bottom", wrap=True)

# Draw cells
for i, module in enumerate(unique_modules):
    for j, layer in enumerate(layer_nums):
        color = color_matrix[layer][module]
        rect = plt.Rectangle(
            (j * cell_width, (len(unique_modules) - 1 - i) * cell_height),
            cell_width - 0.1,
            cell_height - 0.1,
            facecolor=color,
        )
        ax.add_patch(rect)

    # Add module names on the left
    ax.text(
        -0.3,
        (len(unique_modules) - 1 - i) * cell_height + cell_height / 2,
        module,
        ha="right",
        va="center",
    )

# Add layer labels at the bottom
for j, layer in enumerate(layer_nums):
    ax.text(
        j * cell_width + cell_width / 2,
        -0.3,
        f"{layer}",
        ha="center",
        va="top",
    )

# Set the plot limits
plt.xlim(-1, width + 0.5)
plt.ylim(-2, height + 2.5)

plt.show()

# # Save the plot with a tight layout and explicit bbox
# os.makedirs(f"../plots/single_question_per_module", exist_ok=True)
# name = f"question={question_index}_masking={conf.masking}"
# plt.savefig(
#     f"../plots/single_question_per_module/{name}.pdf",
#     bbox_inches="tight",
#     pad_inches=0.8,
#     dpi=300,
# )

# %% legend


def plot_legend(w_ref, d_ref):
    # Create a new figure for the legend
    fig_legend, ax_legend = plt.subplots(figsize=(4, 4))

    # Create a grid of points
    n_points = 100
    x = np.linspace(-_w_ref, 0, n_points)  # WMDP change
    y = np.linspace(0, _d_ref, n_points)  # Disruption change
    X, Y = np.meshgrid(x, y)

    # Create color array
    colors = np.zeros((n_points, n_points, 3))
    colors[:, :, 0] = Y / _d_ref  # red component
    colors[:, :, 1] = -X / _w_ref  # green component
    # blue stays 0

    # Plot the color map
    ax_legend.imshow(
        colors, extent=[-_w_ref, 0, 0, _d_ref], origin="lower", aspect="auto"
    )

    # Add labels and title
    ax_legend.set_xlabel("WMDP Accuracy Change")
    ax_legend.set_ylabel("Disruption Loss Change")
    ax_legend.set_title("Color Legend")

    # Add gridlines
    ax_legend.grid(True, color="white", alpha=0.3)

    plt.tight_layout()


plot_legend(_w_ref, _d_ref)
