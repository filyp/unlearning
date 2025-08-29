# %%
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wandb
from utils.git_and_reproducibility import get_conf_hash, repo_root

# for the paper
plt.style.use("default")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10


# Function to extract data for a given set of runs
def extract_run_data(
    run_names, metrics=["wikitext_loss", "forget_acc_t1", "recall_loss", "retain_loss"]
):
    all_data = {}
    for run_name in run_names:
        run = name_to_run[run_name]
        history = run.history()

        data = []
        for _, row in history.iterrows():
            # if all(metric in row for metric in metrics):
            # if all(pd.notna(row[metric]) for metric in metrics):
            data_point = {}
            for metric in metrics:
                data_point[metric] = row[metric]
            data.append(data_point)
        all_data[run_name] = data
    return all_data


# %%

group = "layer_search_66b9a6"
api = wandb.Api(timeout=3600)

# project_name = "unlearning|src|main_runner.py"
# # get all the runs of the project and this group
# runs = api.runs(project_name, filters={"group": group})
# name_to_run = {run.name: run for run in runs}
# # Extract data for all runs
# data = extract_run_data(list(name_to_run.keys()))
# del data["0||act_proj_num=0_grad_proj_num=0"]
# del data["0||act_proj_num=1_grad_proj_num=0"]

ret_project_name = "ret_unlearning|src|main_runner.py"
# get all the runs of the project and this group
ret_runs = api.runs(ret_project_name, filters={"group": group})
name_to_run = {run.name: run for run in ret_runs}
# Extract data for all runs
ret_data = extract_run_data(list(name_to_run.keys()))

print(ret_data.keys())

# %%
# Prepare data for 2D grid plot
layer_ranges = ["[0, 6]", "[6, 12]", "[12, 18]", "[18, 24]", "[24, 30]"]
methods = ["circuit_breaker_GA", "circuit_breaker|", "mlp_confuse"]
grid_data = np.full((len(methods), len(layer_ranges)), np.nan)
grid_text = np.full((len(methods), len(layer_ranges)), "", dtype=object)

method_to_row = {
    "circuit_breaker_GA": 0,
    "circuit_breaker|": 1, 
    "mlp_confuse": 2
}

for run_name, v in ret_data.items():
    last_acc = v[-1]["forget_acc_t1"]
    
    # Parse run name to get layer range and method
    parts = run_name.split("|")
    layer_range = parts[1].split("_")[0]
    if "circuit_breaker_GA" in parts[1]:
        method = "circuit_breaker_GA"
    elif "circuit_breaker" in parts[1]:
        method = "circuit_breaker|"
    elif "mlp_confuse" in parts[1]:
        method = "mlp_confuse"
    else:
        continue

    # Find indices in the grid
    try:
        col_idx = layer_ranges.index(layer_range)
        row_idx = method_to_row[method]

        # Store the accuracy value (as percentage)
        grid_data[row_idx, col_idx] = last_acc * 100  # Convert to percentage
        grid_text[row_idx, col_idx] = f"{last_acc * 100:.1f}"
    except ValueError:
        print(f"Warning: values {layer_range}, {method} not in expected range")

# Create the plot
plt.figure(figsize=(5.5, 1.3))

# Create a custom colormap where green is lower values (better)
mask = np.isnan(grid_data)
valid_data = grid_data[~mask]
# vmin, vmax = np.min(valid_data), np.max(valid_data)
# Changed aspect to 2.0 to make cells wider
im = plt.imshow(grid_data, cmap="RdYlGn_r", aspect=0.4, vmin=37, vmax=57.5)

# Add text annotations
for i in range(len(methods)):
    for j in range(len(layer_ranges)):
        if not mask[i, j]:
            plt.text(
                j,
                i,
                grid_text[i, j],
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

# Set ticks and labels
plt.xticks(range(len(layer_ranges)), layer_ranges)
plt.yticks(range(len(methods)), ["circuit breaking", "CIR + circuit breaking", "CIR + MLP breaking"])
plt.xlabel("Layer Range")
plt.ylabel("Method")

# Move x-axis to top
ax = plt.gca()
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# # Add colorbar
# cbar = plt.colorbar(im, shrink=0.8)
# cbar.set_label("WMDP-Cyber Accuracy (%)", rotation=270, labelpad=15)

plt.tight_layout()
stem = Path(__file__).stem
plot_path = repo_root() / f"paper/plots/{stem}.pdf"
# bbox_inches=None is needed to actually produce 5.5 inch width
plt.savefig(plot_path, bbox_inches=None, dpi=300)
plt.show()

# %%
