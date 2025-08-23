# %%
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

# group = config_name + "_" + get_conf_hash(config_name)
# group = "cb_target_modules_plot_23.08.2025_2"
group = "cb_target_modules_plot_23.08.2025_3"
# h = "ac6bd2"
# group = config_name + "_" + h

# list(name_to_run.keys())

# %%

gate_proj_runs = [
    "1|['gate_proj']|",
    "10|['.0.mlp.gate_proj.']|",
    "11|['.1.mlp.gate_proj.']|",
    "12|['.2.mlp.gate_proj.']|",
    "13|['.3.mlp.gate_proj.']|",
    "14|['.4.mlp.gate_proj.']|",
    "15|['.5.mlp.gate_proj.']|",
]

up_proj_runs = [
    "2|['up_proj']|",
    "16|['.0.mlp.up_proj.']|",
    "17|['.1.mlp.up_proj.']|",
    "18|['.2.mlp.up_proj.']|",
    "19|['.3.mlp.up_proj.']|",
    "20|['.4.mlp.up_proj.']|",
    "21|['.5.mlp.up_proj.']|",
]

down_proj_runs = [
    "3|['down_proj']|",
    "22|['.0.mlp.down_proj.']|",
    "23|['.1.mlp.down_proj.']|",
    "24|['.2.mlp.down_proj.']|",
    "25|['.3.mlp.down_proj.']|",
    "26|['.4.mlp.down_proj.']|",
    "27|['.5.mlp.down_proj.']|",
]

layer_runs = [
    "0|['gate_proj', 'up_proj', 'down_proj']|",
    "4|['.0.mlp.']|",
    "5|['.1.mlp.']|",
    "6|['.2.mlp.']|",
    "7|['.3.mlp.']|",
    "8|['.4.mlp.']|",
    "9|['.5.mlp.']|",
]


# Function to extract data for a given set of runs
def extract_run_data(run_names, metrics=["wikitext_loss", "forget_acc_t1", "recall_loss"]):
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

project_name = "unlearning|src|main_runner.py"
# get all the runs of the project and this group
runs = wandb.Api().runs(project_name, filters={"group": group})
name_to_run = {run.name: run for run in runs}

# Extract data for all runs
gate_data = extract_run_data(gate_proj_runs)
up_data = extract_run_data(up_proj_runs)
down_data = extract_run_data(down_proj_runs)
layer_data = extract_run_data(layer_runs)

# %%

# Create imshow pixel plot
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

# Prepare data matrix (3 rows x 7 columns)
# Rows: gate, up, down_proj (but you mentioned gate, up, proj - assuming proj means down_proj)
# Columns: 0, 1, 2, 3, 4, 5, plus one general

# Extract recall_loss values for the matrix
recall_matrix = np.zeros((4, 7))

# Gate proj row (row 0)
for i, run_name in enumerate(gate_proj_runs):
    recall_matrix[0, i] = gate_data[run_name][-1]["recall_loss"]

# Up proj row (row 1) 
for i, run_name in enumerate(up_proj_runs):
    recall_matrix[1, i] = up_data[run_name][-1]["recall_loss"]

# Down proj row (row 2)
for i, run_name in enumerate(down_proj_runs):
    recall_matrix[2, i] = down_data[run_name][-1]["recall_loss"]

for i, run_name in enumerate(layer_runs):
    recall_matrix[3, i] = layer_data[run_name][-1]["recall_loss"]


# Create the imshow plot
im = ax.imshow(recall_matrix, cmap='viridis', aspect='auto')

# Set labels
ax.set_xticks(range(7))
ax.set_xticklabels(['0', '1', '2', '3', '4', '5', 'All'])
ax.set_yticks(range(4))
ax.set_yticklabels(['Gate', 'Up', 'Down', 'All'])

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Recall Loss', rotation=270, labelpad=15)

# Add title
ax.set_title('Recall Loss by Target Module and Layer')
ax.set_xlabel('Layer')
ax.set_ylabel('Module Type')

# Add text annotations with values
for i in range(4):
    for j in range(7):
        text = ax.text(j, i, f'{recall_matrix[i, j]:.3f}', 
                      ha="center", va="center", color="black", fontsize=10)

plt.tight_layout()

# # Save the plot
# stem = Path(__file__).stem
# plot_path = repo_root() / f"paper/plots/{stem}_imshow.pdf"
# plt.savefig(plot_path, bbox_inches='tight', dpi=300)
# plt.show()

# %%

project_name = "ret_unlearning|src|main_runner.py"
# get all the runs of the project and this group
runs = wandb.Api().runs(project_name, filters={"group": group})
name_to_run = {run.name: run for run in runs}

# Extract data for all runs
gate_data = extract_run_data(gate_proj_runs)
up_data = extract_run_data(up_proj_runs)
down_data = extract_run_data(down_proj_runs)
layer_data = extract_run_data(layer_runs)

# %%


# Create imshow pixel plot
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

# Prepare data matrix (3 rows x 7 columns)
# Rows: gate, up, down_proj (but you mentioned gate, up, proj - assuming proj means down_proj)
# Columns: 0, 1, 2, 3, 4, 5, plus one general

# Extract recall_loss values for the matrix
recall_matrix = np.zeros((4, 7))

# metric = "recall_loss"
metric = "forget_acc_t1"

# Gate proj row (row 0)
for i, run_name in enumerate(gate_proj_runs):
    recall_matrix[0, i] = gate_data[run_name][-1][metric]

# Up proj row (row 1) 
for i, run_name in enumerate(up_proj_runs):
    recall_matrix[1, i] = up_data[run_name][-1][metric]

# Down proj row (row 2)
for i, run_name in enumerate(down_proj_runs):
    recall_matrix[2, i] = down_data[run_name][-1][metric]

for i, run_name in enumerate(layer_runs):
    recall_matrix[3, i] = layer_data[run_name][-1][metric]


# Create the imshow plot
im = ax.imshow(recall_matrix, cmap='viridis_r', aspect='auto')  # Added _r to invert colormap

# Set labels
ax.set_xticks(range(7))
ax.set_xticklabels(['0', '1', '2', '3', '4', '5', 'All'])
ax.set_yticks(range(4))
ax.set_yticklabels(['Gate', 'Up', 'Down', 'All'])

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label(metric, rotation=270, labelpad=15)

# Add title
ax.set_title(f'{metric} by Target Module and Layer')
ax.set_xlabel('Layer')
ax.set_ylabel('Module Type')

# Add text annotations with values
for i in range(4):
    for j in range(7):
        text = ax.text(j, i, f'{recall_matrix[i, j]:.3f}', 
                      ha="center", va="center", color="black", fontsize=10)

plt.tight_layout()

# # Save the plot
# stem = Path(__file__).stem
# plot_path = repo_root() / f"paper/plots/{stem}_imshow.pdf"
# plt.savefig(plot_path, bbox_inches='tight', dpi=300)
# plt.show()