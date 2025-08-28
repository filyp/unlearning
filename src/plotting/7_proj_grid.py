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

group = "proj_num_grid_search_b73dc7"
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

# remove divergent runs
del ret_data["0||act_proj_num=0_grad_proj_num=0"]
del ret_data["0||act_proj_num=1_grad_proj_num=0"]
# they needed a smaller LR, so were rerun manually with:
# sbatch ~/unlearning/example_job.sh "python src/main_runner.py --config-name=proj_num_grid_search --exp-num=0 --group-name=proj_num_grid_search_b73dc7 act_proj_num=0 grad_proj_num=0 max_norm=0.005"
# sbatch ~/unlearning/example_job.sh "python src/main_runner.py --config-name=proj_num_grid_search --exp-num=0 --group-name=proj_num_grid_search_b73dc7 act_proj_num=1 grad_proj_num=0 max_norm=0.005"

# also remove the fast ones, because they seem to be starkly different!
for k in list(ret_data.keys()):
    if "max_norm=0.05" in k:
        del ret_data[k]

# %%
# Prepare data for 2D grid plot
act_proj_values = [0, 1, 3, 6, 9, 12, 15, 20, 25, 30]
grad_proj_values = [0, 1, 2, 3, 4, 6, 9, 12, 15, 18, 21]
grid_data = np.full((len(grad_proj_values), len(act_proj_values)), np.nan)
grid_text = np.full((len(grad_proj_values), len(act_proj_values)), "", dtype=object)

for run_name, v in ret_data.items():
# for run_name, v in data.items():
    last_acc = v[-1]["forget_acc_t1"]
    # last_acc = v[0]["forget_acc_t1"]
    # last_acc = v[0]["retain_loss"]
    # from strings like 0||act_proj_num=3_grad_proj_num=12 extract proj_nums
    act_proj_nums = int(re.findall(r"act_proj_num=(\d+)", run_name)[0])
    grad_proj_nums = int(re.findall(r"grad_proj_num=(\d+)", run_name)[0])

    # Find indices in the grid
    try:
        act_idx = act_proj_values.index(act_proj_nums)
        grad_idx = grad_proj_values.index(grad_proj_nums)

        # Store the accuracy value (as percentage)
        grid_data[grad_idx, act_idx] = last_acc * 100  # Convert to percentage
        # grid_text[grad_idx, act_idx] = f"{last_acc * 100:.1f}"
        grid_text[grad_idx, act_idx] = f"{last_acc * 100:.0f}"
    except ValueError:
        print(
            f"Warning: proj_num values {act_proj_nums}, {grad_proj_nums} not in expected range"
        )

# Create the plot
# plt.figure(figsize=(5.5, 5))
plt.figure(figsize=(2.75, 2.75))

# Create a custom colormap where green is lower values (better)
# Using RdYlGn_r (reversed Red-Yellow-Green) so green represents lower values
mask = np.isnan(grid_data)
valid_data = grid_data[~mask]
vmin, vmax = np.min(valid_data), np.max(valid_data)
im = plt.imshow(grid_data, cmap="RdYlGn_r", aspect="equal", vmin=vmin, vmax=vmax)

# Add text annotations
for i in range(len(grad_proj_values)):
    for j in range(len(act_proj_values)):
        if not mask[i, j]:
            plt.text(
                j,
                i,
                grid_text[i, j],
                ha="center",
                va="center",
                color="black",
                # fontweight="bold",
                fontsize=10,
            )

# Set ticks and labels
plt.xticks(range(len(act_proj_values)), act_proj_values)
plt.yticks(range(len(grad_proj_values)), grad_proj_values)
plt.xlabel("Act Proj Num")
plt.ylabel("Grad Proj Num")
# plt.title("")

# # Move x-axis to top
# ax = plt.gca()
# ax.xaxis.set_ticks_position('top')
# ax.xaxis.set_label_position('top')

# Add colorbar
cbar = plt.colorbar(im, shrink=0.8)
cbar.set_label("WMDP-Cyber Accuracy (%)", rotation=270, labelpad=15)


plt.tight_layout()
stem = Path(__file__).stem
plot_path = repo_root() / f"paper/plots/{stem}.pdf"
plt.savefig(plot_path, bbox_inches='tight', dpi=300)
plt.show()

# %%
