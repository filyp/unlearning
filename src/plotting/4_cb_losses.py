# %%
from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wandb
from utils.git_and_reproducibility import get_conf_hash, repo_root

# for the paper
plt.style.use("default")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10

project_name = "unlearning|src|main_runner.py"
# group = config_name + "_" + get_conf_hash(config_name)
group = "cb_22.08.2025_2"
# h = "ac6bd2"
# group = config_name + "_" + h

# get all the runs of the project and this group
runs = wandb.Api().runs(project_name, filters={"group": group})
name_to_run = {run.name: run for run in runs}

# %%

run_names = [
    "2|0.15_1.0_0.001_[6, 7, 8]_1.0_dot|",
    "3|0.22_1.0_0.001_[6, 7, 8]_1.0_norm|",
    "4|0.4_1.0_0.001_[6, 7, 8]_1.0_cossim|",
]

# %%


# Function to extract data for a given set of runs
def extract_run_data(run_names, metrics=["wikitext_loss", "forget_acc_t1", "act_norm"]):
    all_data = {}
    for run_name in run_names:
        run = name_to_run[run_name]
        history = run.history()

        data = []
        for _, row in history.iterrows():
            # if all(metric in row for metric in metrics):
            # if all(pd.notna(row[metric]) for metric in metrics):
            data_point = {"run_name": run_name}
            for metric in metrics:
                data_point[metric] = row[metric]
            data.append(data_point)
        all_data[run_name] = data
    return all_data


# Extract data for all runs
data = extract_run_data(run_names)


# %%

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))


# Define colors and labels for each run
run_labels = {
    "2|0.15_1.0_0.001_[6, 7, 8]_1.0_dot|": "dot product",
    "3|0.22_1.0_0.001_[6, 7, 8]_1.0_norm|": "norm",
    "4|0.4_1.0_0.001_[6, 7, 8]_1.0_cossim|": "cossim",
}

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green

# First plot: wikitext_loss vs forget_acc_t1
ax1 = axes[0]
for i, run_name in enumerate(run_names):
    run_data = data[run_name]
    df = pd.DataFrame(run_data)

    # Filter out NaN values
    valid_data = df.dropna(subset=["wikitext_loss", "forget_acc_t1"])

    if not valid_data.empty:
        ax1.plot(
            valid_data["forget_acc_t1"],
            valid_data["wikitext_loss"],
            color=colors[i],
            label=run_labels[run_name],
        )

ax1.annotate(
    "",
    xy=(0.27, valid_data["wikitext_loss"].iloc[0]),  # arrow base
    xytext=(valid_data["forget_acc_t1"].iloc[0], valid_data["wikitext_loss"].iloc[0]),  # arrow tip
    arrowprops=dict(
        arrowstyle="->",
        color="black",
        linewidth=1
    )
)
# Add invisible line for legend entry
ax1.plot([], [], color="black", linewidth=1, label="ideal", marker='>', markersize=4, linestyle='-')


ax1.set_xlabel("WMDP-Cyber Accuracy")
ax1.set_ylabel("Wikitext Loss")
# ax1.grid(True, alpha=0.3)

# Second plot: act_norm vs forget_acc_t1
ax2 = axes[1]
for i, run_name in enumerate(run_names):
    run_data = data[run_name]
    df = pd.DataFrame(run_data)

    # Filter out NaN values
    valid_data = df.dropna(subset=["act_norm", "forget_acc_t1"])

    if not valid_data.empty:
        ax2.plot(
            valid_data["forget_acc_t1"],
            valid_data["act_norm"],
            color=colors[i],
            # label=run_labels[run_name],
        )

ax2.set_xlabel("WMDP-Cyber Accuracy")
ax2.set_ylabel("Activation Norm")
# ax2.grid(True, alpha=0.3)

# Add legend at the bottom of the figure
fig.legend(
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.02),
    fontsize=10,
)

# plt.tight_layout()
# plt.subplots_adjust(bottom=0.1, left=0.08)  # Make room for legend and row labels
plt.subplots_adjust(bottom=0.32, hspace=0.0, wspace=0.36)
# plt.show()


stem = Path(__file__).stem
plot_path = repo_root() / f"paper/plots/{stem}.pdf"
plt.savefig(plot_path, bbox_inches=None, dpi=300)

# %%
