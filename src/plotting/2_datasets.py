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

config_name = "datasets"
project_name = "unlearning|src|main_runner.py"
group = config_name + "_" + get_conf_hash(config_name)
# h = "ac6bd2"
# group = config_name + "_" + h

# get all the runs of the project and this group
runs = wandb.Api().runs(project_name, filters={"group": group})
name_to_run = {run.name: run for run in runs}

max_wikitext_loss = 2.91267 * 1.01

# %%

bio_cir_runs = [
    "CIR_7_7_correct_logit_False_0.001_3|dataset=wmdp_bio_deebs",
    "CIR_7_7_correct_logit_False_0.001_3|dataset=wmdp_bio",
    "CIR_7_7_correct_logit_True_0.001_3|dataset=wmdp_bio",
]
bio_ga_runs = [
    "GA_neg_cross_entropy_False_3e-06_3|dataset=wmdp_bio_deebs",
    "GA_neg_cross_entropy_False_3e-06_3|dataset=wmdp_bio",
    "GA_neg_cross_entropy_True_3e-06_3|dataset=wmdp_bio",
]
cyber_cir_runs = [
    "CIR_7_7_correct_logit_False_0.001_3|dataset=wmdp_cyber_deebs",
    "CIR_7_7_correct_logit_False_0.001_3|dataset=wmdp_cyber",
    "CIR_7_7_correct_logit_True_0.001_3|dataset=wmdp_cyber",
]
cyber_ga_runs = [
    "GA_neg_cross_entropy_False_3e-06_3|dataset=wmdp_cyber_deebs",
    "GA_neg_cross_entropy_False_3e-06_3|dataset=wmdp_cyber",
    "GA_neg_cross_entropy_True_3e-06_3|dataset=wmdp_cyber",
]

# %%


# Function to extract data for a given set of runs
def extract_run_data(run_names, metrics=["wikitext_loss", "forget_loss", "forget_acc"]):
    data = []
    for run_name in run_names:
        if run_name in name_to_run:
            run = name_to_run[run_name]
            history = run.history()

            for _, row in history.iterrows():
                if all(metric in row for metric in metrics):
                    if all(pd.notna(row[metric]) for metric in metrics):
                        data_point = {"run_name": run_name}
                        for metric in metrics:
                            data_point[metric] = row[metric]
                        data.append(data_point)
    return data


# Extract data for all runs
bio_cir_data = extract_run_data(bio_cir_runs)
bio_ga_data = extract_run_data(bio_ga_runs)
cyber_cir_data = extract_run_data(cyber_cir_runs)
cyber_ga_data = extract_run_data(cyber_ga_runs)


# %%


# Define colors based on dataset type
def get_color(run_name):
    if "deebs" in run_name:
        return "red"  # unlearning corpus from Deeb & Roger
    elif "True" in run_name:
        return "green"  # our corpus + unlearning only the answer
    else:
        return "gold"  # our corpus with answers at the end (using gold instead of yellow for better visibility)


# Define labels for legend
def get_label(run_name):
    if "deebs" in run_name:
        return "unlearning corpus from Deeb & Roger"
    elif "True" in run_name:
        return "our corpus + unlearning only the answer"
    else:
        return "our corpus with answers at the end"


# Create figure with subplots
fig, axes = plt.subplots(2, 4, figsize=(5.5, 4))


# Function to plot data on a given axis
def plot_data(ax, data, run_names, y_metric, invert_y=False):
    for run_name in run_names:
        run_data = [d for d in data if d["run_name"] == run_name]
        if run_data:
            x_vals = [d["wikitext_loss"] for d in run_data]
            y_vals = [d[y_metric] for d in run_data]

            ax.plot(
                x_vals,
                y_vals,
                color=get_color(run_name),
                linestyle="-",
                label=get_label(run_name),
                alpha=0.8,
                linewidth=1,
            )

    # ax.set_xlabel("Wikitext Loss")
    # ax.set_ylabel(y_metric.replace("_", " ").title())
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="y", rotation=90)
    ax.tick_params(axis="x")

    if invert_y:
        ax.invert_yaxis()


# Plot BIO row (row 0)
plot_data(axes[0, 0], bio_cir_data, bio_cir_runs, "forget_loss")
plot_data(axes[0, 1], bio_ga_data, bio_ga_runs, "forget_loss")
plot_data(axes[0, 2], bio_cir_data, bio_cir_runs, "forget_acc", invert_y=True)
plot_data(axes[0, 3], bio_ga_data, bio_ga_runs, "forget_acc", invert_y=True)

# Plot CYBER row (row 1)
axes[1, 0].set_xlabel("Wikitext Loss")
axes[1, 1].set_xlabel("Wikitext Loss")
axes[1, 2].set_xlabel("Wikitext Loss")
axes[1, 3].set_xlabel("Wikitext Loss")
plot_data(axes[1, 0], cyber_cir_data, cyber_cir_runs, "forget_loss")
plot_data(axes[1, 1], cyber_ga_data, cyber_ga_runs, "forget_loss")
plot_data(axes[1, 2], cyber_cir_data, cyber_cir_runs, "forget_acc", invert_y=True)
plot_data(axes[1, 3], cyber_ga_data, cyber_ga_runs, "forget_acc", invert_y=True)

# Set titles for columns
axes[0, 0].set_title("CIR")
axes[0, 1].set_title("GA")
axes[0, 2].set_title("CIR")
axes[0, 3].set_title("GA")

# Add group labels above the column titles
fig.text(0.30, 0.97, "Forget Loss", ha="center", fontsize=10)
fig.text(0.72, 0.97, "WMDP Accuracy", ha="center", fontsize=10)

# Add row labels
fig.text(
    0.04,
    0.75,
    "BIO",
    rotation=90,
    verticalalignment="center",
    fontsize=10,
    # fontweight="bold",
)
fig.text(
    0.04,
    0.45,
    "CYBER",
    rotation=90,
    verticalalignment="center",
    fontsize=10,
    # fontweight="bold",
)

# Match x and y limits for corresponding CIR and GA plots
# For forget_loss plots
for row in range(2):
    # Get x and y limits for CIR and GA forget_loss plots
    cir_xlim = axes[row, 0].get_xlim()
    cir_ylim = axes[row, 0].get_ylim()
    ga_xlim = axes[row, 1].get_xlim()
    ga_ylim = axes[row, 1].get_ylim()

    # Set common limits for forget_loss plots
    common_xlim = (min(cir_xlim[0], ga_xlim[0]), max_wikitext_loss) 
     # max(cir_xlim[1], ga_xlim[1]))
    # common_ylim = (min(cir_ylim[0], ga_ylim[0]), max(cir_ylim[1], ga_ylim[1]))
    common_ylim = (min(cir_ylim[0], ga_ylim[0]), 13)

    axes[row, 0].set_xlim(common_xlim)
    axes[row, 0].set_ylim(common_ylim)
    axes[row, 1].set_xlim(common_xlim)
    axes[row, 1].set_ylim(common_ylim)

    # Get x and y limits for CIR and GA forget_acc plots
    cir_xlim = axes[row, 2].get_xlim()
    cir_ylim = axes[row, 2].get_ylim()
    ga_xlim = axes[row, 3].get_xlim()
    ga_ylim = axes[row, 3].get_ylim()

    # Set common limits for forget_acc plots
    common_xlim = (min(cir_xlim[0], ga_xlim[0]), max_wikitext_loss)
    # max(cir_xlim[1], ga_xlim[1]))
    # Calculate the actual min and max values
    # y_min = min(cir_ylim[0], cir_ylim[1], ga_ylim[0], ga_ylim[1])
    # y_max = max(cir_ylim[0], cir_ylim[1], ga_ylim[0], ga_ylim[1])
    y_min = 0.4
    y_max = 0.65

    axes[row, 2].set_xlim(common_xlim)
    axes[row, 2].set_ylim(y_max, y_min)  # Set max first for inverted axis
    axes[row, 3].set_xlim(common_xlim)
    axes[row, 3].set_ylim(y_max, y_min)  # Set max first for inverted axis

# Create a single legend at the bottom
# Get unique labels from one of the plots
handles, labels = axes[0, 0].get_legend_handles_labels()
unique_labels = []
unique_handles = []
seen_labels = set()

for handle, label in zip(handles, labels):
    if label not in seen_labels:
        seen_labels.add(label)
        unique_labels.append(label)
        unique_handles.append(handle)

# Add legend at the bottom of the figure
fig.legend(
    unique_handles,
    unique_labels,
    loc="lower center",
    # ncol=3,
    # bbox_to_anchor=(0.5, -0.05),
    fontsize=10,
)

# plt.tight_layout()
# plt.subplots_adjust(bottom=0.1, left=0.08)  # Make room for legend and row labels
plt.subplots_adjust(bottom=0.32, hspace=0.3, wspace=0.42)
# plt.show()

stem = Path(__file__).stem
plot_path = repo_root() / f"paper/plots/{stem}.pdf"
plt.savefig(plot_path, bbox_inches=None, dpi=300)

# %%
