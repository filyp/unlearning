# %%
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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
# Create dual plot: wikitext loss vs forget accuracy, and steps vs forget accuracy
def dual_plot(data, ret_data, id_to_label, dataset, num_rel_epochs=60, big_disr_perc=1):
    init_wikitext_loss = list(data.values())[0][0]["wikitext_loss"]
    small_thresh = init_wikitext_loss * 1.001
    big_thresh = init_wikitext_loss * (1 + big_disr_perc / 100)
    init_acc = list(data.values())[0][0]["forget_acc_t1"]

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.75))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 3.05))

    # Process data separately to avoid key conflicts
    for run_name, run_data in data.items():
        # Only use runs where IDs are defined
        id_ = int(run_name.split("|")[0])
        if id_ not in id_to_label:
            continue

        label = id_to_label[id_]
        print(label, run_name)
        wikitext = [x["wikitext_loss"] for x in run_data]
        accs = [x["forget_acc_t1"] * 100 for x in run_data]  # Convert to percent

        color = (
            "green"
            if "CIR" in label
            else ("purple" if "circuit breakers" in label else "orange")
        )
        style = "--" if "handicapped" in label else "-"

        # ax1.plot(wikitext[:-1], accs[:-1], color=color, linestyle=style)  # stops early
        ax1.plot(wikitext, accs, color=color, linestyle=style)


    # Process data separately to avoid key conflicts
    for run_name, run_data in ret_data.items():
        # Only use runs where IDs are defined
        id_ = int(run_name.split("|")[0])
        if id_ not in id_to_label:
            continue

        label = id_to_label[id_]
        # print(label)
        accs = [x["forget_acc_t1"] * 100 for x in run_data]  # Convert to percent

        color = (
            "green"
            if "CIR" in label
            else ("purple" if "circuit breakers" in label else "orange")
        )
        style = "--" if "handicapped" in label else "-"

        ax2.plot(accs[:num_rel_epochs], color=color, linestyle=style)
        # print(accs[num_rel_epochs], label)

    # set the same y axis range for both plots
    y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # Add titles with more padding
    ax1.set_title("Unlearning", pad=17)
    ax2.set_title("Relearning", pad=17)

    # Add axis labels
    ax1.set_xlabel("Wikitext Loss")
    ax1.set_ylabel(f"WMDP-{dataset.capitalize()} Accuracy (%)")
    ax2.set_xlabel("Epochs")

    # ax1.axvline(x=init_wikitext_loss, color="black", linestyle="-", alpha=0.2)
    ax1.text(small_thresh - 0.002, ax1.get_ylim()[1], "0.1% disr.", ha="left", va="bottom")
    ax1.axvline(x=small_thresh, color="black", linestyle="-", alpha=0.2)
    ax1.text(big_thresh - 0.002, ax1.get_ylim()[1], f"{big_disr_perc}% disr.", ha="left", va="bottom")
    ax1.axvline(x=big_thresh, color="black", linestyle="-", alpha=0.2)


    ax1.axhline(y=init_acc * 100, color="black", linestyle="-", alpha=0.2)
    ax1.axhline(y=25, color="black", linestyle="-", alpha=0.2)  # Convert to percent

    ax2.axhline(y=init_acc * 100, color="black", linestyle="-", alpha=0.2)
    # ax2.axhline(y=25, color="black", linestyle="-", alpha=0.2)  # Convert to percent

    ax1.text(ax1.get_xlim()[1] - 0.004, 25.7, "random level", ha="right", va="bottom")
    # ax2.text(ax2.get_xlim()[1] - 3, 25.7, "random level", ha="right", va="bottom")


    # ideal trajectory
    ax1.annotate(
        "",
        xy=(init_wikitext_loss, 24),  # arrow tip
        xytext=(init_wikitext_loss, init_acc * 100),  # arrow base
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            linewidth=1.5
        ),
    )
    ax2.annotate(
        "",
        xy=(num_rel_epochs, 25),  # arrow tip
        xytext=(0, 25),  # arrow base
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            linewidth=1.5
        ),
    )
    # # Add invisible line for legend entry
    # ax1.plot([], [], color="black", linewidth=1, label="ideal")

    # Create legend handles manually
    legend_elements = [
        Line2D([0], [0], color="green", label="CIR", linestyle="--"),
        Line2D([0], [0], color="green", label="CIR (less disruption)", linestyle="-"),
        Line2D([0], [0], color="purple", label="circuit breakers", linestyle="--"),
        Line2D([0], [0], color="black", label="ideal", marker='>', markersize=4, linestyle='-'),
        Line2D([0], [0], color="orange", label="gradient diff", linestyle="--"),
        # Line2D([0], [0], color="black", label="ideal"),
    ]

    # Add legend at the bottom with 3 columns
    # fig.legend(bbox_to_anchor=(0.5, -0.23), loc="center", ncol=3)
    # fig.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.27), loc="center", ncol=3)
    fig.legend(handles=legend_elements, bbox_to_anchor=(0.5, 0.10), loc="center", ncol=3)
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.37)
    plt.subplots_adjust(bottom=0.36)

    # ax1.set_yticks([25, 50])
    # ax2.set_yticks([25, 50])

    # Save the plot
    stem = Path(__file__).stem
    plot_path = repo_root() / f"paper/plots/{stem}_{dataset}_{big_disr_perc}.pdf"
    plt.savefig(plot_path, bbox_inches=None, dpi=300)
    plt.show()


# %%

group = "main_comparison_llama_29.08.25_2"
api = wandb.Api(timeout=3600)

project_name = "unlearning|src|main_runner.py"
# get all the runs of the project and this group
runs = api.runs(project_name, filters={"group": group})
name_to_run = {run.name: run for run in runs}
# Extract data for all runs
data = extract_run_data(list(name_to_run.keys()))

ret_project_name = "ret_unlearning|src|main_runner.py"
# get all the runs of the project and this group
ret_runs = api.runs(ret_project_name, filters={"group": group})
name_to_run = {run.name: run for run in ret_runs}
# Extract data for all runs
ret_data = extract_run_data(list(name_to_run.keys()))

print(ret_data.keys())

id_to_label = {
    3: "CIR",
    30: "CIR (handicapped)",
    # 16: "circuit breakers",
    6: "circuit breakers (handicapped)",
    # 33: "circuit breakers",
    12: "gradient diff (handicapped)",
}
# the in-between ones, which turned out too weak and lose with retaining!:
# 32: "circuit breakers (handicapped)",
# 22: "gradient diff",

# ! more than 23x drop in post-attack accuracy, despite 10x less disruption
dual_plot(data, ret_data, id_to_label, "cyber", num_rel_epochs=60, big_disr_perc=1)

# %%

group = "main_comparison_llama_bio"
api = wandb.Api(timeout=3600)

project_name = "unlearning|src|main_runner.py"
# get all the runs of the project and this group
runs = api.runs(project_name, filters={"group": group})
name_to_run = {run.name: run for run in runs}
# Extract data for all runs
data = extract_run_data(list(name_to_run.keys()))

ret_project_name = "ret_unlearning|src|main_runner.py"
# get all the runs of the project and this group
ret_runs = api.runs(ret_project_name, filters={"group": group})
name_to_run = {run.name: run for run in ret_runs}
# Extract data for all runs
ret_data = extract_run_data(list(name_to_run.keys()))

print(ret_data.keys())

id_to_label = {
    3: "CIR",
    18: "CIR (handicapped)",
    # 7: "circuit breakers (handicapped)",  # 7 is too fast
    6: "circuit breakers (handicapped)",
    13: "gradient diff (handicapped)", # todo update this after 10,11,12 finished
}

# 
dual_plot(data, ret_data, id_to_label, "bio", num_rel_epochs=60, big_disr_perc=1)
