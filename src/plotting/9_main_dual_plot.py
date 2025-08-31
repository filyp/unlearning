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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.75))
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 3.05))

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
        # style = "-"

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
        # style = "-"

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

    ax1.text(ax1.get_xlim()[1] - 0.020, 25.7, "random level", ha="right", va="bottom")
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

    # # Create legend handles manually
    # legend_elements = [
    #     Line2D([0], [0], color="green", label="CIR", linestyle="--"),
    #     Line2D([0], [0], color="green", label="CIR (less disruption)", linestyle="-"),
    #     Line2D([0], [0], color="purple", label="circuit breakers", linestyle="--"),
    #     Line2D([0], [0], color="black", label="ideal", marker='>', markersize=4, linestyle='-'),
    #     Line2D([0], [0], color="orange", label="gradient diff", linestyle="--"),
    #     # Line2D([0], [0], color="black", label="ideal"),
    # ]

    # Create legend handles manually
    legend_elements = [
        Line2D([0], [0], color="black", label="ideal", marker='>', markersize=4, linestyle='-'),
        Line2D([0], [0], color="green", label="CIR", linestyle="-"),
        # Line2D([0], [0], color="green", label="CIR (less disruption)", linestyle="-"),
        Line2D([0], [0], color="purple", label="Circuit Breakers", linestyle="-"),
        Line2D([0], [0], color="orange", label="Gradient Difference", linestyle="-"),
        # Line2D([0], [0], color="black", label="ideal"),
    ]
    if "CIR (handicapped)" in id_to_label.values():
        legend_elements.insert(1, Line2D([0], [0], color="green", label="CIR (with 1% disr.)", linestyle="--"))

    # Add legend at the bottom with 3 columns
    # fig.legend(bbox_to_anchor=(0.5, -0.23), loc="center", ncol=3)
    # fig.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.27), loc="center", ncol=3)
    # fig.legend(handles=legend_elements, bbox_to_anchor=(0.5, 0.10), loc="center", ncol=3)
    _ncol = 3 if "CIR (handicapped)" in id_to_label.values() else 4
    fig.legend(handles=legend_elements, bbox_to_anchor=(0.5, 0.10), loc="center", ncol=_ncol)
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.37)
    plt.subplots_adjust(bottom=0.37 if "CIR (handicapped)" in id_to_label.values() else 0.32)

    ax1.set_yticks([20, 30, 40, 50, 60])
    ax2.set_yticks([20, 30, 40, 50, 60])

    # Save the plot
    stem = Path(__file__).stem
    plot_path = repo_root() / f"paper/plots/{stem}_{dataset}_{big_disr_perc}.pdf"
    plt.savefig(plot_path, bbox_inches=None, dpi=300)
    plt.show()


# %% bio

group = "main_comparison_llama_bio_2perc"
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

# * 2 perc version
# id_to_label = {
#     # 2: "CIR",
#     # 17: "CIR (handicapped)",
#     3: "CIR",
#     # 16: "CIR (handicapped)",
#     6: "circuit breakers (handicapped)",
#     12: "gradient diff (handicapped)",
# }
# dual_plot(data, ret_data, id_to_label, "bio", num_rel_epochs=100, big_disr_perc=2)

# * 3 perc version
id_to_label = {
    # 2: "CIR",
    # 17: "CIR (handicapped)",
    3: "CIR",
    # 16: "CIR (handicapped)",
    21: "circuit breakers",
    # 28: "gradient diff",
    # 27: "gradient diff",
    30: "gradient diff",
    # 15: "CIR no retain",  # it has the final accuracy of 51%
}
dual_plot(data, ret_data, id_to_label, "bio", num_rel_epochs=100, big_disr_perc=3)

# %% performance analysis

# # ! 24x improvement in terms of pre-attack accuracy
# init_wikitext_loss = list(data.values())[0][0]["wikitext_loss"]
# init_acc = list(data.values())[0][0]["forget_acc_t1"]
# small_thresh = init_wikitext_loss * 1.001
# for x in data["21|circuit_breaker_GA_1.03_0.002|"]:
#     if x["wikitext_loss"] > small_thresh:
#         break
# cb_drop = init_acc - x["forget_acc_t1"]
# cir_drop = init_acc - data["3|mlp_confuse_0.2|"][-1]["forget_acc_t1"]
# cir_drop / cb_drop

init_acc = list(data.values())[0][0]["forget_acc_t1"]

# cb_accs = [x["forget_acc_t1"] for x in ret_data["21|circuit_breaker_GA_1.03_0.002|"]]
# cir_accs = [x["forget_acc_t1"] for x in ret_data["3|mlp_confuse_0.2|"]]
# ga_accs = [x["forget_acc_t1"] for x in ret_data["30|neg_cross_entropy_GA_1e-05_['cross_entropy']_1.03_0.0001_1000|"]]
# # bucket into buckets of 10 each and average in bucket, then take the max
# cb_post_drop = init_acc - max([np.mean(cb_accs[i:i+10]) for i in range(0, len(cb_accs), 10)])
# cir_post_drop = init_acc - max([np.mean(cir_accs[i:i+10]) for i in range(0, len(cir_accs), 10)])
# ga_post_drop = init_acc - max([np.mean(ga_accs[i:i+10]) for i in range(0, len(ga_accs), 10)])

cb_post_drop = init_acc - np.mean([x["forget_acc_t1"] for x in ret_data["21|circuit_breaker_GA_1.03_0.002|"][-10:]])
cir_post_drop = init_acc - np.mean([x["forget_acc_t1"] for x in ret_data["3|mlp_confuse_0.2|"][-10:]])
ga_post_drop = init_acc - np.mean([x["forget_acc_t1"] for x in ret_data["30|neg_cross_entropy_GA_1e-05_['cross_entropy']_1.03_0.0001_1000|"][-10:]])
print(cb_post_drop, cir_post_drop, ga_post_drop)
cir_post_drop / max(cb_post_drop, ga_post_drop)
# ! so it's a 86x improvement
# * but expaining this process would be complicated so let's report the more modest 30x drop based just on the final 10 samples


# %% cyber

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


# # * 1 perc version
# id_to_label = {
#     3: "CIR",
#     30: "CIR (handicapped)",
#     # 16: "circuit breakers",
#     6: "circuit breakers (handicapped)",
#     # 33: "circuit breakers",
#     12: "gradient diff (handicapped)",
# }
# # the in-between ones, which turned out too weak and lose with retaining!:
# # 32: "circuit breakers (handicapped)",
# # 22: "gradient diff",
# # ! more than 23x drop in post-attack accuracy, despite 10x less disruption
# dual_plot(data, ret_data, id_to_label, "cyber", num_rel_epochs=60, big_disr_perc=1)


# * 3 perc version
id_to_label = {
    3: "CIR",
    30: "CIR (handicapped)",
    # 45: "CIR (handicapped)",  # could not reach 3%!
    6: "circuit breakers",
    36: "circuit breakers",
    # 12: "gradient diff",
    43: "gradient diff",
    # 25: "CIR no retain",  # it has the final accuracy of 41%
}
# the in-between ones, which turned out too weak and lose with retaining!:
# 32: "circuit breakers (handicapped)",
# 22: "gradient diff",
# ! more than 23x drop in post-attack accuracy, despite 10x less disruption
dual_plot(data, ret_data, id_to_label, "cyber", num_rel_epochs=100, big_disr_perc=3)



# %% performance analysis

init_acc = list(data.values())[0][0]["forget_acc_t1"]
cir_post_drop = init_acc - np.mean([x["forget_acc_t1"] for x in ret_data["3|mlp_confuse_0.2|"][-10:]])
cb_post_drop = init_acc - np.mean([x["forget_acc_t1"] for x in ret_data["6|circuit_breaker_GA_1.01_0.002|"][-10:]])
cb_post_drop2 = init_acc - np.mean([x["forget_acc_t1"] for x in ret_data["36|circuit_breaker_GA_1.03_0.002|"][-10:]])
ga_post_drop = init_acc - np.mean([x["forget_acc_t1"] for x in ret_data["43|neg_cross_entropy_GA_1e-05_['cross_entropy']_1.03_0.0002|"][-10:]])
# cir_accs = [x["forget_acc_t1"] for x in ret_data["3|mlp_confuse_0.2|"]]
# cb_accs = [x["forget_acc_t1"] for x in ret_data["6|circuit_breaker_GA_1.01_0.002|"]]
# cb_accs2 = [x["forget_acc_t1"] for x in ret_data["36|circuit_breaker_GA_1.03_0.002|"]]
# ga_accs = [x["forget_acc_t1"] for x in ret_data["43|neg_cross_entropy_GA_1e-05_['cross_entropy']_1.03_0.0002|"]]
# # bucket into buckets of 10 each and average in bucket, then take the max
# cir_post_drop = init_acc - max([np.mean(cir_accs[i:i+10]) for i in range(0, len(cir_accs), 10)])
# cb_post_drop = init_acc - max([np.mean(cb_accs[i:i+10]) for i in range(0, len(cb_accs), 10)])
# cb_post_drop2 = init_acc - max([np.mean(cb_accs2[i:i+10]) for i in range(0, len(cb_accs2), 10)])
# ga_post_drop = init_acc - max([np.mean(ga_accs[i:i+10]) for i in range(0, len(ga_accs), 10)])
print(cir_post_drop, cb_post_drop, cb_post_drop2, ga_post_drop)
cir_post_drop / max(cb_post_drop, cb_post_drop2, ga_post_drop)
# ! so it's a 31x drop
# * but expaining this process would be complicated so let's report the more modest 6.4x drop based just on the final 10 samples

# %%


# %% bio - unused!

# group = "main_comparison_llama_bio"
# api = wandb.Api(timeout=3600)

# project_name = "unlearning|src|main_runner.py"
# # get all the runs of the project and this group
# runs = api.runs(project_name, filters={"group": group})
# name_to_run = {run.name: run for run in runs}
# # Extract data for all runs
# data = extract_run_data(list(name_to_run.keys()))

# ret_project_name = "ret_unlearning|src|main_runner.py"
# # get all the runs of the project and this group
# ret_runs = api.runs(ret_project_name, filters={"group": group})
# name_to_run = {run.name: run for run in ret_runs}
# # Extract data for all runs
# ret_data = extract_run_data(list(name_to_run.keys()))

# print(ret_data.keys())

# # * 1 perc version
# id_to_label = {
#     3: "CIR",
#     18: "CIR (handicapped)",
#     # 7: "circuit breakers (handicapped)",  # 7 is too fast
#     6: "circuit breakers (handicapped)",
#     13: "gradient diff (handicapped)", # todo update this after 10,11,12 finished
#     # ! nah, actually don't use this one at all, OOMs are unavoidable with this config
# }
# dual_plot(data, ret_data, id_to_label, "bio", num_rel_epochs=60, big_disr_perc=1)
