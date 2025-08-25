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
# for rerunning with exact reproducibility, you may want to limit date to 25.08.2025

project_name = "unlearning|src|main_runner.py"
# get all the runs of the project and this group

runs = wandb.Api().runs(project_name)

# Remove runs with duplicated names (remove both if there's a collision)
from collections import Counter

name_counts = Counter(run.name for run in runs)
duplicate_names = {name for name, count in name_counts.items() if count > 1}
print(f"Found {len(duplicate_names)} duplicate names: {duplicate_names}")

# Filter out runs with duplicate names
runs_no_duplicates = [run for run in runs if run.name not in duplicate_names]
print(f"Removed {len(runs) - len(runs_no_duplicates)} runs with duplicate names")
print(f"Remaining runs: {len(runs_no_duplicates)}")

# %%

name_to_run = {run.name: run for run in runs_no_duplicates}
name_to_config = {run.name: run.config for run in runs_no_duplicates}
len(name_to_run.keys())

# %%
# Extract data for all runs
data = extract_run_data(list(name_to_run.keys())[-200:])

# %%
project_name = "ret_unlearning|src|main_runner.py"
# get all the runs of the project and this group
runs = wandb.Api().runs(project_name)

# Remove runs with duplicated names (remove both if there's a collision)
from collections import Counter

name_counts = Counter(run.name for run in runs)
duplicate_names = {name for name, count in name_counts.items() if count > 1}
print(f"Found {len(duplicate_names)} duplicate names: {duplicate_names}")

# Filter out runs with duplicate names
runs_no_duplicates = [run for run in runs if run.name not in duplicate_names]
print(f"Removed {len(runs) - len(runs_no_duplicates)} runs with duplicate names")
print(f"Remaining runs: {len(runs_no_duplicates)}")

name_to_run = {run.name: run for run in runs_no_duplicates}
len(name_to_run.keys())

# %%
ret_data = extract_run_data(list(name_to_run.keys())[-200:])

# %%

init_wikitext_loss = list(data.values())[-1][0]["wikitext_loss"]
init_retain_loss = list(data.values())[-1][0]["retain_loss"]
print(init_wikitext_loss)
print(init_retain_loss)
last_retraining_rate = list(name_to_config.values())[-1]["retraining_rate"]
print(last_retraining_rate)
# %%
wik_thresh = init_wikitext_loss * 1.001
ret_thresh = init_retain_loss * 1.001
print(wik_thresh)
print(ret_thresh)
rel_len = 51

breaking_points = []
first_ret_accs = []
last_ret_accs = []
colors = []
for k in list(data.keys()):
    config = name_to_config[k]
    if data[k][0]["wikitext_loss"] != init_wikitext_loss:
        continue
    if config.get("retraining_rate", 0) != last_retraining_rate:
        continue

    acc_at_breaking_point = None
    ws = [x["wikitext_loss"] for x in data[k]]
    rs = [x["retain_loss"] for x in data[k]]

    # # smoothing is kinda unnecessary
    # padded_ws = np.pad(ws, (2, 2), mode='edge')  # pad 2 on each side for kernel size 5
    # ws = np.convolve(padded_ws, np.ones(5)/5, mode='valid')
    padded_rs = np.pad(rs, (2, 2), mode="edge")  # pad 2 on each side for kernel size 5
    rs = np.convolve(padded_rs, np.ones(5) / 5, mode="valid")
    # for i, w in enumerate(ws):
    #     if w > wik_thresh:
    #         acc_at_breaking_point = data[k][i]["forget_acc_t1"]
    #         break
    # if x["retain_loss"] > ret_thresh:
    # if x["wikitext_loss"] > wik_thresh and x["retain_loss"] > ret_thresh:

    for i, r in enumerate(rs):
        w = ws[i]
        # if r > ret_thresh or w > wik_thresh:
        if r > ret_thresh:
            # if w > wik_thresh:
            acc_at_breaking_point = data[k][i]["forget_acc_t1"]
            break

    # print(f"k: {k}")
    if k not in ret_data:
        continue
    if len(ret_data[k]) != rel_len:
        continue
    if acc_at_breaking_point is None:
        continue
    if len(data[k]) < 10:
        continue

    first_ret_acc = ret_data[k][0]["forget_acc_t1"]
    last_ret_acc = ret_data[k][-1]["forget_acc_t1"]
    color = "blue" if name_to_config[k].get("retaining_rate", 0) > 0 else "black"

    breaking_points.append(acc_at_breaking_point)
    first_ret_accs.append(first_ret_acc)
    last_ret_accs.append(last_ret_acc)

    colors.append(color)

len(breaking_points)

plt.scatter(breaking_points, last_ret_accs, color=colors)
plt.plot([0.25, 0.65], [0.25, 0.65], color="red")
# set equal aspect ratio
plt.gca().set_aspect("equal")

# %%

plt.scatter(breaking_points, first_ret_accs, color=colors)
plt.plot([0.25, 0.65], [0.25, 0.65], color="red")
plt.gca().set_aspect("equal")


# %%
# for the paper
plt.style.use("default")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10

# Combined plot with arrows showing transition from first to last retention accuracy (horizontal)
plt.figure(figsize=(5.5, 4))

# Add arrows from first to last retention accuracy (flipped axes)
for i in range(len(breaking_points)):
    plt.annotate(
        "",
        xy=(last_ret_accs[i], breaking_points[i]),  # arrow end (last_ret_acc on x-axis)
        xytext=(
            first_ret_accs[i],
            breaking_points[i],
        ),  # arrow start (first_ret_acc on x-axis)
        # arrowprops=dict(arrowstyle="->", color=colors[i], alpha=0.9, lw=1.5),
        arrowprops=dict(arrowstyle="->", alpha=0.9, lw=1.5),
    )

# Add the diagonal reference line (flipped)
plt.plot([0.25, 0.65], [0.25, 0.65], color="red", linestyle="-", alpha=0.7, label="y=x")

# Set equal aspect ratio
plt.gca().set_aspect("equal")

# Add labels and title (flipped)
plt.xlabel("WMDP Accuracy During Fine-Tuning Attack")
plt.ylabel("WMDP Accuracy at 0.1% Disruption Threshold")
# plt.title('Retention Recovery: First to Last Accuracy Transition')
plt.legend()
# plt.grid(True, alpha=0.3)

# Set axis limits to show the data nicely (flipped)
x_min = min(min(first_ret_accs), min(last_ret_accs)) - 0.02
x_max = max(max(first_ret_accs), max(last_ret_accs)) + 0.02
y_min, y_max = min(breaking_points) - 0.02, max(breaking_points) + 0.02
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.tight_layout()

stem = Path(__file__).stem
plot_path = repo_root() / f"paper/plots/{stem}.pdf"
plt.savefig(plot_path, bbox_inches=None, dpi=300)

# %%
# save breaking points first_ret_accs and last_ret_accs
np.savez(
    repo_root() / "data/breaking_points_plot.npz",
    breaking_points=breaking_points,
    first_ret_accs=first_ret_accs,
    last_ret_accs=last_ret_accs,
)

# %%
# load
data = np.load(repo_root() / "data/breaking_points_plot.npz")
breaking_points = data["breaking_points"]
first_ret_accs = data["first_ret_accs"]
last_ret_accs = data["last_ret_accs"]

# %%
