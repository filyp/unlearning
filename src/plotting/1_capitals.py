# %%
# %load_ext autoreload
# %autoreload 2
import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from utils.common_cir import prepare_model
from utils.git_and_reproducibility import repo_root
from utils.plots import visualize
from utils.training import get_grad, prepare_answer_mask, set_seeds

# plt dark theme
# plt.style.use("dark_background")
plt.style.use("default")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10

logging.basicConfig(level=logging.INFO)
pt.set_default_device("cuda")

conf = OmegaConf.load(repo_root() / "configs/transferability.yaml")
# conf.model_id = "meta-llama/Llama-3.2-3B"
conf.model_id = "meta-llama/Llama-3.2-1B"
# ! setup
set_seeds(42)
tokenizer = AutoTokenizer.from_pretrained(conf.model_id)
tokenizer.pad_token = tokenizer.eos_token

# * install hooks
conf.target_modules = ["gate_proj", "up_proj", "down_proj", "k_proj", "v_proj", "q_proj", "o_proj"]  # fmt: skip
conf.device = "cuda" if pt.cuda.is_available() else "cpu"
model = prepare_model(conf)


def tensor_dict_dot_product(a, b):
    acc = 0
    for k in a.keys():
        acc += (a[k].to(pt.float32) * b[k].to(pt.float32)).sum()
    return acc


def tensor_dict_cossim(a, b):
    a_dot_b = tensor_dict_dot_product(a, b)
    a_norm = tensor_dict_dot_product(a, a).sqrt()
    b_norm = tensor_dict_dot_product(b, b).sqrt()
    return a_dot_b / (a_norm * b_norm)


# %%
beginning, ending = "The capital of France is", "Paris"
# beginning, ending = "The capital of Italy is", "Paris"
# beginning, ending = "The capital of Gondor is", "Minas Tirith"

beginning_batch = tokenizer(beginning, **conf.tokenizer)
full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
loss_mask = prepare_answer_mask(beginning_batch, full_batch)
ref_grad = get_grad(model, full_batch, loss_mask, "cross_entropy")

infos = []
for beginning, ending in [
    ("The capital of France is", "Paris"),
    # ("The capital of Germany is", "Berlin"),
    # ("The capital of England is", "London"),
    ("The capital of Spain is", "Madrid"),
    ("The capital of China is", "Beijing"),
    # ("The capital of Italy is", "Rome"),
    ("The capital of Poland is", "Warsaw"),
    ("The capital of Ukraine is", "Kyiv"),
    ("The capital of Japan is", "Tokio"),
    ("The capital of Skyrim is", "Solitude"),
    ("The capital of Rohan is", "Edoras"),
    # ("The capital of the Underworld is", "Hades"),
    # ("The capital of Italy is", "Paris"),
    # ("The capital of USA is", "Paris"),
    # ("The capital of Italy is", "Warsaw"),
    # ("The capital of USA is", "Warsaw"),
    # ("Die Hauptstadt von Frankreich ist", "Paris"),
    # ("La capital de Francia es", "París"),
    # ("Столица Франции", "Париж"),
    # ("Stolica Francji to", "Paryż"),
    # ("A capital de França é", "Paris"),
    ("The French anthem is", "La Marseillaise"),
    ("The city of love is", "Paris"),
    ("The Eiffel Tower is in", "Paris"),
    ("The Statue of Liberty is in", "New York"),
    # ("The biggest city in France is", "Paris"),
    # ("The Brandenburg Gate is in", "Berlin"),
    ("The hardest metal is", "Tungsten"),
    # ("Water is", "wet"),
    # ("The term for self-fertilization is", "autogamy"),
    # ("The term for the death of cells is", "apoptosis"),
    # ("The symbol of helium is", "He"),
    # ("The oldest building in the world is", "The Great Pyramid of Giza"),
    # ("The languages of Canada are", "English and French"),
]:
    beginning_batch = tokenizer(beginning, **conf.tokenizer)
    full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
    loss_mask = prepare_answer_mask(beginning_batch, full_batch)
    grad = get_grad(model, full_batch, loss_mask, "cross_entropy")

    cossim = tensor_dict_cossim(grad, ref_grad)

    module = model.model.layers[8].mlp.gate_proj
    pre_answer_pos = beginning_batch["input_ids"].shape[1] - 1
    pre_answer_act = module.last_act_full[0, pre_answer_pos]

    infos.append(
        dict(
            beginning=beginning,
            ending=ending,
            cossim=cossim,
            act=pre_answer_act,
        )
    )

# acts = pt.stack([info["act"] for info in infos])
# visualize(acts)


# %%
def create_activation_visualization(act_tensor, max_vals=60):
    """Create visualization for activation values"""
    if isinstance(act_tensor, pt.Tensor):
        act_values = act_tensor.cpu().float().numpy()
    else:
        act_values = act_tensor

    # Take only first 60 values
    act_values = act_values[:max_vals]

    # Normalize to [-1, 1]
    max_abs = np.abs(act_values).max()
    if max_abs > 0:
        act_values = act_values / max_abs

    # Create RGB image (1 x 60 x 3) - same as visualize function
    # Green for positive, red for negative, black for 0
    rgb_image = np.zeros((1, len(act_values), 3))
    rgb_image[0, :, 0] = np.clip(-act_values, 0, 1)  # Red channel for negative values
    rgb_image[0, :, 1] = np.clip(act_values, 0, 1)  # Green channel for positive values
    rgb_image[0, :, 2] = 0  # Blue channel stays 0

    return rgb_image


def create_plot(infos):
    text_y_pos = 0.4
    """Create the main visualization plot"""
    n_rows = len(infos)

    # Split into two groups: first 11 and remaining 4
    group1 = infos[:-5]
    group2 = infos[-5:]

    # Create figure with custom gridspec for the two groups
    # ICLR template is 5.5 inches wide
    fig = plt.figure(figsize=(5.5, n_rows * 0.2 + 0.2))  # Extra height for spacing

    # Calculate height ratios: rows for group1, spacing, rows for group2
    height_ratios = [1] * len(group1) + [0.5] + [1] * len(group2)  # 0.5 for white gap

    gs = gridspec.GridSpec(
        len(group1) + 1 + len(group2),  # +1 for spacing row
        3,
        width_ratios=[0.45, 0.1, 0.45],
        height_ratios=height_ratios,
        hspace=-0.01,  # No vertical spacing between rows
        wspace=0.15,  # Minimal horizontal spacing
        left=0.0,  # Remove left margin
        right=1.0,  # Remove right margin
        top=0.92,
        bottom=0.02,
    )

    # Create activation visualizations for each group
    def create_group_activation_image(group_infos):
        """Stack activation visualizations for a group vertically"""
        group_acts = []
        for info in group_infos:
            act_vis = create_activation_visualization(info["act"])
            # Remove the first dimension (1, 60, 3) -> (60, 3)
            group_acts.append(act_vis[0])

        # Stack vertically to create (n_rows, 60, 3)
        return np.stack(group_acts, axis=0)

    group1_acts = create_group_activation_image(group1)
    group2_acts = create_group_activation_image(group2)

    # Process both groups
    all_infos = group1 + group2
    row_indices = list(range(len(group1))) + list(
        range(len(group1) + 1, len(group1) + 1 + len(group2))
    )

    for i, info in enumerate(all_infos):
        actual_row = row_indices[i]

        # Column 1: Text with purple ending
        ax1 = fig.add_subplot(gs[actual_row, 0])
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis("off")

        # Split text and color the ending purple
        beginning = info["beginning"]
        ending = info["ending"]

        # Display beginning text in black
        ax1.text(
            0.01,
            text_y_pos,
            beginning,
            va="center",
            ha="left",
            color="black",
        )

        # Display ending text in purple
        ax1.text(
            0.99,
            text_y_pos,
            ending,
            va="center",
            ha="right",
            color="purple",
            # weight="bold",
        )

        # Column 2: Cosine similarity with colored background
        ax2 = fig.add_subplot(gs[actual_row, 1])
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis("off")

        # Convert cosine similarity to percentage
        cossim_val = (
            info["cossim"].item()
            if isinstance(info["cossim"], pt.Tensor)
            else info["cossim"]
        )
        percentage = int(cossim_val * 100)

        # Create colored background: white (1,1,1) to red (1,0,0)
        # Interpolate based on absolute cosine similarity value
        abs_cossim = abs(cossim_val)
        bg_color = (1.0, 1.0 - abs_cossim, 1.0 - abs_cossim)  # From white to red

        # Create background rectangle
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=0, facecolor=bg_color)
        ax2.add_patch(rect)

        # Add percentage text - always black
        ax2.text(
            0.5,
            text_y_pos,
            f"{percentage}%",
            va="center",
            ha="center",
            color="black",
            # weight="bold",
        )

    # Create single imshow for group1 activations
    ax3_group1 = fig.add_subplot(gs[: len(group1), 2])
    ax3_group1.imshow(group1_acts, aspect="auto", extent=[0, 60, len(group1), 0])
    ax3_group1.set_xlim(0, 60)
    ax3_group1.set_ylim(0, len(group1))
    ax3_group1.set_xticks([])
    ax3_group1.set_yticks([])
    for spine in ax3_group1.spines.values():
        spine.set_visible(False)

    # Create single imshow for group2 activations
    ax3_group2 = fig.add_subplot(gs[len(group1) + 1 :, 2])
    ax3_group2.imshow(group2_acts, aspect="auto", extent=[0, 60, len(group2), 0])
    ax3_group2.set_xlim(0, 60)
    ax3_group2.set_ylim(0, len(group2))
    ax3_group2.set_xticks([])
    ax3_group2.set_yticks([])
    for spine in ax3_group2.spines.values():
        spine.set_visible(False)

    # Add white spacing row (it will be empty by default)
    spacing_ax = fig.add_subplot(gs[len(group1), :])
    spacing_ax.axis("off")
    spacing_ax.set_facecolor("white")

    # Set column titles with proper positioning
    if n_rows > 0:
        # Calculate proper x positions based on the new gridspec
        fig.text(
            0.2,
            0.97,
            "Prompt",
            fontsize=12,
            ha="center",
            # weight="bold",
        )
        fig.text(
            0.5,
            0.97,
            "Grad Cossim",
            fontsize=12,
            ha="center",
            # weight="bold",
        )
        fig.text(0.8, 0.97, "Activations Slice", fontsize=12, ha="center")


create_plot(infos)

# save
stem = Path(__file__).stem
plot_path = repo_root() / f"paper/plots/{stem}.pdf"
plt.savefig(plot_path, bbox_inches="tight", dpi=300)
