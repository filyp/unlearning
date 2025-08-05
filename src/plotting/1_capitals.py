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

from utils.common_cir import *
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


model = AutoModelForCausalLM.from_pretrained(
    conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)
model.config.use_cache = False

# * set trainable params
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in conf.target_modules)

install_hooks(model)


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

    # ("The French anthem is", "La Marseillaise"),
    # ("The city of love is", "Paris"),
    # ("The Eiffel Tower is in", "Paris"),
    # ("The Statue of Liberty is in", "New York"),
    # # ("The biggest city in France is", "Paris"),
    # ("The Brandenburg Gate is in", "Berlin"),
    # ("The hardest metal is", "Tungsten"),

    # ("The capital of the Underworld is", "Hades"),
    # ("The capital of Italy is", "Paris"),
    # ("The capital of USA is", "Paris"),
    # ("The capital of Italy is", "Warsaw"),
    # ("The capital of USA is", "Warsaw"),

    # ("Water is", "wet"),
    # ("The term for self-fertilization is", "autogamy"),
    # ("The term for the death of cells is", "apoptosis"),
    # ("The symbol of helium is", "He"),
    # ("The languages of Canada are", "English and French"),

    # ("The Eternal City is", "Rome"),
    # ("The Burj Khalifa is in", "Dubai"),
    # ("The largest city in the USA is", "New York"),
    # ("The largest city in India is", "Mumbai"),
    # ("The largest city in Japan is", "Tokyo"),
    # ("The largest city in Turkey is", "Istanbul"),
    # ("The City of Angels is", "Los Angeles"),
    # ("The Venice of the North is", "Stockholm"),
    # ("The White City is", "Tel Aviv"),
    # ("The Mile High City is", "Denver"),

    # ("The capital of Germany is", "Berlin"),
    # ("The capital of England is", "London"),
    # ("The capital of Italy is", "Rome"),


# %%
beginning_ending_pairs_groups1 = [
    [
        ("The capital of France is", "Paris"),
    ],
    [
        ("The capital of Spain is", "Madrid"),
        ("The capital of China is", "Beijing"),
        # ("The capital of Poland is", "Warsaw"),
        ("The capital of Ukraine is", "Kyiv"),
        # ("The capital of Japan is", "Tokio"),
    ],
    [
        ("The capital of France is", "Madrid"),
        ("The capital of Spain is", "Beijing"),
        ("The capital of China is", "Kyiv"),
        ("The capital of Ukraine is", "Paris"),
    ],
    [
        # ("The largest continent is", "Asia"),
        ("The largest planet is", "Jupiter"),
        ("The author of 1984 is", "George Orwell"),
        # ("The author of 1984 is", "Ernest Hemingway"),
        ("Marie Curie discovered", "radium"),
        ("Prometheus stole", "fire"),
        # ("The chemical symbol for gold is", "Au"),
    ],
]
beginning_ending_pairs_groups2 = [
    [
        # * what's interesting, their disruption is completely stopped when using correct logit loss
        # * for wrong capitals it's the opposite though! - correct logit makes disruption much stronger
        ("The capital of Skyrim is", "Solitude"),
        ("The capital of Rohan is", "Edoras"),
    ],
    [
        ("Die Hauptstadt von Frankreich ist", "Paris"),
        ("La capital de Francia es", "París"),
        # ("Die Hauptstadt von Japan ist", "Tokio"),
        ("Столица Франции", "Париж"),
        # ("Stolica Francji to", "Paryż"),
        ("A capital de França é", "Paris"),
    ],
    [
        # Using "contains"
        ("Water contains", "hydrogen"),
        ("Salt contains", "sodium"),
        ("Diamond contains", "carbon"),
        ("Air contains", "oxygen"),
    ],
    [
        ("Napoleon is", "French"),
        ("Napoleon was", "French"),
        # ("Einstein is", "German"),
        # ("Einstein was", "German"),
        ("Mozart is", "Austrian"),
        ("Mozart was", "Austrian"),
    ],
    [
        ("Gold is", "valuable"),
        ("Gold was", "valuable"),
    ],
    [
        ("The library is", "quiet"),
        ("The library was", "quiet"),
    ],
    # [
    #     # Using "invented"
    #     ("Atlas holds", "the world"),
    #     ("Thomas Edison invented", "the light bulb"),
    # ],
    # [
    #     # Using "created"
    #     ("Mark Zuckerberg created", "Facebook"),
    #     ("Bill Gates created", "Microsoft"),
    #     ("Steve Jobs created", "Apple"),
    #     ("Guido van Rossum created", "Python"),
    # ],
    # [
    #     # Using "flows through"
    #     ("The Seine flows through", "Paris"),
    #     ("The Thames flows through", "London"),
    #     ("The Nile flows through", "Egypt"),
    #     ("The Amazon flows through", "Brazil"),
    # ],
    # [
    #     # Using "borders"
    #     ("Mexico borders", "the United States"),
    #     ("Canada borders", "Alaska"),
    #     ("Germany borders", "Poland"),
    #     ("France borders", "Spain"),
    # ],
]


def get_infos(beginning_ending_pairs, loss_fn_name="cross_entropy"):
    # * refecence prompt
    beginning, ending = "The capital of France is", "Paris"
    # beginning, ending = "The capital of Italy is", "Paris"
    # beginning, ending = "The capital of Gondor is", "Minas Tirith"

    beginning_batch = tokenizer(beginning, **conf.tokenizer)
    full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
    loss_mask = prepare_answer_mask(beginning_batch, full_batch)
    ref_grad = get_grad(model, full_batch, loss_mask, loss_fn_name)

    infos = []
    for beginning, ending in beginning_ending_pairs:
        beginning_batch = tokenizer(beginning, **conf.tokenizer)
        full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
        loss_mask = prepare_answer_mask(beginning_batch, full_batch)
        grad = get_grad(model, full_batch, loss_mask, loss_fn_name)

        cossim = tensor_dict_cossim(grad, ref_grad)

        module = model.model.layers[8].mlp.gate_proj
        pre_answer_pos = beginning_batch["input_ids"].shape[1] - 1
        pre_answer_act = module.last_act_full[0, pre_answer_pos]
        pre_answer_grad = module.last_grad_full[0, pre_answer_pos]

        infos.append(
            dict(
                beginning=beginning,
                ending=ending,
                cossim=cossim,
                act=pre_answer_act,
                grad=pre_answer_grad,
            )
        )
    return infos


# acts = pt.stack([info["act"] for info in infos])
# visualize(acts)


def create_activation_visualization(act_tensor, max_vals):
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


def create_plot(group_infos):
    text_y_pos = 0.4
    """Create the main visualization plot with both activations and gradients"""

    # Calculate total rows and create height ratios
    total_rows = sum(len(group) for group in group_infos)
    num_gaps = len(group_infos) - 1  # gaps between groups

    # Build height ratios: group rows + gaps between groups
    height_ratios = []
    for i, group in enumerate(group_infos):
        height_ratios.extend([1] * len(group))  # rows for this group
        if i < len(group_infos) - 1:  # not the last group
            height_ratios.append(0.35)  # gap after this group

    # Create figure with custom gridspec for all groups
    # ICLR template is 5.5 inches wide
    fig = plt.figure(figsize=(5.5, total_rows * 0.2 + 0.0))  # Extra height for spacing

    gs = gridspec.GridSpec(
        total_rows + num_gaps,  # total rows + gaps
        4,  # Changed to 4 columns
        width_ratios=[0.45, 0.1, 0.225, 0.225],
        height_ratios=height_ratios,
        hspace=-0.01,  # No vertical spacing between rows
        wspace=0.07,  # Minimal horizontal spacing
        left=0.0,  # Remove left margin
        right=1.0,  # Remove right margin
        top=0.92,
        bottom=0.02,
    )

    # Create activation visualizations for each group
    def create_group_slice_image(group_infos, slice_type):
        """Stack slice visualizations for a group vertically"""
        group_slices = []
        for info in reversed(group_infos):
            slice_vis = create_activation_visualization(info[slice_type], max_vals=35)
            # Remove the first dimension (1, 60, 3) -> (60, 3)
            group_slices.append(slice_vis[0])

        # Stack vertically to create (n_rows, 60, 3)
        return np.stack(group_slices, axis=0)

    # Create activation and gradient images for all groups
    group_activation_images = []
    group_gradient_images = []
    for group in group_infos:
        group_activation_images.append(create_group_slice_image(group, "act"))
        group_gradient_images.append(create_group_slice_image(group, "grad"))

    # Build mapping from info index to grid row
    all_infos = []
    row_indices = []
    current_row = 0

    for group_idx, group in enumerate(group_infos):
        for info in group:
            all_infos.append(info)
            row_indices.append(current_row)
            current_row += 1

        # Add gap after each group except the last
        if group_idx < len(group_infos) - 1:
            current_row += 1  # skip gap row

    # Process all infos with their corresponding row indices
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
        cossim_val = info["cossim"]
        percentage = int(cossim_val * 100)

        # Create colored background: white (1,1,1) to red (1,0,0)
        # Interpolate based on absolute cosine similarity value
        assert cossim_val <= 1.001, cossim_val  # allow some numerical errors
        clipped_cossim = cossim_val.clamp(0, 1).item()
        bg_color = (
            1.0,
            1.0 - clipped_cossim,
            1.0 - clipped_cossim,
        )  # From white to red

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

    # Create imshow for each group's activations and gradients
    current_row = 0
    for group_idx, group in enumerate(group_infos):
        group_size = len(group)

        # Column 3: Activations
        ax3_group = fig.add_subplot(gs[current_row : current_row + group_size, 2])
        ax3_group.imshow(
            group_activation_images[group_idx],
            aspect="auto",
            extent=[0, 60, group_size, 0],
        )
        ax3_group.set_xlim(0, 60)
        ax3_group.set_ylim(0, group_size)
        ax3_group.set_xticks([])
        ax3_group.set_yticks([])
        for spine in ax3_group.spines.values():
            spine.set_visible(False)

        # Column 4: Gradients
        ax4_group = fig.add_subplot(gs[current_row : current_row + group_size, 3])
        ax4_group.imshow(
            group_gradient_images[group_idx],
            aspect="auto",
            extent=[0, 60, group_size, 0],
        )
        ax4_group.set_xlim(0, 60)
        ax4_group.set_ylim(0, group_size)
        ax4_group.set_xticks([])
        ax4_group.set_yticks([])
        for spine in ax4_group.spines.values():
            spine.set_visible(False)

        current_row += group_size

        # Add white spacing row if not the last group
        if group_idx < len(group_infos) - 1:
            spacing_ax = fig.add_subplot(gs[current_row, :])
            spacing_ax.axis("off")
            spacing_ax.set_facecolor("white")
            current_row += 1

    # Set column titles with proper positioning
    if total_rows > 0:
        # Calculate proper x positions based on the new gridspec
        caption_y_pos = 0.94
        fig.text(
            0.2,
            caption_y_pos,
            "Prompt",
            fontsize=12,
            ha="center",
            # weight="bold",
        )
        fig.text(
            0.49,
            caption_y_pos,
            "Disruption",
            fontsize=12,
            ha="center",
            # weight="bold",
        )
        fig.text(0.662, caption_y_pos, "Activations", fontsize=12, ha="center")
        fig.text(0.892, caption_y_pos, "Gradients", fontsize=12, ha="center")


# %%
group_infos = []
for group in beginning_ending_pairs_groups1:
    group_infos.append(get_infos(group, loss_fn_name="cross_entropy"))
    # group_infos.append(get_infos(group, loss_fn_name="correct_logit"))

create_plot(group_infos)

# save
stem = Path(__file__).stem
plot_path = repo_root() / f"paper/plots/{stem}.pdf"
plt.savefig(plot_path, bbox_inches=None, dpi=300)

# %%
group_infos = []
for group in beginning_ending_pairs_groups1[:1] + beginning_ending_pairs_groups2:
    group_infos.append(get_infos(group, loss_fn_name="cross_entropy"))
    # group_infos.append(get_infos(group, loss_fn_name="correct_logit"))

create_plot(group_infos)

# save
stem = Path(__file__).stem
plot_path = repo_root() / f"paper/plots/{stem}_2.pdf"
plt.savefig(plot_path, bbox_inches=None, dpi=300)
