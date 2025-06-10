import matplotlib.pyplot as plt
import numpy as np
import torch as pt


def visualize_rgb(values, shape_lim=60, scale=None):
    assert len(values.shape) == 3

    if isinstance(values, pt.Tensor):
        values = values.cpu().float().numpy()
    values = values[:shape_lim, :shape_lim]

    max_val = np.abs(values).max()
    if scale is None:
        values = values / max_val
    else:
        if max_val > scale:
            print(f"WARNING: max_val: {max_val} > scale: {scale}")
        values = values / scale

    # rgb, where green is positive, red is negative, black is 0
    plt.imshow(values)


def visualize(values, shape_lim=60, scale=None):
    if isinstance(values, pt.Tensor):
        values = values.cpu().float().numpy()
    # elif isinstance(values, list):
    #     values = np.array(values)

    if len(values.shape) == 1:
        # make it 2D
        values = values.reshape(1, -1)
    values = values[:shape_lim, :shape_lim]

    max_val = np.abs(values).max()
    if scale is None:
        values = values / max_val
    else:
        if max_val > scale:
            print(f"WARNING: max_val: {max_val} > scale: {scale}")
        values = values / scale

    # rgb, where green is positive, red is negative, black is 0
    values = np.stack(
        [np.clip(-values, 0, 1), np.clip(values, 0, 1), np.zeros_like(values)], axis=-1
    )
    plt.imshow(values)


# # %% module x layer, all corpus examples
# # example usage
# interven_grad = TensorDict(
#     {n: pt.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
# )
# # for ex in f_corpus:
# ex = f_corpus[22]
# interven_grad += get_grad_from_example(model, ex)
# n_to_color = {
#     n: [
#         max((interven_grad[n] * control_grad[n]).sum(), 0),
#         max((interven_grad[n] * target_grad[n]).sum(), 0),
#         0,
#     ]
#     for n, p in model.named_parameters()
#     if p.requires_grad
# }
# # Normalize
# max_val = max(max(color) for color in n_to_color.values())
# n_to_color = {n: [float(x / max_val) for x in v] for n, v in n_to_color.items()}
# visualize_module_values(n_to_color, "")


def visualize_module_values(n_to_color, title):
    # Parse labels and create color mapping
    layer_module_colors = []
    for param_name, color in n_to_color.items():
        if "layers." not in param_name or "layernorm" in param_name:
            continue

        # Parse layer and module
        parts = param_name.split("layers.")[1].split(".")
        layer = int(parts[0])
        param_name = ".".join(parts[1:]).replace(".weight", "")
        layer_module_colors.append((layer, param_name, color))

    # Get unique sorted modules and layers
    unique_modules = sorted(set(module for _, module, _ in layer_module_colors))
    layer_nums = sorted(set(layer for layer, _, _ in layer_module_colors))

    # Create color lookup dictionary
    color_matrix = {layer: {} for layer in layer_nums}
    for layer, param_name, color in layer_module_colors:
        color_matrix[layer][param_name] = color

    # Create the visualization
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_axis_off()

    # Calculate grid dimensions
    cell_height = 1
    cell_width = 1.5
    width = len(layer_nums) * cell_width
    height = len(unique_modules) * cell_height

    # Add padding around the plot
    plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.2)

    # Draw cells
    for i, module in enumerate(unique_modules):
        for j, layer in enumerate(layer_nums):
            color = color_matrix[layer][module]
            rect = plt.Rectangle(
                (j * cell_width, (len(unique_modules) - 1 - i) * cell_height),
                cell_width - 0.1,
                cell_height - 0.1,
                facecolor=color,
            )
            ax.add_patch(rect)

        # Add module names on the left
        ax.text(
            -0.3,
            (len(unique_modules) - 1 - i) * cell_height + cell_height / 2,
            module,
            ha="right",
            va="center",
        )

    # Add layer labels at the bottom
    for j, layer in enumerate(layer_nums):
        ax.text(
            j * cell_width + cell_width / 2,
            -0.3,
            f"{layer}",
            ha="center",
            va="top",
        )

    # Set the plot limits
    plt.xlim(-1, width + 0.5)
    plt.ylim(-2, height + 2.5)

    plt.title(title, y=0.85)
    plt.show()


def visualize_token_layer_values(all_control_sims, all_target_sims, tokens, title=""):
    """
    Creates a heatmap visualization of token-layer values.

    Args:
        all_control_sims: Tensor of shape (num_layers, num_tokens) containing control similarities
        tokens: List of token strings
        title: Optional title for the plot
    """
    # Create the visualization
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_axis_off()

    # Calculate grid dimensions
    cell_height = 1
    cell_width = 1.5
    width = all_control_sims.shape[0] * cell_width
    height = len(tokens) * cell_height

    # Add padding around the plot
    plt.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.2)

    # Draw cells
    for i, token in enumerate(tokens):
        for j in range(all_control_sims.shape[0]):
            # Get the control similarity value and create a red color
            value_control = all_control_sims[j, i].item()
            value_target = all_target_sims[j, i].item()
            color = (value_control, value_target, 0)  # Red channel only
            # color = (0, value_target, 0)  # Red channel only
            rect = plt.Rectangle(
                (j * cell_width, (len(tokens) - 1 - i) * cell_height),
                cell_width - 0.1,
                cell_height - 0.1,
                facecolor=color,
            )
            ax.add_patch(rect)

        # Add token names on the left
        ax.text(
            -0.3,
            (len(tokens) - 1 - i) * cell_height + cell_height / 2,
            token,
            ha="right",
            va="center",
        )

    # Add layer labels at the bottom
    for j in range(all_control_sims.shape[0]):
        ax.text(
            j * cell_width + cell_width / 2,
            -0.3,
            f"{j}",
            ha="center",
            va="top",
        )

    # Set the plot limits
    plt.xlim(-1, width + 0.5)
    plt.ylim(-2, height + 2.5)

    plt.title(title, y=0.85)
    plt.show()
