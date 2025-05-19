import matplotlib.pyplot as plt


def visualize_module_values(n_to_value, title):
    """
    Creates a heatmap visualization of parameter values.

    Args:
        n_to_value: Dict mapping parameter names to values between -1 and 1
            -1 is red, 0 is black, 1 is green
    """
    # Parse labels and create color mapping
    layer_module_colors = []
    for param_name, value in n_to_value.items():
        if "layers." not in param_name or "layernorm" in param_name:
            continue

        # Parse layer and module
        parts = param_name.split("layers.")[1].split(".")
        layer = int(parts[0])
        param_name = ".".join(parts[1:]).replace(".weight", "")

        # Create color (value is already -1 to 1)
        color = (
            max(0, -value),  # red
            max(0, value),  # green
            0.0,  # blue
        )
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
