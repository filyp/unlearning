
# %%

center_wmdp, center_disr = _eval_callback(orig_model)
print(f"{center_wmdp=:.4f} {center_disr=:.4f}")


_d_ref = 0.001
_w_ref = -0.10

# Parse labels and create color mapping
layer_module_colors = []
for param_name in param_names:
    if "layers." not in param_name:
        continue
    if "layernorm" in param_name:
        continue
    w = float(wmdp_accs[param_name] - center_wmdp)
    d = float(disr_losses[param_name] - center_disr)

    # Parse layer and module
    parts = param_name.split("layers.")[1].split(".")
    layer = int(parts[0])
    param_name = ".".join(parts[1:]).replace(".weight", "")
    # Create color
    color = (
        min(1.0, max(0.0, d / _d_ref)),  # red
        min(1.0, max(0.0, w / _w_ref)),  # green
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
fig, ax = plt.subplots(figsize=(4.5, 10))
ax.set_axis_off()

# Calculate grid dimensions
cell_height = 1
cell_width = 1.5
height = len(layer_nums) * cell_height
width = len(unique_modules) * cell_width

# Add padding around the plot
plt.subplots_adjust(left=0.2, right=0.8, top=0.95, bottom=0.15)

# Add title with question info
title_text = f"Model: {conf.model_id}\n{q['question']}"
ax.text(
    width / 2,
    height + 0.5,
    title_text,
    ha="center",
    va="bottom",
    wrap=True,
    # fontsize=8,  # Smaller font size for better wrapping
)

# Draw cells
for i, layer in enumerate(layer_nums):
    for j, param_name in enumerate(unique_modules):
        color = color_matrix[layer][param_name]
        rect = plt.Rectangle(
            (j * cell_width, (len(layer_nums) - 1 - i) * cell_height),
            cell_width - 0.1,
            cell_height - 0.1,
            facecolor=color,
        )
        ax.add_patch(rect)

    # Add layer number on the left
    ax.text(
        -0.3,
        (len(layer_nums) - 1 - i) * cell_height + cell_height / 2,
        f"Layer {layer}",
        ha="right",
        va="center",
    )

# Add module labels at the bottom
for j, param_name in enumerate(unique_modules):
    ax.text(
        j * cell_width + cell_width / 2,
        -0.3,  # Increased spacing below the grid
        param_name,
        ha="center",  # Changed to center
        va="top",
        rotation=90,
    )

# Set the plot limits with more padding
plt.xlim(-1, width + 0.5)
plt.ylim(-2, height + 2.5)  # More space for bottom labels and title

# Save the plot with a tight layout and explicit bbox
os.makedirs(f"../plots/single_question_per_module", exist_ok=True)
name = f"question={question_index}_masking={conf.masking}"
plt.savefig(
    f"../plots/single_question_per_module/{name}.pdf",
    bbox_inches="tight",
    pad_inches=0.8,
    dpi=300,
)

# %% legend

# # Create a new figure for the legend
# fig_legend, ax_legend = plt.subplots(figsize=(4, 4))

# # Create a grid of points
# n_points = 100
# x = np.linspace(-_w_ref, 0, n_points)  # WMDP change
# y = np.linspace(0, _d_ref, n_points)  # Disruption change
# X, Y = np.meshgrid(x, y)

# # Create color array
# colors = np.zeros((n_points, n_points, 3))
# colors[:, :, 0] = Y / _d_ref  # red component
# colors[:, :, 1] = -X / _w_ref  # green component
# # blue stays 0

# # Plot the color map
# ax_legend.imshow(colors, extent=[-_w_ref, 0, 0, _d_ref], origin="lower", aspect="auto")

# # Add labels and title
# ax_legend.set_xlabel("WMDP Accuracy Change")
# ax_legend.set_ylabel("Disruption Loss Change")
# ax_legend.set_title("Color Legend")

# # Add gridlines
# ax_legend.grid(True, color="white", alpha=0.3)

# plt.tight_layout()
