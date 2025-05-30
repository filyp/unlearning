# %%
# %load_ext autoreload
# %autoreload 2
import json
import logging
import os
import random
from copy import deepcopy

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch as pt
import torch.nn.functional as F
import umap
from datasets import Dataset, load_dataset
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
pt.set_default_device("cuda")

# %%

# ! setup
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)
model.config.use_cache = False

# %%

lim = 1000

w = model.model.layers[-1].mlp.down_proj.weight
vectors = w.T.float().cpu().detach().numpy()

# w2 = model.model.layers[10].mlp.down_proj.weight
# # %%
# w = pt.concatenate([w, w2], axis=1)
# w.shape

# ind = 7
# pt.cosine_similarity(w[:, ind], w[:, ind+100], dim=0)


# w = model.model.layers[9].mlp.gate_proj.weight
# vectors = w.float().cpu().detach().numpy()

# # %%
# # shuffle indices
# indices = np.arange(vectors.shape[0])
# np.random.shuffle(indices)
# vectors = vectors[indices]

vectors = vectors[:lim]
vectors.shape

# # %%

# reducer = umap.UMAP(n_components=1, n_neighbors=50)
# vectors_1d = reducer.fit_transform(vectors)
# # plot vectors_1d
# plt.plot(vectors_1d, "o", markersize=1)
# plt.show()

# %%
alignments = model.lm_head.weight @ w
ids = alignments.abs().max(dim=0).indices
words = [tokenizer.decode(ids[i]) for i in range(ids.shape[0])]

# %%
# Assuming your vectors are in a numpy array called 'vectors'
# vectors shape: (8000, 2000)
# Reduce to 3D using UMAP
# reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=100, metric="cosine")
reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=100, negative_sample_rate=10)
vectors_3d = reducer.fit_transform(vectors)


# If not already, convert your vectors_3d to a DataFrame
df = pd.DataFrame(vectors_3d, columns=["UMAP1", "UMAP2", "UMAP3"])
df["index"] = np.arange(len(df))
df["words"] = words[:lim]

fig = px.scatter_3d(
    df,
    x="UMAP1",
    y="UMAP2",
    z="UMAP3",
    color="index",  # or use another feature/label if you have one
    title="Interactive 3D UMAP projection of vectors",
    opacity=0.4,
    color_continuous_scale="Viridis",
    hover_data=["words"],
    # width=1200,  # Set width
    # height=800,  # Set height
)

# Force display in browser with larger size
fig.show(renderer="browser")

# %%
# # %%
# # Create hollow 3D sphere of points
# # 8000 points, 2000 dimensions, only first 3 dimensions set, rest are zeros

# def create_hollow_sphere_points(n_points=8000, n_dimensions=2000, radius=1.0):
#     """
#     Create points uniformly distributed on the surface of a 3D sphere.
#     Only the first 3 dimensions are set, the rest are zeros.
    
#     Args:
#         n_points: Number of points to generate (default: 8000)
#         n_dimensions: Total number of dimensions (default: 2000)
#         radius: Radius of the sphere (default: 1.0)
    
#     Returns:
#         numpy array of shape (n_points, n_dimensions)
#     """
#     # Initialize array with zeros
#     points = np.zeros((n_points, n_dimensions))
    
#     # Generate random points on unit sphere surface using normal distribution
#     # This method ensures uniform distribution on sphere surface
#     sphere_points = np.random.randn(n_points, 2)
    
#     # Normalize to unit sphere
#     norms = np.linalg.norm(sphere_points, axis=1, keepdims=True)
#     sphere_points = sphere_points / norms
    
#     # Scale by radius
#     sphere_points *= radius
    
#     # Set only the first n dimensions
#     points[:, :2] = sphere_points
    
#     return points

# # Generate the hollow sphere points
# np.random.seed(42)  # For reproducibility
# vectors = create_hollow_sphere_points(n_points=8000, n_dimensions=2000, radius=1.0)

# Plot histogram of distances to

# # %%
# # 2d plot
# reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=100)
# vectors_2d = reducer.fit_transform(vectors)
# df = pd.DataFrame(vectors_2d, columns=["UMAP1", "UMAP2"])
# df["index"] = np.arange(len(df))
# fig = px.scatter(
#     df,
#     x="UMAP1",
#     y="UMAP2",
#     color="index",
#     title="Interactive 2D UMAP projection of vectors",
# )
# fig.show(renderer="browser")

# # %%
# import pyvista as pv
# # Set backend for browser display
# pv.set_jupyter_backend('trame')  # Modern web-based backend
# # Create point cloud
# points = df[['UMAP1', 'UMAP2', 'UMAP3']].values
# point_cloud = pv.PolyData(points)
# point_cloud['colors'] = df['index'].values
# # Create plotter for browser
# plotter = pv.Plotter()
# plotter.add_mesh(point_cloud, scalars='colors', point_size=8, 
#                 render_points_as_spheres=True, cmap='viridis')
# plotter.add_scalar_bar(title="Index")
# # Show in browser
# plotter.show(jupyter_backend='trame')  # Opens in browser