# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
d = 2
a = np.random.randn(d)
b = np.random.randn(d)

# only test negative cossim case, because for positive, A-GEM does nothing
if a @ b > 0:
    b = -b

# A-GEM
c = b - (a @ b) / (a @ a) * a

# Disruption Masking
d = b.copy()
d[np.sign(a) != np.sign(b)] = 0

# Plot vectors as arrows from origin
plt.figure(figsize=(8, 8))
plt.quiver(0, 0, a[0], a[1], angles="xy", scale_units="xy", scale=1, color="b", label="a")
plt.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale=1, color="r", label="b")
plt.quiver(0, 0, c[0], c[1], angles="xy", scale_units="xy", scale=1, color="g", label="A-GEM")
plt.quiver(0, 0, d[0], d[1], angles="xy", scale_units="xy", scale=1, color="y", label="Disruption Masking")

# Set equal aspect ratio and add grid
plt.axis("equal")

# Add labels and legend
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Vectors a and b")

# Set reasonable axis limits
max_val = max(abs(a).max(), abs(b).max())
plt.xlim(-max_val * 1.2, max_val * 1.2)
plt.ylim(-max_val * 1.2, max_val * 1.2)


# %%
# ratios = []
for _ in range(1000):
    d = 10000
    a = np.random.randn(d)
    b = np.random.randn(d)

    # only test negative cossim case, because for positive, A-GEM does nothing
    if a @ b > 0:
        b = -b

    # A-GEM
    c = b - (a @ b) / (a @ a) * a

    # Disruption Masking
    d = b.copy()
    d[np.sign(a) != np.sign(b)] = 0

    # print(np.linalg.norm(c), np.linalg.norm(d))
    # ratios.append(np.linalg.norm(d) / np.linalg.norm(c))

    # turns out the each vector position's absolute value is strictly smaller for Disruption Masking
    assert (np.abs(c) > np.abs(d)).all()
    assert (np.abs(b) >= np.abs(d)).all()
    # assert (np.abs(b) >= np.abs(c)).all()  # this one fails!

# %%
