# %%

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# Example data: curves with different speeds/lengths
curves = [
    # (np.array([1, 2, 2, 1.8, 3, 4]), np.array([1, 2, 3, 4, 4.2, 4.5])),
    (np.array([0.8, 2.1, 2.2, 2.1, 1.5]), np.array([0.9, 2.2, 3.1, 3.8, 4.0])),
    # (np.array([1.2, 1.9, 2.1, 1.7, 2.9, 3.5, 4.2]), np.array([1.1, 1.8, 2.9, 4.2, 4.3, 4.4, 4.6]))
]

# %%
# print curves
for x, y in curves:
    plt.plot(x, y)
plt.legend()
plt.show()

# %%

x , y = curves[0]
f = interpolate.interp1d(x, y, kind='linear', bounds_error=False)
space = np.linspace(0, 4, 100)
sim_y = f(space)
plt.plot(space, sim_y)
