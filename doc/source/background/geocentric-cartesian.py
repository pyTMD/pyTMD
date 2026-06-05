import pyTMD
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig, ax = plt.subplots(
    num=1, figsize=(4.5, 4.5), subplot_kw={"projection": "3d"}
)

# extend of quiver arrows
quiver_extend = 1.25

# circles for the sphere
lons = np.linspace(0, 360, 360)
lats = np.linspace(-90, 90, 180)

# geocenter
ax.scatter(0, 0, 0, color="k", s=5)

lon = np.radians(70.0)  # longitude in radians
lat = np.radians(30.0)  # latitude in radians
x, y, z = pyTMD.astro._cartesian(lat, lon)
ax.scatter(x, y, z, color="mediumseagreen", s=5)
ax.quiver(
    0,
    0,
    0,
    x,
    y,
    z,
    color="mediumseagreen",
    lw=0.8,
    arrow_length_ratio=0.07,
)
ax.text(
    x + 0.15,
    y + 0.15,
    z + 0.15,
    "(\u03c6, \u03bb)",
    ha="center",
    va="center",
    fontsize=9,
    color="mediumseagreen",
    bbox=dict(boxstyle="square,pad=0", ec="w", fc="w", alpha=0.8),
)
# meridian from pole to position
mu = pyTMD.interpolate.slerp(0, 0, 1, x, y, z)
ml = pyTMD.interpolate.slerp(x, y, z, np.cos(lon), np.sin(lon), 0.0)
ax.plot(
    *mu,
    color="mediumseagreen",
    lw=0.8,
    ls="--",
)
ax.plot(
    *ml,
    color="mediumseagreen",
    lw=0.8,
    ls="--",
)

# cartesian coordinates
for i in range(4):
    j = i % 2
    k = i // 2
    ax.plot(
        [0, x],
        [j * y, j * y],
        [k * z, k * z],
        color="darkorchid",
        lw=0.8,
        ls="--",
    )
    ax.plot(
        [j * x, j * x],
        [0, y],
        [k * z, k * z],
        color="darkorange",
        lw=0.8,
        ls="--",
    )
    ax.plot(
        [j * x, j * x],
        [k * y, k * y],
        [0, z],
        color="red",
        lw=0.8,
        ls="--",
    )
ax.text(
    0.5 * x,
    y,
    0,
    "X",
    ha="center",
    va="center",
    fontsize=9,
    color="darkorchid",
    bbox=dict(boxstyle="square,pad=0", ec="w", fc="w", alpha=0.8),
)
ax.text(
    x,
    0.5 * y,
    0,
    "Y",
    ha="center",
    va="center",
    fontsize=9,
    color="darkorange",
    bbox=dict(boxstyle="square,pad=0", ec="w", fc="w", alpha=0.8),
)
ax.text(
    x,
    y,
    0.5 * z,
    "Z",
    ha="center",
    va="center",
    fontsize=9,
    color="red",
    bbox=dict(boxstyle="square,pad=0", ec="w", fc="w", alpha=0.8),
)
ax.text(
    0.5 * x,
    0.5 * y,
    0.5 * z + 0.05,
    "r",
    ha="center",
    va="center",
    fontsize=9,
    color="mediumseagreen",
    bbox=dict(boxstyle="square,pad=0", ec="w", fc="w", alpha=0.8),
)

# parallels at 30 degree intervals
for p in np.arange(-60, 90, 30):
    x, y, z = pyTMD.astro._cartesian(np.radians(p), np.radians(lons))
    ax.plot(x, y, z, color="0.4", lw=0.5)

# meridians at 30 degree intervals
for m in np.arange(0, 360, 30):
    x, y, z = pyTMD.astro._cartesian(np.radians(lats), np.radians(m))
    ax.plot(x, y, z, color="0.4", lw=0.5)

# cartesian axes
ax.quiver(
    0, 0, 0, quiver_extend, 0, 0, color="k", lw=0.5, arrow_length_ratio=0.07
)
ax.quiver(
    0, 0, 0, 0, quiver_extend, 0, color="k", lw=0.5, arrow_length_ratio=0.07
)
ax.quiver(
    0, 0, 0, 0, 0, quiver_extend, color="k", lw=0.5, arrow_length_ratio=0.07
)

ax.text(
    quiver_extend + 0.1,
    0.0,
    0.0,
    "x",
    ha="center",
    va="bottom",
    fontsize=9,
    color="k",
)
ax.text(
    0.0,
    quiver_extend + 0.1,
    0.0,
    "y",
    ha="center",
    va="bottom",
    fontsize=9,
    color="k",
)
ax.text(
    0.0,
    0.0,
    quiver_extend + 0.1,
    "z",
    ha="center",
    va="bottom",
    fontsize=9,
    color="k",
)

# set the aspect ratio and view angle
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.8, 0.8)
ax.set_zlim(-0.8, 0.8)
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=20, azim=45)
ax.set_axis_off()

fig.tight_layout()
plt.show()
