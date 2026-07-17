import pyTMD
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# adjust style
facecolor = "#fcfcfc"
plt.rcParams["figure.facecolor"] = facecolor
plt.rcParams["axes.facecolor"] = facecolor
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Lato"]

fig, ax = plt.subplots(
    num=1,
    figsize=(4.5, 4.5),
    subplot_kw={"projection": "3d"},
)

# quiver arrow radius
arrow_radius = 1.25
# bounding box style
bbox = dict(boxstyle="square,pad=0", ec=facecolor, fc=facecolor, alpha=0.8)

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
    linewidth=0.8,
    arrow_length_ratio=0.07,
)
ax.text(
    x + 0.15,
    y + 0.15,
    z + 0.15,
    "(\u03c6, \u03bb)",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=10,
    color="mediumseagreen",
    bbox=bbox,
)
# meridian from pole to position
mu = pyTMD.interpolate.slerp(0, 0, 1, x, y, z)
ml = pyTMD.interpolate.slerp(x, y, z, np.cos(lon), np.sin(lon), 0.0)
ax.plot(
    *mu,
    color="mediumseagreen",
    linewidth=0.8,
    linestyle="--",
)
ax.plot(
    *ml,
    color="mediumseagreen",
    linewidth=0.8,
    linestyle="--",
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
        linewidth=0.8,
        linestyle="--",
    )
    ax.plot(
        [j * x, j * x],
        [0, y],
        [k * z, k * z],
        color="darkorange",
        linewidth=0.8,
        linestyle="--",
    )
    ax.plot(
        [j * x, j * x],
        [k * y, k * y],
        [0, z],
        color="red",
        linewidth=0.8,
        linestyle="--",
    )
ax.text(
    0.5 * x,
    y,
    0,
    "X",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=10,
    color="darkorchid",
    bbox=bbox,
)
ax.text(
    x,
    0.5 * y,
    0,
    "Y",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=10,
    color="darkorange",
    bbox=bbox,
)
ax.text(
    x,
    y,
    0.5 * z,
    "Z",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=10,
    color="red",
    bbox=bbox,
)
ax.text(
    0.5 * x,
    0.5 * y,
    0.5 * z + 0.05,
    "r",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=10,
    color="mediumseagreen",
    bbox=bbox,
)

# fill in the quadrants
x, y, z = pyTMD.interpolate.slerp(1, 0, 0, 0, 1, 0)
ax.fill_between(
    0,
    0,
    0,
    x,
    y,
    z,
    color="0.4",
    hatch="//",
    alpha=0.1,
)

x, y, z = pyTMD.interpolate.slerp(1, 0, 0, 0, 0, 1)
ax.fill_between(
    0,
    0,
    0,
    x,
    y,
    z,
    color="0.4",
    hatch="\\\\",
    alpha=0.1,
)

# parallels at 30 degree intervals
for p in np.arange(-60, 90, 30):
    x, y, z = pyTMD.astro._cartesian(np.radians(p), np.radians(lons))
    # plot the equator in black and the other parallels in gray
    if p == 0:
        ax.plot(x, y, z, color="k", linewidth=0.8)
    else:
        ax.plot(x, y, z, color="0.4", linewidth=0.5)

# meridians at 30 degree intervals
for m in np.arange(0, 360, 30):
    x, y, z = pyTMD.astro._cartesian(np.radians(lats), np.radians(m))
    # plot the prime meridian and 180 degree meridian in black
    # and the other meridians in gray
    if m == 0 or m == 180:
        ax.plot(x, y, z, color="k", linewidth=0.8)
    else:
        ax.plot(x, y, z, color="0.4", linewidth=0.5)

# cartesian axes
ax.quiver(
    0,
    0,
    0,
    arrow_radius,
    0,
    0,
    color="k",
    linewidth=0.5,
    arrow_length_ratio=0.07,
)
ax.quiver(
    0,
    0,
    0,
    0,
    arrow_radius,
    0,
    color="k",
    linewidth=0.5,
    arrow_length_ratio=0.07,
)
ax.quiver(
    0,
    0,
    0,
    0,
    0,
    arrow_radius,
    color="k",
    linewidth=0.5,
    arrow_length_ratio=0.07,
)

ax.text(
    arrow_radius + 0.1,
    0.0,
    0.0,
    "x",
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=10,
    color="k",
)
ax.text(
    0.0,
    arrow_radius + 0.1,
    0.0,
    "y",
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=10,
    color="k",
)
ax.text(
    0.0,
    0.0,
    arrow_radius + 0.1,
    "z",
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=10,
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
