from __future__ import annotations

import pyTMD
import numpy as np
import matplotlib.pyplot as plt

# adjust style
facecolor = "#fcfcfc"
plt.rcParams["figure.facecolor"] = facecolor
plt.rcParams["axes.facecolor"] = facecolor
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Lato"]


def cartesian(
    phi: float | np.ndarray,
    radius: float | np.ndarray = 1.0,
):
    """
    Convert from polar coordinates to Cartesian coordinates

    Parameters
    ----------
    phi: float or np.ndarray
        angular coordinate(s) in degrees
    radius: float or np.ndarray, default 1.0
        radial coordinate(s)

    Returns
    -------
    x: np.ndarray
        Cartesian x-coordinates (units of radius)
    y: np.ndarray
        Cartesian y-coordinates (units of radius)
    """
    x = radius * np.cos(np.radians(phi))
    y = radius * np.sin(np.radians(phi))
    return x, y


# create figure and subplots
fig, ax1 = plt.subplots(ncols=2, figsize=(10, 5))
# bounding box style
bboxpad = dict(boxstyle="square,pad=0.1", ec=facecolor, fc=facecolor)

# central point
for ax in ax1:
    ax.scatter(0, 0, color="k", s=5)
    ax.axvline(0.0, color="0.4", linewidth=0.5, linestyle="--", dashes=(5, 5))
    ax.axhline(0.0, color="0.4", linewidth=0.5, linestyle="--", dashes=(5, 5))

# Earth
rad_e = 0.1
x, y = cartesian(np.linspace(0, 180, 180), radius=rad_e)
ax1[0].fill_between(x, y, y2=-y, color="dodgerblue", alpha=0.3)
ax1[0].text(
    0,
    -0.15,
    "Earth",
    color="dodgerblue",
    horizontalalignment="center",
    verticalalignment="top",
    fontsize=10,
    bbox=bboxpad,
)

# lunar ellipsoidal parameters
a_axis = 1.0  # semi-major axis
ecc = 0.25  # eccentricity of ellipse
b_axis = a_axis * np.sqrt(1.0 - ecc**2)  # semi-minor axis
focus = np.sqrt(a_axis**2 - b_axis**2)
xy = (-focus, 0.0)  # center point

# actual lunar orbit
x, y = pyTMD.ellipse._xy(a_axis, b_axis, 0, xy=xy, N=180)
ax1[0].plot(x, y, color="red", linewidth=0.8, linestyle="--")
ax1[0].annotate(
    "True Lunar\nOrbit",
    xy=(x[20], y[20]),
    xytext=(0.25, 0.4),
    color="red",
    horizontalalignment="center",
    verticalalignment="top",
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="red", mutation_scale=15),
)

# lunar mean distance
distance = np.sqrt(0.5 * (a_axis**2 + b_axis**2))
x, y = cartesian(np.linspace(0, 360, 180), radius=distance)
ax1[0].plot(x, y, color="0.4", linewidth=0.7, linestyle="--")
ax1[0].annotate(
    "Average Lunar\nDistance",
    xy=(x[110], y[110]),
    xytext=(-0.4, -0.4),
    color="0.4",
    horizontalalignment="center",
    verticalalignment="top",
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="0.4", mutation_scale=15),
)

# exaggerated size of the moon
rad_m = 0.03
# circle for the moon
x, y = cartesian(np.linspace(0, 180, 180), radius=rad_m)

# Apogee
apogee_x = -a_axis + xy[0]
ax1[0].fill_between(apogee_x + x, y, y2=-y, color="mediumseagreen", alpha=0.3)
ax1[0].annotate(
    "",
    xy=(apogee_x, 0),
    xytext=(-rad_e, 0),
    color="mediumseagreen",
    horizontalalignment="right",
    verticalalignment="center",
    arrowprops=dict(
        arrowstyle="<->", color="mediumseagreen", mutation_scale=15
    ),
)
ax1[0].text(
    (apogee_x - rad_e) / 2.0,
    0.01,
    "Apogee",
    color="mediumseagreen",
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=10,
)

# Perigee
perigee_x = a_axis + xy[0]
ax1[0].fill_between(perigee_x + x, y, y2=-y, color="darkorchid", alpha=0.3)
ax1[0].annotate(
    "",
    xy=(perigee_x, 0),
    xytext=(rad_e, 0),
    color="darkorchid",
    horizontalalignment="right",
    verticalalignment="center",
    arrowprops=dict(arrowstyle="<->", color="darkorchid", mutation_scale=15),
)
ax1[0].text(
    (perigee_x + rad_e) / 2.0,
    0.01,
    "Perigee",
    color="darkorchid",
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=10,
)

# Sun
rad_s = 0.1
x, y = cartesian(np.linspace(0, 180, 180), radius=rad_s)
ax1[1].fill_between(x, y, y2=-y, color="darkorange", alpha=0.3)
ax1[1].text(
    0,
    -0.15,
    "Sun",
    color="darkorange",
    horizontalalignment="center",
    verticalalignment="top",
    fontsize=10,
    bbox=bboxpad,
)


# solar ellipsoidal parameters
a_axis = 1.0  # semi-major axis
ecc = 0.25  # eccentricity of ellipse
b_axis = a_axis * np.sqrt(1.0 - ecc**2)  # semi-minor axis
focus = np.sqrt(a_axis**2 - b_axis**2)
xy = (focus, 0.0)  # center point

# actual orbit
x, y = pyTMD.ellipse._xy(a_axis, b_axis, 0, xy=xy, N=180)
ax1[1].plot(x, y, color="dodgerblue", linewidth=0.8, linestyle="--")
ax1[1].annotate(
    "True Earth\nOrbit",
    xy=(x[70], y[70]),
    xytext=(-0.25, 0.4),
    color="dodgerblue",
    horizontalalignment="center",
    verticalalignment="top",
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="dodgerblue", mutation_scale=15),
)

# Earth mean distance
distance = np.sqrt(0.5 * (a_axis**2 + b_axis**2))
x, y = cartesian(np.linspace(0, 360, 180), radius=distance)
ax1[1].plot(x, y, visible=False)

# exaggerated size of the earth
rad_e = 0.02
# circle for the earth
x, y = cartesian(np.linspace(0, 180, 180), radius=rad_e)

# Aphelion
aphelion_x = a_axis + xy[0]
ax1[1].fill_between(aphelion_x + x, y, y2=-y, color="mediumseagreen", alpha=0.3)
ax1[1].annotate(
    "",
    xy=(aphelion_x, 0),
    xytext=(rad_s, 0),
    color="mediumseagreen",
    horizontalalignment="right",
    verticalalignment="center",
    arrowprops=dict(
        arrowstyle="<->", color="mediumseagreen", mutation_scale=15
    ),
)
ax1[1].text(
    (aphelion_x + rad_s) / 2.0,
    0.01,
    "Aphelion",
    color="mediumseagreen",
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=10,
)

# Perihelion
perihelion_x = -a_axis + xy[0]
ax1[1].fill_between(perihelion_x + x, y, y2=-y, color="darkorchid", alpha=0.3)
ax1[1].annotate(
    "",
    xy=(perihelion_x, 0),
    xytext=(-rad_s, 0),
    color="darkorchid",
    horizontalalignment="right",
    verticalalignment="center",
    arrowprops=dict(arrowstyle="<->", color="darkorchid", mutation_scale=15),
)
ax1[1].text(
    (perihelion_x - rad_s) / 2.0,
    0.01,
    "Perihelion",
    color="darkorchid",
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=10,
)

# set x and y limits
# turn off axes
ax1[0].set_xlim(-1.35, 1.0)
ax1[1].set_xlim(-1.0, 1.35)
for ax in ax1:
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

# set axis limits and show plot
fig.tight_layout()
fig.subplots_adjust(wspace=-0.1)
plt.show()
