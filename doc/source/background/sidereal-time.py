from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pyTMD.interpolate import slerp


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


def linterp(a, b, radius=1.0, n=100):
    """
    Interpolate between two points on a circle
    """
    xa, ya = cartesian(a, radius=radius)
    xb, yb = cartesian(b, radius=radius)
    x, y, _ = slerp(xa, ya, 0, xb, yb, 0, n=n)
    return x, y


fig, ax = plt.subplots(figsize=(6, 6), facecolor="#fcfcfc")
# quiver arrow radius
arrow_radius = 5.0

# equation of the equinoxes
eqeq = 8.0
# mean vernal equinox
ve_mean = 48.0
# apparent vernal equinox
ve_apparent = ve_mean - eqeq
# Greenwich prime meridian
pm = 0.0
# observer longitude
lmda = -32.0

# central point
ax.scatter(0, 0, color="k", s=5)

# Earth
radius = 0.3
color = "dodgerblue"
x, y = cartesian(np.linspace(0, 180, 180), radius=radius)
ax.fill_between(x, y, y2=-y, color=color, alpha=0.3)

# Earth rotation direction
radius = 0.75
color = "dodgerblue"
x, y = linterp(270 - 30, 270 + 30, radius=radius)
ax.plot(x, y, color=color)
ax.annotate(
    "",
    xy=(x[-1], y[-1]),
    xytext=(x[-2], y[-2]),
    color=color,
    arrowprops=dict(arrowstyle="->", color=color, mutation_scale=15),
)
ax.text(
    x.mean(),
    y.mean() - 0.1,
    "Earth\nrotation",
    color=color,
    ha="center",
    va="top",
)

# mean vernal equinox
color = "mediumseagreen"
x, y = cartesian(ve_mean, arrow_radius)
ax.annotate(
    "",
    xy=(0, 0),
    xytext=(x, y),
    color=color,
    arrowprops=dict(arrowstyle="<-", color=color, mutation_scale=15),
)
ax.text(x, y + 0.1, "Mean \u2648", color=color, ha="left", va="bottom")

# true (apparent) vernal equinox
color = "red"
x, y = cartesian(ve_apparent, arrow_radius)
ax.annotate(
    "",
    xy=(0, 0),
    xytext=(x, y),
    color=color,
    arrowprops=dict(arrowstyle="<-", color=color, mutation_scale=15),
)
ax.text(x, y + 0.1, "True \u2648", color=color, ha="left", va="bottom")

# prime meridian (Greenwich)
color = "0.4"
x, y = cartesian(pm, arrow_radius)
ax.annotate(
    "",
    xy=(0, 0),
    xytext=(x, y),
    color=color,
    arrowprops=dict(arrowstyle="<-", color=color, mutation_scale=15),
)
ax.text(x + 0.1, y, "Greenwich\nMeridian", color=color, ha="left", va="center")

# observer meridian
color = "darkorchid"
x, y = cartesian(lmda, arrow_radius)
ax.annotate(
    "",
    xy=(0, 0),
    xytext=(x, y),
    color=color,
    arrowprops=dict(arrowstyle="<-", color=color, mutation_scale=15),
)
ax.text(x + 0.1, y, "Observer", color=color, ha="left", va="top")

# longitude: observer to prime meridian
radius = 1.25
color = "0.4"
x, y = linterp(lmda, pm, radius=radius)
ax.plot(x, y, color=color)
ax.text(
    x.mean(),
    y.mean(),
    r"$\lambda$",
    color=color,
    ha="center",
    va="center",
    bbox=dict(boxstyle="square,pad=0", ec="w", fc="w"),
)

# LMST: observer to mean vernal equinox
radius = 2.0
color = "darkorchid"
x, y = linterp(lmda, ve_mean, radius=radius)
ax.plot(x, y, color=color)
ax.annotate(
    "",
    xy=(x[-1], y[-1]),
    xytext=(x[-2], y[-2]),
    color=color,
    arrowprops=dict(arrowstyle="->", color=color, mutation_scale=15),
)
ax.text(
    x.mean(),
    y.mean(),
    "LMST",
    color=color,
    ha="center",
    va="center",
    bbox=dict(boxstyle="square,pad=0", ec="w", fc="w"),
)
# reference circle for LMST
x, y = cartesian(np.linspace(0, 360, 180), radius=radius)
ax.plot(x, y, color=color, lw=0.7, ls="--", alpha=0.2)

# GMST: prime meridian to mean vernal equinox
radius = 2.75
color = "mediumseagreen"
x, y = linterp(pm, ve_mean, radius=radius)
ax.plot(x, y, color=color)
ax.annotate(
    "",
    xy=(x[-1], y[-1]),
    xytext=(x[-2], y[-2]),
    color=color,
    arrowprops=dict(arrowstyle="->", color=color, mutation_scale=15),
)
ax.text(
    x.mean(),
    y.mean(),
    "GMST",
    color=color,
    ha="center",
    va="center",
    bbox=dict(boxstyle="square,pad=0", ec="w", fc="w"),
)
# reference circle for GMST
x, y = cartesian(np.linspace(0, 360, 180), radius=radius)
ax.plot(x, y, color=color, lw=0.7, ls="--", alpha=0.2)

# GAST: prime meridian to true (apparent) vernal equinox
radius = 3.5
color = "red"
x, y = linterp(pm, ve_apparent, radius=radius)
ax.plot(x, y, color=color)
ax.annotate(
    "",
    xy=(x[-1], y[-1]),
    xytext=(x[-2], y[-2]),
    color=color,
    arrowprops=dict(arrowstyle="->", color=color, mutation_scale=15),
)
ax.text(
    x.mean(),
    y.mean(),
    "GAST",
    color=color,
    ha="center",
    va="center",
    bbox=dict(boxstyle="square,pad=0", ec="w", fc="w"),
)
# reference circle for GAST
x, y = cartesian(np.linspace(0, 360, 180), radius=radius)
ax.plot(x, y, color=color, lw=0.7, ls="--", alpha=0.2)

# equation of equinoxes (eqeq)
# difference between mean and apparent vernal equinox
radius = 4.25
color = "darkorange"
x, y = linterp(ve_apparent, ve_mean, radius=radius)
ax.plot(x, y, color=color)
ax.text(
    x.mean(),
    y.mean(),
    r"$E_e$",
    color=color,
    ha="center",
    va="center",
    bbox=dict(boxstyle="square,pad=0", ec="w", fc="w"),
)

# set the axes facecolor
ax.set_facecolor("#fcfcfc")
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout()
plt.show()
