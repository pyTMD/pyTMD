from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pyTMD.interpolate import slerp

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


def linterp(a, b, radius=1.0, n=100):
    """
    Interpolate between two points on a circle
    """
    xa, ya = cartesian(a, radius=radius)
    xb, yb = cartesian(b, radius=radius)
    x, y, _ = slerp(xa, ya, 0, xb, yb, 0, n=n)
    return x, y


fig, ax1 = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 5))
# quiver arrow radius
arrow_radius = 5.0
# bounding box styles
bbox = dict(boxstyle="square,pad=0", ec=facecolor, fc=facecolor)
bboxpad = dict(boxstyle="square,pad=0.1", ec=facecolor, fc=facecolor)

# equation of the equinoxes
eqeq = 10.0
# equation of time
eot = -12.0
# mean vernal equinox
ve_mean = 108.0
# apparent vernal equinox
ve_apparent = ve_mean - eqeq
# mean solar longitude
sun_mean = 48.0
# apparent solar longitude
sun_apparent = sun_mean - eot
# Greenwich prime meridian
pm = 0.0
# observer longitude
lmda = -32.0

for ax in fig.axes:
    # central point
    ax.scatter(0, 0, color="k", s=5)

    # Earth
    radius = 0.3
    color = "dodgerblue"
    x, y = cartesian(np.linspace(0, 180, 180), radius=radius)
    ax.fill_between(x, y, y2=-y, color=color, alpha=0.3)

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
    ax.text(
        x,
        y,
        r"Mean $\Upsilon$",
        color=color,
        horizontalalignment="center",
        verticalalignment="bottom",
    )

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
    ax.text(
        x + 0.1,
        y,
        "Greenwich\nMeridian",
        color=color,
        horizontalalignment="left",
        verticalalignment="center",
    )

# Earth rotation direction
radius = 0.75
center = 270
color = "dodgerblue"
x, y = linterp(center - 30, center + 30, radius=radius)
ax1[0].plot(x, y, color=color)
ax1[0].annotate(
    "",
    xy=(x[-1], y[-1]),
    xytext=(x[-2], y[-2]),
    color=color,
    arrowprops=dict(arrowstyle="->", color=color, mutation_scale=15),
)
xm, ym = cartesian(center, radius=radius)
ax1[0].text(
    xm,
    ym - 0.1,
    "Earth\nrotation",
    color=color,
    horizontalalignment="center",
    verticalalignment="top",
)

# true (apparent) vernal equinox
color = "red"
x, y = cartesian(ve_apparent, arrow_radius)
ax1[0].annotate(
    "",
    xy=(0, 0),
    xytext=(x, y),
    color=color,
    arrowprops=dict(arrowstyle="<-", color=color, mutation_scale=15),
)
ax1[0].text(
    x,
    y,
    r"True $\Upsilon$",
    color=color,
    horizontalalignment="left",
    verticalalignment="bottom",
)

# observer meridian
color = "darkorchid"
x, y = cartesian(lmda, arrow_radius)
ax1[0].annotate(
    "",
    xy=(0, 0),
    xytext=(x, y),
    color=color,
    arrowprops=dict(arrowstyle="<-", color=color, mutation_scale=15),
)
ax1[0].text(
    x + 0.1,
    y,
    "Observer\nMeridian",
    color=color,
    horizontalalignment="left",
    verticalalignment="center",
)

# mean sun
color = "darkorchid"
x, y = cartesian(sun_mean, arrow_radius)
ax1[1].annotate(
    "",
    xy=(0, 0),
    xytext=(x, y),
    color=color,
    arrowprops=dict(arrowstyle="<-", color=color, mutation_scale=15),
)
ax1[1].text(
    x,
    y,
    r"Mean $\odot$",
    color=color,
    horizontalalignment="left",
    verticalalignment="bottom",
)

# true (apparent) solar sun
color = "red"
x, y = cartesian(sun_apparent, arrow_radius)
ax1[1].annotate(
    "",
    xy=(0, 0),
    xytext=(x, y),
    color=color,
    arrowprops=dict(arrowstyle="<-", color=color, mutation_scale=15),
)
ax1[1].text(
    x,
    y,
    r"True $\odot$",
    color=color,
    horizontalalignment="center",
    verticalalignment="bottom",
)

# longitude: observer to prime meridian
radius = 1.25
color = "0.4"
x, y = linterp(lmda, pm, radius=radius)
ax1[0].plot(x, y, color=color)
xm, ym = cartesian((lmda + pm) / 2.0, radius=radius)
ax1[0].text(
    xm,
    ym,
    r"$\lambda$",
    color=color,
    horizontalalignment="center",
    verticalalignment="center",
    bbox=bboxpad,
)

# LMST: observer to mean vernal equinox
radius = 2.0
color = "darkorchid"
x, y = linterp(lmda, ve_mean, radius=radius)
ax1[0].plot(x, y, color=color)
xm, ym = cartesian((lmda + ve_mean) / 2.0, radius=radius)
ax1[0].text(
    xm,
    ym,
    "LMST",
    color=color,
    horizontalalignment="center",
    verticalalignment="center",
    bbox=bboxpad,
)
# reference circle for LMST
x, y = cartesian(np.linspace(0, 360, 180), radius=radius)
ax1[0].plot(x, y, color=color, linewidth=0.7, linestyle="--", alpha=0.2)

# GMST: prime meridian to mean vernal equinox
radius = 3.5
color = "mediumseagreen"
x, y = linterp(pm, ve_mean, radius=radius)
ax1[0].plot(x, y, color=color)
xm, ym = cartesian((pm + ve_mean) / 2.0, radius=radius)
ax1[0].text(
    xm,
    ym,
    "GMST",
    color=color,
    horizontalalignment="center",
    verticalalignment="center",
    bbox=bboxpad,
)
# reference circle for GMST
x, y = cartesian(np.linspace(0, 360, 180), radius=radius)
ax1[0].plot(x, y, color=color, linewidth=0.7, linestyle="--", alpha=0.2)

# GAST: prime meridian to true (apparent) vernal equinox
radius = 2.75
color = "red"
x, y = linterp(pm, ve_apparent, radius=radius)
ax1[0].plot(x, y, color=color)
xm, ym = cartesian((pm + ve_apparent) / 2.0, radius=radius)
ax1[0].text(
    xm,
    ym,
    "GAST",
    color=color,
    horizontalalignment="center",
    verticalalignment="center",
    bbox=bboxpad,
)
# reference circle for GAST
x, y = cartesian(np.linspace(0, 360, 180), radius=radius)
ax1[0].plot(x, y, color=color, linewidth=0.7, linestyle="--", alpha=0.2)

# equation of equinoxes (eqeq)
# difference between mean and apparent vernal equinox
radius = 4.25
color = "darkorange"
x, y = linterp(ve_apparent, ve_mean, radius=radius)
ax1[0].plot(x, y, color=color)
xm, ym = cartesian((ve_apparent + ve_mean) / 2.0, radius=radius)
ax1[0].text(
    xm,
    ym,
    r"$E_e$",
    color=color,
    horizontalalignment="center",
    verticalalignment="center",
    bbox=bbox,
)

# mean solar right ascension: mean vernal equinox to mean sun
radius = 3.5
color = "mediumseagreen"
x, y = linterp(ve_mean, sun_mean, radius=radius)
ax1[1].plot(x, y, color=color)
xm, ym = cartesian((ve_mean + sun_mean) / 2.0, radius=radius)
ax1[1].text(
    xm,
    ym,
    r"$\alpha_M$",
    color=color,
    horizontalalignment="center",
    verticalalignment="center",
    bbox=bboxpad,
)
# reference circle for RA mean
x, y = cartesian(np.linspace(0, 360, 180), radius=radius)
ax1[1].plot(x, y, color=color, linewidth=0.7, linestyle="--", alpha=0.2)

# true solar right ascension: mean vernal equinox to true sun
radius = 2.75
color = "red"
x, y = linterp(ve_mean, sun_apparent, radius=radius)
ax1[1].plot(x, y, color=color)
xm, ym = cartesian((ve_mean + sun_apparent) / 2.0, radius=radius)
ax1[1].text(
    xm,
    ym,
    r"$\alpha_T$",
    color=color,
    horizontalalignment="center",
    verticalalignment="center",
    bbox=bboxpad,
)
# reference circle for RA apparent
x, y = cartesian(np.linspace(0, 360, 180), radius=radius)
ax1[1].plot(x, y, color=color, linewidth=0.7, linestyle="--", alpha=0.2)

# Greenwich mean hour angle (GHA): prime meridian to mean sun
radius = 1.25
color = "darkorchid"
x, y = linterp(sun_mean, pm, radius=radius)
ax1[1].plot(x, y, color=color)
xm, ym = cartesian((sun_mean + pm) / 2.0, radius=radius)
ax1[1].text(
    xm,
    ym,
    r"$\text{GHA}_M$",
    color=color,
    horizontalalignment="center",
    verticalalignment="center",
    bbox=bbox,
)
# reference circle for GHA mean
x, y = cartesian(np.linspace(0, 360, 180), radius=radius)
ax1[1].plot(x, y, color=color, linewidth=0.7, linestyle="--", alpha=0.2)

# Greenwich true hour angle (GHA): prime meridian to true sun
radius = 2.0
color = "0.4"
x, y = linterp(sun_apparent, pm, radius=radius)
ax1[1].plot(x, y, color=color)
xm, ym = cartesian((sun_apparent + pm) / 2.0, radius=radius)
ax1[1].text(
    xm,
    ym,
    r"$\text{GHA}_T$",
    color=color,
    horizontalalignment="center",
    verticalalignment="center",
    bbox=bbox,
)
# reference circle for GHA apparent
x, y = cartesian(np.linspace(0, 360, 180), radius=radius)
ax1[1].plot(x, y, color=color, linewidth=0.7, linestyle="--", alpha=0.2)

# equation of time (eot)
# difference between mean and apparent solar longitude
radius = 4.25
color = "darkorange"
x, y = linterp(sun_apparent, sun_mean, radius=radius)
ax1[1].plot(x, y, color=color)
xm, ym = cartesian((sun_apparent + sun_mean) / 2.0, radius=radius)
ax1[1].text(
    xm,
    ym,
    r"$E_{oT}$",
    color=color,
    horizontalalignment="center",
    verticalalignment="center",
    bbox=bbox,
)

for ax in fig.axes:
    # set the axes facecolor
    ax.set_facecolor(facecolor)
    ax.set_aspect("equal")
    ax.axis("off")

fig.tight_layout()
plt.show()
