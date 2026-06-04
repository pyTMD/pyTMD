import pyTMD
import timescale
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig, ax = plt.subplots(
    num=1, figsize=(4.5, 4.5), subplot_kw={"projection": "3d"}
)

# extend of quiver arrows
quiver_extend = 1.25

# Modified Julian Day for calculations
MJD = 61120.5

# create timescale from Modified Julian Day (MJD)
ts = timescale.time.Timescale(MJD=MJD)
# number of days between MJD and the J2000 epoch
_mjd_j2000 = 51544.5
# Julian century
_century = 36525.0
# centuries since J2000 epoch
T = (MJD - _mjd_j2000) / _century

# circles for the celestial sphere
lons = np.linspace(0, 360, 360)
lats = np.linspace(-90, 90, 180)

# celestial center
ax.scatter(0, 0, 0, color="k", s=5)

# parallels at 30 degree intervals
# celestial equator in dark orange
for p in np.arange(-60, 90, 30):
    x, y, z = pyTMD.astro._cartesian(np.radians(p), np.radians(lons))
    if p == 0:
        ax.plot(
            x,
            y,
            z,
            color="darkorange",
            lw=0.8,
            ls="--",
            label="Celestial Equator",
        )
    else:
        ax.plot(x, y, z, color="0.4", lw=0.5)

# meridians at 30 degree intervals
for m in np.arange(0, 360, 30):
    x, y, z = pyTMD.astro._cartesian(np.radians(lats), np.radians(m))
    ax.plot(x, y, z, color="0.4", lw=0.5)

# celestial pole and vernal equinox
ax.quiver(
    0, 0, 0, 0, 0, quiver_extend, color="k", lw=0.5, arrow_length_ratio=0.07
)
ax.quiver(
    0, 0, 0, 0, 0, -quiver_extend, color="k", lw=0.5, arrow_length_ratio=0.07
)
ax.quiver(
    0,
    0,
    0,
    quiver_extend,
    0,
    0,
    color="darkorchid",
    lw=0.5,
    arrow_length_ratio=0.07,
    label="Vernal Equinox",
    zorder=4,
)
ax.text(
    0.0,
    0.0,
    quiver_extend + 0.1,
    "NCP",
    ha="center",
    va="bottom",
    fontsize=9,
    color="k",
)
ax.text(
    0.0,
    0.0,
    -quiver_extend - 0.1,
    "SCP",
    ha="center",
    va="top",
    fontsize=9,
    color="k",
)
ax.text(
    quiver_extend + 0.1,
    0.0,
    0.0,
    "\u2648",
    ha="center",
    va="center",
    fontsize=9,
    color="darkorchid",
)

# obliquity of the ecliptic
epsilon = pyTMD.astro.mean_obliquity(ts.MJD + ts.tt_ut1)
# simple correction for principal nutation (radians)
omega = np.radians(1934.136 * ts.T + 235.0)
epsilon += np.radians(0.00256 * np.cos(omega))
# convert to position vectors
x, y, z = pyTMD.astro._cartesian(0, np.radians(lons), inclination=epsilon)
ax.fill_between(
    0,
    0,
    0,
    x,
    y,
    z,
    edgecolor="mediumseagreen",
    facecolor="mediumseagreen",
    linestyle="--",
    alpha=0.1,
    label="Ecliptic",
)

# add legend
ax.legend(loc="lower left", fontsize=9, frameon=False)
# set the aspect ratio and view angle
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.8, 0.8)
ax.set_zlim(-0.8, 0.8)
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=20, azim=-45)
ax.set_axis_off()

fig.tight_layout()
plt.show()
