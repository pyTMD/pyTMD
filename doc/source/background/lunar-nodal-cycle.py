import pyTMD
import timescale
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig, ax = plt.subplots(
    num=1,
    figsize=(4.5, 4.5),
    subplot_kw={"projection": "3d"},
    facecolor="#fcfcfc",
)

# quiver arrow radius
arrow_radius = 1.25

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
# lunar and earth radius
rad_e = 6.3781e6
rad_m = 1.7375e6

# celestial center
ax.scatter(0, 0, 0, color="k", s=5)

# Earth surface
ph, th = np.meshgrid(np.radians(lons), np.radians(lats))
X, Y, Z = pyTMD.astro._cartesian(th, ph, radius=0.1)
ax.plot_surface(X, Y, Z, color="dodgerblue", alpha=0.3)

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

# celestial pole
ax.quiver(
    0, 0, 0, 0, 0, arrow_radius, color="k", lw=0.5, arrow_length_ratio=0.07
)
ax.quiver(
    0, 0, 0, 0, 0, -arrow_radius, color="k", lw=0.5, arrow_length_ratio=0.07
)
ax.text(
    0.0,
    0.0,
    arrow_radius + 0.1,
    "NCP",
    ha="center",
    va="bottom",
    fontsize=9,
    color="k",
)
ax.text(
    0.0,
    0.0,
    -arrow_radius - 0.1,
    "SCP",
    ha="center",
    va="top",
    fontsize=9,
    color="k",
)

# lunar surface
radius = 0.1 * (rad_m / rad_e)
LX, LY, LZ = pyTMD.astro._cartesian(th, ph, radius=radius)
# maximum declinations of the Moon
major_standstill = np.radians(28.5)
minor_standstill = np.radians(18.5)
# convert to position vectors
x, y, z = pyTMD.astro._cartesian(
    0, np.radians(lons), inclination=major_standstill
)
above = z >= 0
ax.plot(
    x[above],
    y[above],
    z[above],
    color="red",
    lw=0.8,
    linestyle="--",
    label="Lunar Orbit (Major)",
)
ax.plot(
    x[~above],
    y[~above],
    z[~above],
    color="red",
    lw=0.8,
    linestyle=":",
)
ax.plot_surface(LX + x[90], LY + y[90], LZ + z[90], color="red", alpha=0.3)

# convert to position vectors
x, y, z = pyTMD.astro._cartesian(
    0, np.radians(lons), inclination=minor_standstill
)
above = z >= 0
ax.plot(
    x[above],
    y[above],
    z[above],
    color="darkorchid",
    lw=0.8,
    linestyle="--",
    label="Lunar Orbit (Minor)",
)
ax.plot(
    x[~above],
    y[~above],
    z[~above],
    color="darkorchid",
    lw=0.8,
    linestyle=":",
)
ax.plot_surface(
    LX + x[90], LY + y[90], LZ + z[90], color="darkorchid", alpha=0.3
)

# add legend
ax.legend(loc="lower left", fontsize=9, frameon=False)
# set the axes facecolor
ax.set_facecolor("#fcfcfc")
# set the aspect ratio and view angle
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.8, 0.8)
ax.set_zlim(-0.8, 0.8)
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=20, azim=-45)
ax.set_axis_off()

fig.tight_layout()
plt.show()
