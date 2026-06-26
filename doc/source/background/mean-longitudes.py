import numpy as np
import pyTMD.astro
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig, ax1 = plt.subplots(
    num=1,
    ncols=2,
    figsize=(9.0, 4.5),
    subplot_kw={"projection": "3d"},
    facecolor="#fcfcfc",
)

# extend of quiver arrows
quiver_extend = 1.25
# circles for the sphere
lons = np.linspace(0, 360, 360)
lats = np.linspace(-90, 90, 180)
# Modified Julian Day for calculations
MJD = 60500.5
# obliquity of the ecliptic (J2000)
epsilon = np.radians(23.43929111)
# lunar, solar and earth radius
rad_e = 6.3781e6
rad_m = 1.7375e6
rad_s = 6.957e8
# distances between the Earth and the Sun/Moon (meters)
AU = 1.495978707e11
LD = 3.84399e8

# Earth surface
ph, th = np.meshgrid(np.radians(lons), np.radians(lats))
X, Y, Z = pyTMD.astro._cartesian(th, ph, radius=0.1)

for ax in fig.axes:
    ax.plot_surface(X, Y, Z, color="dodgerblue", alpha=0.3)
    # parallels at 30 degree intervals
    for p in np.arange(-60, 90, 30):
        x, y, z = pyTMD.astro._cartesian(
            np.radians(p), np.radians(lons), inclination=-epsilon
        )
        ax.plot(x, y, z, color="0.4", lw=0.5)

    # meridians at 30 degree intervals
    for m in np.arange(0, 360, 30):
        x, y, z = pyTMD.astro._cartesian(
            np.radians(lats),
            np.radians(m),
            inclination=-epsilon,
        )
        ax.plot(x, y, z, color="0.4", lw=0.5)

    # Ecliptic pole
    ax.quiver(
        0,
        0,
        0,
        0,
        0,
        quiver_extend,
        color="mediumseagreen",
        linewidth=0.5,
        arrow_length_ratio=0.07,
    )
    ax.text(
        0.0,
        0.0,
        quiver_extend + 0.1,
        "Ecliptic Pole",
        ha="center",
        va="bottom",
        fontsize=9,
        color="mediumseagreen",
    )
    x, y, z = pyTMD.astro._cartesian(0, np.radians(lons))
    ax.plot(
        x,
        y,
        z,
        color="mediumseagreen",
        lw=0.8,
        ls="--",
        label="Ecliptic",
    )

    # rotation axis
    ax.quiver(
        0,
        0,
        0,
        0,
        quiver_extend * np.sin(epsilon),
        quiver_extend * np.cos(epsilon),
        color="k",
        linewidth=0.5,
        arrow_length_ratio=0.07,
    )
    ax.text(
        0,
        quiver_extend * np.sin(epsilon) + 0.1,
        quiver_extend * np.cos(epsilon) + 0.1,
        "CNP",
        va="center",
        ha="center",
        fontsize=9,
        color="k",
    )

    # vernal equinox
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
        quiver_extend + 0.1,
        0.0,
        0.0,
        "\u2648",
        ha="center",
        va="center",
        fontsize=9,
        color="darkorchid",
    )

# mean longitudes at date
S, H, P, N, Ps = pyTMD.astro.mean_longitudes(MJD, method="ASTRO5")

# lunar surface
radius = 0.1 * (rad_m / rad_e)
X, Y, Z = pyTMD.astro._cartesian(th, ph, radius=radius)
# maximum declinations of the Moon
major_standstill = np.radians(28.5)
inclination = major_standstill - epsilon
# convert to position vectors
x, y, z = pyTMD.astro._cartesian(0, np.radians(lons), inclination=inclination)
ax1[0].plot(
    x,
    y,
    z,
    color="red",
    lw=0.8,
    linestyle="--",
    label="Lunar Orbit",
)
# translate the lunar surface to the mean longitude of the Moon
# along the lunar orbit plane
SX, SY, SZ = pyTMD.astro._cartesian(0, np.radians(S), inclination=inclination)
ax1[0].plot_surface(X + SX[0], Y + SY[0], Z + SZ[0], color="red", alpha=0.3)
ax1[0].text(
    quiver_extend * SX[0],
    quiver_extend * SY[0],
    quiver_extend * SZ[0],
    "Moon",
    ha="center",
    va="center",
    fontsize=9,
    color="red",
    bbox=dict(boxstyle="square,pad=0", ec="w", fc="w", alpha=0.8),
)
# mean longitudes of the Moon (S)
sx, sy, sz = pyTMD.interpolate.slerp(1, 0, 0, SX[0], SY[0], 0, n=120)
ax1[0].fill_between(
    0,
    0,
    0,
    sx,
    sy,
    sz,
    edgecolor="0.4",
    facecolor="0.4",
    hatch="//",
    alpha=0.1,
    label="Lunar Mean Longitude",
)

# solar surface
radius = 0.1 * (rad_s / rad_e) * (LD / AU)
X, Y, Z = pyTMD.astro._cartesian(th, ph, radius=radius)
# translate the solar surface to the mean longitude of the Sun
HX, HY, HZ = pyTMD.astro._cartesian(0, np.radians(H))
ax1[1].plot_surface(X + HX[0], Y + HY[0], Z + HZ[0], color="red", alpha=0.3)
ax1[1].text(
    quiver_extend * HX[0],
    quiver_extend * HY[0],
    quiver_extend * HZ[0],
    "Sun",
    ha="center",
    va="center",
    fontsize=9,
    color="red",
    bbox=dict(boxstyle="square,pad=0", ec="w", fc="w", alpha=0.8),
)
# mean longitudes of the Sun (H)
hx, hy, hz = pyTMD.interpolate.slerp(1, 0, 0, HX[0], HY[0], 0, n=120)
ax1[1].fill_between(
    0,
    0,
    0,
    hx,
    hy,
    hz,
    edgecolor="0.4",
    facecolor="0.4",
    hatch="//",
    alpha=0.1,
    label="Solar Mean Longitude",
)

for ax in fig.axes:
    # set the axes facecolor
    ax.set_facecolor("#fcfcfc")
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
