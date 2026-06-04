import pyTMD
import timescale
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig, ax1 = plt.subplots(
    num=1, ncols=2, figsize=(9.0, 4.5), subplot_kw={"projection": "3d"}
)

# extend of quiver arrows
quiver_extend = 1.25

# observer position in radians
lat = np.radians(38.992222)  # observer latitude
lon = np.radians(-76.8525)  # observer longitude
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

# compute the right ascension and declination of the moon/sun
lunar_right_ascension, lunar_declination = pyTMD.astro.lunar_equatorial(MJD)
solar_right_ascension, solar_declination = pyTMD.astro.solar_equatorial(MJD)

# greenwich apparent sidereal time in radians
(gast,) = 2.0 * np.pi * pyTMD.astro.gast(T)
# local apparent sidereal time in radians
last = gast + lon
# hour angle in radians
lunar_hour_angle = last - lunar_right_ascension
solar_hour_angle = last - solar_right_ascension

# circles for the celestial sphere
lons = np.linspace(0, 360, 360)
lats = np.linspace(-90, 90, 180)

for ax in fig.axes:
    # celestial center
    ax.scatter(0, 0, 0, color="k", s=5)

    # observer position
    X, Y, Z = pyTMD.astro._cartesian(lat, last)
    ax.scatter(X, Y, Z, color="darkorchid", s=5)
    ax.quiver(
        0,
        0,
        0,
        quiver_extend * X,
        quiver_extend * Y,
        quiver_extend * Z,
        color="darkorchid",
        lw=0.8,
        arrow_length_ratio=0.07,
    )
    ax.text(
        quiver_extend * X + 0.1,
        quiver_extend * Y,
        quiver_extend * Z + 0.1,
        "Observer",
        ha="center",
        va="center",
        fontsize=9,
        color="darkorchid",
        bbox=dict(boxstyle="square,pad=0", ec="w", fc="w", alpha=0.8),
    )

    # body positions
    LX, LY, LZ = pyTMD.astro._cartesian(
        lunar_declination, lunar_right_ascension
    )
    ax.scatter(LX, LY, LZ, color="mediumseagreen", s=5)
    ax.quiver(
        0,
        0,
        0,
        quiver_extend * LX,
        quiver_extend * LY,
        quiver_extend * LZ,
        color="mediumseagreen",
        lw=0.8,
        arrow_length_ratio=0.07,
    )
    ax.text(
        quiver_extend * LX + 0.1,
        quiver_extend * LY,
        quiver_extend * LZ + 0.1,
        "Moon",
        ha="center",
        va="center",
        fontsize=9,
        color="mediumseagreen",
        bbox=dict(boxstyle="square,pad=0", ec="w", fc="w", alpha=0.8),
    )
    # body positions
    SX, SY, SZ = pyTMD.astro._cartesian(
        solar_declination, solar_right_ascension
    )
    ax.scatter(SX, SY, SZ, color="dodgerblue", s=5)
    ax.quiver(
        0,
        0,
        0,
        quiver_extend * SX,
        quiver_extend * SY,
        quiver_extend * SZ,
        color="dodgerblue",
        lw=0.8,
        arrow_length_ratio=0.07,
    )
    ax.text(
        quiver_extend * SX + 0.1,
        quiver_extend * SY + 0.1,
        quiver_extend * SZ,
        "Sun",
        ha="center",
        va="center",
        fontsize=9,
        color="dodgerblue",
        bbox=dict(boxstyle="square,pad=0", ec="w", fc="w", alpha=0.8),
    )

    # meridian from celestial pole to observer position
    mu = pyTMD.interpolate.slerp(0, 0, 1, X, Y, Z)
    ml = pyTMD.interpolate.slerp(X, Y, Z, np.cos(last), np.sin(last), 0.0)
    ax.plot(
        *mu,
        color="darkorchid",
        lw=0.8,
        ls="--",
    )
    ax.plot(
        *ml,
        color="darkorchid",
        lw=0.8,
        ls="--",
    )

    # meridian from celestial pole to body position
    mu = pyTMD.interpolate.slerp(0, 0, 1, LX, LY, LZ)
    ml = pyTMD.interpolate.slerp(
        LX,
        LY,
        LZ,
        np.cos(lunar_right_ascension),
        np.sin(lunar_right_ascension),
        0.0,
    )
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
    mu = pyTMD.interpolate.slerp(0, 0, 1, SX, SY, SZ)
    ml = pyTMD.interpolate.slerp(
        SX,
        SY,
        SZ,
        np.cos(solar_right_ascension),
        np.sin(solar_right_ascension),
        0.0,
    )
    ax.plot(
        *mu,
        color="dodgerblue",
        lw=0.8,
        ls="--",
    )
    ax.plot(
        *ml,
        color="dodgerblue",
        lw=0.8,
        ls="--",
    )

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
        0,
        0,
        0,
        0,
        0,
        -quiver_extend,
        color="k",
        lw=0.5,
        arrow_length_ratio=0.07,
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

# right ascension at the celestial equator
LRAx, LRAy, LRAz = pyTMD.astro._cartesian(0, lunar_right_ascension)
SRAx, SRAy, SRAz = pyTMD.astro._cartesian(0, solar_right_ascension)
# local apparent sidereal time at the celestial equator
EQx, EQy, EQz = pyTMD.astro._cartesian(0, last)

# right ascension to local apparent sidereal time at the celestial equator
# will show the hour angle of the body from the local meridian
hx, hy, hz = pyTMD.interpolate.slerp(LRAx, LRAy, LRAz, EQx, EQy, EQz, n=120)
ax1[0].fill_between(
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
    label="Lunar Hour Angle",
)

# local apparent sidereal time at the celestial equator
hx, hy, hz = pyTMD.interpolate.slerp(SRAx, SRAy, SRAz, EQx, EQy, EQz, n=120)
ax1[0].fill_between(
    0,
    0,
    0,
    hx,
    hy,
    hz,
    edgecolor="red",
    facecolor="red",
    hatch="\\\\",
    alpha=0.1,
    label="Solar Hour Angle",
)

# zenith vector from observer to body
zx, zy, zz = pyTMD.interpolate.slerp(X, Y, Z, LX, LY, LZ, n=120)
ax1[1].fill_between(
    0,
    0,
    0,
    zx,
    zy,
    zz,
    edgecolor="0.4",
    facecolor="0.4",
    hatch="//",
    alpha=0.1,
    label="Lunar Zenith Angle",
)

zx, zy, zz = pyTMD.interpolate.slerp(X, Y, Z, SX, SY, SZ, n=120)
ax1[1].fill_between(
    0,
    0,
    0,
    zx,
    zy,
    zz,
    edgecolor="red",
    facecolor="red",
    hatch="\\\\",
    alpha=0.1,
    label="Solar Zenith Angle",
)

for ax in fig.axes:
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
