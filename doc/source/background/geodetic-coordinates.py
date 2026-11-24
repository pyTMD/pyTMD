import numpy as np
import matplotlib.pyplot as plt
from pyTMD.interpolate import slerp
import pyTMD.spatial

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
    z: np.ndarray
        Cartesian z-coordinates (units of radius)
    """
    x = radius * np.cos(np.radians(phi))
    z = radius * np.sin(np.radians(phi))
    return x, z


def linterp(a, b, radius=1.0, n=100):
    """
    Interpolate between two points on a circle
    """
    xa, za = cartesian(a, radius=radius)
    xb, zb = cartesian(b, radius=radius)
    x, _, z = slerp(xa, 0, za, xb, 0, zb, n=n)
    return x, z


# use a quarter rotation
N = 1000
ii = np.linspace(0.0, 1.0, N)

latitude = 90.0 * ii
th = np.radians(latitude)
phi = np.zeros((N), dtype=np.float64)

# calculate x and z coordinates for an exaggerated Earth
flat = 1.0 / 5.0
major = 6e6
minor = major - flat * major
x = major * np.cos(th) * np.cos(phi) - minor * np.sin(th) * np.sin(phi)
z = major * np.cos(th) * np.sin(phi) + minor * np.sin(th) * np.cos(phi)

fig, ax = plt.subplots(figsize=(5.5, 4))
bbox = dict(boxstyle="square,pad=0", ec=facecolor, fc=facecolor)

ax.plot(x, z, color="darkorange", linestyle="--")
# add annotations for ellipsoid
d = 800
m = (z[d + 5] - z[d - 5]) / (x[d + 5] - x[d - 5])
rotation = np.degrees(np.arctan(m))
ax.text(
    x[d] + 1e5,
    z[d] + 1e5,
    "Reference Ellipsoid",
    horizontalalignment="center",
    verticalalignment="center",
    color="darkorange",
    rotation=rotation,
)

# add a point above the ellipsoid
P = (50e5, 35e5)
ax.annotate(
    r"P$(\varphi,\lambda,h)$",
    xy=P,
    xytext=(51e5, 36e5),
    ha="left",
    color="k",
)

# convert point to geodetic coordinates and get ECEF of surface point
lon, lat, h = pyTMD.spatial.to_geodetic(P[0], 0, P[1], a_axis=major, flat=flat)
px, py, pz = pyTMD.spatial.to_cartesian(lon, lat, 0, a_axis=major, flat=flat)
geolat = pyTMD.spatial.geocentric_latitude(lat, flat=flat)

# plot a line for the geocentric latitude
ax.plot([0, px], [0, pz], color="darkorchid")
ax.plot(0, 0, ".", color="darkorchid")
sx, sz = linterp(0, geolat, radius=15e5)
ax.plot(sx, sz, color="darkorchid")
xm, zm = cartesian(0.5 * geolat, radius=15e5)
ax.text(
    xm,
    zm,
    r"$\phi$",
    color="darkorchid",
    horizontalalignment="center",
    verticalalignment="center",
    bbox=bbox,
)
xtext, ztext = cartesian(geolat + 3, radius=3e6)
ax.text(
    xtext,
    ztext,
    "Geocentric Latitude",
    color="darkorchid",
    horizontalalignment="center",
    verticalalignment="center",
    rotation=geolat,
    bbox=bbox,
)

# plot a line from the surface to the point
ax.plot([px, P[0]], [pz, P[1]], color="k")
ax.plot(P[0], P[1], ".", color="k")

# plot a line to the center of figure
x0 = px - pz / np.tan(np.radians(lat))
ax.plot([px, x0], [pz, 0], color="mediumseagreen")
ax.plot(x0, 0, ".", color="mediumseagreen")
sx, sz = linterp(0, lat, radius=15e5)
ax.plot(x0 + sx, sz, color="mediumseagreen")
xm, zm = cartesian(0.5 * lat, radius=15e5)
ax.text(
    x0 + xm,
    zm,
    r"$\varphi$",
    color="mediumseagreen",
    horizontalalignment="center",
    verticalalignment="center",
    bbox=bbox,
)
xtext, ztext = cartesian(lat - 3, radius=3e6)
ax.text(
    x0 + xtext,
    ztext,
    "Geodetic Latitude",
    color="mediumseagreen",
    horizontalalignment="center",
    verticalalignment="center",
    rotation=lat,
    bbox=bbox,
)

# show tangent point
# find the tangent point on each ellipsoid
d = np.argmin(np.sqrt((x - px) ** 2 + (z - pz) ** 2))
m = (z[d + 1] - z[d - 1]) / (x[d + 1] - x[d - 1])
# create a tangent line on each ellipsoid
yt = np.zeros((2))
yt[0] = z[d] + m * (x[d - 8] - x[d])
yt[1] = z[d] + m * (x[d + 8] - x[d])
# plot tangent lines
ax.plot([x[d - 8], x[d + 8]], yt, color="k")

# set axes
ax.spines["left"].set_position("zero")
ax.spines["bottom"].set_position("zero")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_aspect("equal", adjustable="box")
# no ticks on the x and y axes
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
fig.tight_layout()
plt.show()
