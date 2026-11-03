import numpy as np
import pyTMD.math
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


# create figure and axis
fig, ax = plt.subplots(num=1, figsize=(6, 3.5))

# Earth
radius = 1.0
phi = np.arange(0, 361)
x, y = cartesian(phi, radius)
ax.fill(x, y, color="dodgerblue", alpha=0.3)
ax.plot(x, y, color="0.4", lw=1.0, zorder=1)

# degree-2 deformation of the surface
epsilon = 0.2
theta = np.radians(phi)
P2 = pyTMD.math._assoc_legendre(2, 0, np.cos(theta))
rd = (radius + epsilon * P2) / radius
ax.plot(
    rd * x,
    rd * y,
    color="mediumseagreen",
    ls="--",
    lw=1.5,
)

# arrows on the surface
phi = np.arange(0, 360, 15)
x, y = cartesian(phi, radius)
u = 2.0 * x
v = -y
ax.quiver(
    x,
    y,
    u,
    v,
    color="darkorchid",
    angles="xy",
    scale_units="xy",
    scale=8.0,
    width=0.005,
)
# arrows within the interior
for inner in [0.3, 0.7]:
    phi = np.arange(0, 360, 30)
    x, y = cartesian(phi, inner)
    u = 2.0 * x
    v = -y
    ax.quiver(
        x,
        y,
        u,
        v,
        color="darkorchid",
        angles="xy",
        scale_units="xy",
        scale=8.0,
        width=0.005,
    )

# direction to the perturbing body
ax.annotate(
    "",
    xy=(2.0, 0.0),
    xytext=(1.3, 0.0),
    arrowprops=dict(arrowstyle="-|>"),
)
ax.text(1.3, 0.05, "Planetary Body", ha="left", va="bottom")
ax.text(-1.3, 0, "Antipodal\nPoint", ha="right", va="center")

# set axis properties
ax.set_aspect("equal")
ax.set_xlim(-2.1, 2.1)
ax.set_ylim(-1.3, 1.3)
ax.axis("off")
# adjust subplot within figure
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
plt.show()
