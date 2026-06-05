import numpy as np
import pyTMD.astro
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig, ax = plt.subplots(
    num=1,
    figsize=(4.5, 4.5),
    subplot_kw={"projection": "3d"},
    facecolor="#fcfcfc",
)

# extend of quiver arrows
quiver_extend = 3.25
# circles for the sphere
lons = np.linspace(0, 360, 360)
lats = np.linspace(-90, 90, 180)
# obliquity of the ecliptic (J2000)
epsilon = np.radians(23.43929111)

# Earth surface
ph, th = np.meshgrid(np.radians(lons), np.radians(lats))
X, Y, Z = pyTMD.astro._cartesian(th, ph)
ax.plot_surface(X, Y, Z, color="dodgerblue", alpha=0.3)

# Precession cone
height = 2.5
ph_cone = np.linspace(0, 2 * np.pi, 40)
h_cone = np.linspace(0, height, 20)
ph_surf, h_surf = np.meshgrid(ph_cone, h_cone)
x_surf = h_surf * np.tan(epsilon) * np.cos(ph_surf)
y_surf = h_surf * np.tan(epsilon) * np.sin(ph_surf)
ax.plot_surface(
    x_surf, y_surf, h_surf, alpha=0.1, color="0.4", antialiased=False
)

# Precession arc
px = height * np.tan(epsilon) * np.cos(np.radians(lons))
py = height * np.tan(epsilon) * np.sin(np.radians(lons))
ax.plot(px, py, height, color="0.4", lw=1.0)
ax.text(-1.2, -0.2, height, "Precession", ha="right", fontsize=9, color="0.4")

# Nutation wiggles
amp = 0.1
omega = 12.0
wave = amp * np.sin(omega * np.radians(lons))
# curve with nutation wiggles
nx = (height * np.tan(epsilon) + wave) * np.cos(np.radians(lons))
ny = (height * np.tan(epsilon) + wave) * np.sin(np.radians(lons))
ax.plot(nx, ny, height, color="darkorchid", lw=1.0)
ax.text(1.2, 0.2, height, "Nutation", ha="left", fontsize=9, color="darkorchid")

# parallels at 30 degree intervals
for p in np.arange(-60, 90, 30):
    x, y, z = pyTMD.astro._cartesian(
        np.radians(p), np.radians(lons), inclination=-epsilon
    )
    ax.plot(x, y, z, color="0.4", lw=0.5)

# meridians at 30 degree intervals
for m in np.arange(0, 360, 30):
    x, y, z = pyTMD.astro._cartesian(
        np.radians(lats), np.radians(m), inclination=-epsilon
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
    linewidth=1.5,
    arrow_length_ratio=0.07,
)
ax.text(
    0.0,
    0.0,
    quiver_extend + 0.1,
    "Ecliptic\nPole",
    ha="center",
    va="bottom",
    fontsize=9,
    color="mediumseagreen",
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
    linewidth=1.5,
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

# set the axes facecolor
ax.set_facecolor("#fcfcfc")
# set the aspect ratio and view angle
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_zlim(-0.2, 2.6)
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=20, azim=-55)
ax.set_axis_off()

fig.tight_layout()
plt.show()
