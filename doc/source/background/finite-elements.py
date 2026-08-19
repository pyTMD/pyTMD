import numpy as np
import matplotlib.pyplot as plt

# adjust style
facecolor = "#fcfcfc"
plt.rcParams["figure.facecolor"] = facecolor
plt.rcParams["axes.facecolor"] = facecolor
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Lato"]

# setup figure and subplots
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

# Linear Elements
# nodes at triangle vertices
N1 = np.array([0.0, 0.0])
N2 = np.array([1.0, 0.0])
N3 = np.array([0.5, np.sqrt(3.0 / 4.0)])
# interior point from some barycentric coordinates
xi, eta = 0.3, 0.5
lmda = 1.0 - eta - xi
P = xi * N1 + eta * N2 + lmda * N3

# plot triangle vertices
xv = [N1[0], N2[0], N3[0], N1[0]]
yv = [N1[1], N2[1], N3[1], N1[1]]
ax[0].plot(xv, yv, color="k", lw=2)

# plot nodes and connections to interior point
r12 = np.sqrt(1.0 / 12.0)
dx = np.array(
    [
        -0.05,
        0.05,
        0.0,
    ]
)
dy = np.array([-0.1 * r12, -0.1 * r12, 0.2 * r12])
nodelabel = [r"$\mathbf{N_1}$", r"$\mathbf{N_2}$", r"$\mathbf{N_3}$"]
for i, node in enumerate([N1, N2, N3]):
    ax[0].plot(
        node[0],
        node[1],
        "o",
        color="0.4",
        markersize=10,
        markeredgecolor="k",
        zorder=3,
    )
    ax[0].plot(
        [P[0], node[0]],
        [P[1], node[1]],
        "--",
        color="0.4",
    )
    ax[0].text(
        node[0] + dx[i],
        node[1] + dy[i],
        nodelabel[i],
        fontsize=12,
        ha="center",
        va="center",
    )

# add interior point
ax[0].plot(
    P[0],
    P[1],
    "*",
    color="red",
    markersize=15,
    zorder=3,
)
ax[0].text(
    P[0] + 0.05,
    P[1] + 0.1 * r12,
    r"$\mathbf{P}$",
    fontsize=12,
    ha="center",
    va="center",
    zorder=3,
)

# plot barycentric areas as sub-triangles
connections = [(P, N1, N2), (P, N2, N3), (P, N3, N1)]
arealabel = [r"$\mathbf{A_3}$", r"$\mathbf{A_1}$", r"$\mathbf{A_2}$"]
color = ["mediumseagreen", "darkorange", "darkorchid"]
for i, node in enumerate([N1, N2, N3]):
    # sub-area vertices
    xs, ys = np.transpose(connections[i])
    ax[0].fill(xs, ys, color=color[i], alpha=0.3)
    # labels at sub-area centroids
    xc, yc = np.mean(xs), np.mean(ys)
    ax[0].text(
        xc,
        yc,
        arealabel[i],
        weight="bold",
        fontsize=12,
        ha="center",
        va="center",
    )


# Quadratic Elements
# use a clockwise ordering scheme
# nodes at triangle vertices
N1 = np.array([0.0, 0.0])
N3 = np.array([1.0, 0.0])
N5 = np.array([0.5, np.sqrt(3.0 / 4.0)])
# nodes at triangle edges
N2 = (N1 + N3) / 2.0
N4 = (N3 + N5) / 2.0
N6 = (N5 + N1) / 2.0

# plot triangle vertices
ax[1].plot(xv, yv, color="k", lw=2)

# plot nodes at vertices
nodelabel = [r"$\mathbf{N_1}$", r"$\mathbf{N_3}$", r"$\mathbf{N_5}$"]
for i, node in enumerate([N1, N3, N5]):
    ax[1].plot(
        node[0],
        node[1],
        "o",
        color="0.4",
        markersize=10,
        markeredgecolor="k",
        zorder=3,
    )
    ax[1].text(
        node[0] + dx[i],
        node[1] + dy[i],
        nodelabel[i],
        fontsize=12,
        ha="center",
        va="center",
    )
# plot nodes at edges
dx = np.array([0.0, -0.05, 0.05])
dy = np.array([0.2 * r12, -0.1 * r12, -0.1 * r12])
nodelabel = [r"$\mathbf{N_2}$", r"$\mathbf{N_4}$", r"$\mathbf{N_6}$"]
for i, node in enumerate([N2, N4, N6]):
    ax[1].plot(
        node[0],
        node[1],
        "o",
        color="dodgerblue",
        markersize=10,
        markeredgecolor="k",
        zorder=3,
    )
    ax[1].text(
        node[0] + dx[i],
        node[1] + dy[i],
        nodelabel[i],
        fontsize=12,
        ha="center",
        va="center",
    )

# linear shape functions
LSF = [
    r"$S_1=\xi$",
    r"$S_2=\eta$",
    r"$S_3=1 - \xi - \eta$",
]
# quadratic shape functions
QSF = [
    r"$S_1=\xi (2 - \xi - 1)$",
    r"$S_2=4\xi\eta$",
    r"$S_3=\eta (2\eta - 1)$",
    r"$S_4=4\eta(1 - \xi - \eta)$",
    r"$S_5=(1 - \xi - \eta) (1 - 2\xi - 2\eta)$",
    r"$S_6=4\xi(1 - \xi - \eta)$",
]

# add titles and shape functions
# adjust axes
title = ["Linear (3-Node) Elements", "Quadratic (6-Node) Elements"]
shape_functions = [LSF, QSF]
for i, ax1 in enumerate(fig.axes):
    ax1.set_title(title[i], y=0, pad=0, fontsize=14)
    # ax1.text(
    #     0.0,
    #     0.95,
    #     "\n".join(shape_functions[i]),
    #     fontsize=9,
    #     linespacing=1.5,
    #     transform=ax1.transAxes,
    #     ha="left",
    #     va="top",
    # )
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1 * np.sqrt(3.0 / 4.0), 1.1 * np.sqrt(3.0 / 4.0))
    ax1.set_aspect("equal")
    ax1.axis("off")

fig.tight_layout()
plt.show()
