import numpy as np
import pyTMD.constituents
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

# number of periods
N = 5000
# calculate over diurnal range
periods = np.linspace(0.85, 1.15, N)
# convert to radians per second
omegas = 2.0 * np.pi / (86400.0 * periods)
# spherical harmonic degree
n = 2
# 1066A-N values from Wahr (1979)
h0, k0, l0 = np.array([6.03e-1, 2.98e-1, 8.42e-2])
# tidal factors from combinations of Love numbers
delta0 = 1.0 + 2.0 * h0 / n - (n + 1.0) * k0 / n
gamma0 = 1.0 + k0 - h0
# initialize arrays
love = {}
for i, key in enumerate(["h", "k", "l"]):
    love[key] = np.zeros(N)
# calculate Love numbers
for i, omega in enumerate(omegas):
    love["h"][i], love["k"][i], love["l"][i] = pyTMD.earth.love_numbers(
        omega, model="1066A-N"
    )

# create figure and subplots
fig, ax = plt.subplots(
    num=1,
    nrows=2,
    sharex=True,
    figsize=(6, 3.5),
    facecolor="#fcfcfc",
)

# tidal factors from combinations of Love numbers
love["delta"] = 1.0 + 2.0 * love["h"] / n - (n + 1.0) * love["k"] / n
love["gamma"] = 1.0 + love["k"] - love["h"]
# plot tidal factors
for i, key in enumerate(["delta", "gamma"]):
    # remove the largest gradient
    grad = np.gradient(love[key])
    (ii,) = np.nonzero(np.abs(grad) == np.abs(grad).max())
    love[key][ii] = np.nan
    # plot tidal factors
    ax[i].plot(periods, love[key], "0.4")

# add markers for individual constituents
cons = ["o1", "p1", "k1", "phi1", "j1", "oo1"]
labels = ["o1", "p1", "k1", "\u03c61", "j1", "oo1"]
plot_colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(cons))))
for i, c in enumerate(cons):
    om = pyTMD.constituents.frequency(c)
    p = 2.0 * np.pi / (86400.0 * om)
    h, k, l = pyTMD.earth.love_numbers(om, model="1066A-N")
    delta = 1.0 + 2.0 * h / n - (n + 1.0) * k / n
    gamma = 1.0 + k - h
    (s,) = ax[0].plot(p, delta, ".", color=next(plot_colors), label=labels[i])
    ax[1].plot(p, gamma, ".", color=s.get_markerfacecolor(), label=labels[i])

# adjust axes
ax[0].set_xlim(periods.max(), periods.min())
ax[0].set_ylim(delta0 - 0.11, delta0 + 0.11)
ax[1].set_ylim(gamma0 - 0.11, gamma0 + 0.11)

# set axis labels
ax[1].set_xlabel("Period [day]")
ax[0].set_ylabel(r"$\delta_2$")
ax[1].set_ylabel(r"$\gamma_2$")
labels = ["a)", "b)"]
for i, label in enumerate(labels):
    ax[i].tick_params(which="both", direction="in")
    at = offsetbox.AnchoredText(
        label,
        loc=2,
        pad=0.0,
        borderpad=0.5,
        frameon=False,
        prop=dict(size=12, weight="bold", color="k"),
    )
    ax[i].axes.add_artist(at)

# add legend
lgd = ax[1].legend(
    loc=4, frameon=False, ncols=2, labelspacing=0.2, borderpad=0.05
)
for line in lgd.get_lines():
    line.set_markersize(10.0)

# adjust subplots
fig.subplots_adjust(top=0.99, bottom=0.115, left=0.10, right=0.95, hspace=0.1)
plt.show()
