import numpy as np
import pyTMD.earth
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

# adjust style
facecolor = "#fcfcfc"
plt.rcParams["figure.facecolor"] = facecolor
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Lato"]

# create figure and subplots
fig, ax = plt.subplots(
    num=1,
    nrows=3,
    sharex=True,
    figsize=(6, 5),
)

# read table of load Love numbers from Wang (2012)
table = pyTMD.earth._wang_prem_lln_table
hl, kl, ll = pyTMD.earth.load_love_numbers(table, reference="CF")
# array of spherical harmonic degrees
n = np.arange(len(hl))
# only plot for degrees 3+
hl[n < 3] = np.nan
kl[n < 3] = np.nan
ll[n < 3] = np.nan

# plot load Love numbers
ax[0].semilogx(n, hl, color="0.4")
ax[1].semilogx(n, n * kl, color="0.4")
ax[2].semilogx(n, n * ll, color="0.4")

ax[0].set_xlim(2, 7000)
# set axis labels
ax[2].set_xlabel("Degree $l$", labelpad=3)
ax[0].yaxis.set_major_formatter("{x:.1f}")
ax[0].set_ylabel("$h_l$", labelpad=3)
ax[1].set_ylabel("$k_l$ (scaled)", labelpad=3)
ax[2].set_ylabel("$l_l$ (scaled)", labelpad=9)

labels = ["a)", "b)", "c)"]
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

# adjust subplots
fig.subplots_adjust(top=0.99, bottom=0.085, left=0.10, right=0.95, hspace=0.1)
plt.show()
