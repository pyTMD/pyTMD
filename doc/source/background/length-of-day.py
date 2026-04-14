import pyTMD
import numpy as np
import timescale
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

# MJD dates from Ray and Erofeeva (2014)
MJD1 = np.arange(37680, 55021)
# dates from Ray (1994)
MJD2 = 48830 + np.arange(86400 * 4) / 86400.0

# plot rotation rate variations
fig, ax = plt.subplots(num=1, nrows=2, sharex=False, figsize=(8, 4))

# predict rotation rate variations for the time period of Ray (1994)
ts = timescale.time.Timescale(MJD2)
# short-period tides
dsSP = pyTMD.predict.earth_orientation(ts.tide).sum(dim="constituent")
ax[0].plot(MJD2, 1e3 * dsSP.dUT, color="0.4")

# predict rotation rate variations for both time periods
for MJD, color in zip([MJD1, MJD2], ["0.4", "red"]):
    ts = timescale.time.Timescale(MJD)
    # long-period tides
    dsLP = pyTMD.predict.length_of_day(ts.tide).sum(dim="constituent")
    ax[1].plot(MJD, 1e3 * dsLP.dUT, color=color)
# add x and y labels
ax[1].set_xlabel("MJD")
ax[0].set_ylabel("\u0394UT [ms]")
ax[1].set_ylabel("\u0394UT [ms]")
ax[0].xaxis.get_major_locator().set_params(integer=True)
ax[0].ticklabel_format(useOffset=False, style="plain")
# add axes labels
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
# set axis limits and show plot
fig.tight_layout()
plt.show()
