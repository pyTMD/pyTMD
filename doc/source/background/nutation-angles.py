import timescale
import pyTMD.astro
from pyTMD.math import rad2asec
import matplotlib.pyplot as plt

# create a timescale object from a range of dates
ts = timescale.from_range("2000-01-01T12:00:00", "2018-08-08T12:00:00", 1, "D")
T = (ts.MJD - pyTMD.astro._mjd_j2000) / pyTMD.astro._century
# estimate the nutation angles
dpsi, deps = pyTMD.astro._nutation_angles(T)
# plot nutation angles in arcseconds
fig, ax = plt.subplots(num=1, figsize=(5, 5), facecolor="#fcfcfc")
ax.plot(rad2asec(dpsi), rad2asec(deps), color="0.4")
ax.axvline(0.0, color="0.4", lw=0.5, ls="--", dashes=(5, 5))
ax.axhline(0.0, color="0.4", lw=0.5, ls="--", dashes=(5, 5))
ax.set_xlabel("Nutation in Longitude $\Delta\psi$ [\u2033]")
ax.set_ylabel("Nutation in Obliquity $\Delta\epsilon$ [\u2033]")
fig.tight_layout()
plt.show()
