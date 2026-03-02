import pyTMD
import numpy as np
import timescale
import matplotlib.pyplot as plt
# dates from Ray (1994)
MJD = 48830 + np.arange(86400*4)/86400.0
ts = timescale.time.Timescale(MJD)
ds = pyTMD.predict.earth_orientation(ts.tide)
# plot length of day variations
fig, ax = plt.subplots(num=1, figsize=(8,4))
ax.plot(MJD, 1e6*ds.dUT, color='0.4')
ax.set_xlabel('MJD')
ax.set_ylabel('\u0394UT1 [\u03BCs]')
fig.tight_layout()
plt.show()
