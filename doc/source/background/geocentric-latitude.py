import numpy as np
import matplotlib.pyplot as plt
import pyTMD.spatial
from matplotlib.ticker import MultipleLocator

lat = np.arange(-90.0, 91.0)
theta = pyTMD.spatial.geocentric_latitude(lat)

fig, ax = plt.subplots(num=1, figsize=(8, 4), facecolor="#fcfcfc")
ax.plot(lat, lat - theta, color="0.4")
ax.set_xlim(-90, 90)
ax.xaxis.set_minor_locator(MultipleLocator(10))
# add labels
ax.set_xlabel("Geodetic Latitude [\u00b0]")
ax.set_ylabel("Geodetic - Geocentric Latitude [\u00b0]")
fig.tight_layout()
plt.show()
