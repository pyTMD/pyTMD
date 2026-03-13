import numpy as np
import matplotlib.pyplot as plt
import pyTMD.spatial

lat = np.arange(-90.0, 91.0)
theta = pyTMD.spatial.geocentric_latitude(lat)
fig, ax = plt.subplots(num=1, figsize=(8, 4))
ax.plot(lat, lat - theta, color="0.4")
ax.set_xlabel("Geodetic Latitude [\u00b0]")
ax.set_ylabel("Geodetic - Geocentric Latitude [\u00b0]")
fig.tight_layout()
plt.show()
