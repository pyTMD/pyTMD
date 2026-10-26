import pyTMD
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# adjust style
facecolor = "#fcfcfc"
plt.rcParams["figure.facecolor"] = facecolor
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Lato"]

# Earth parameters and physical constants
a_axis = pyTMD.earth._wgs84.a_axis
flat = pyTMD.earth._wgs84.flat
GM = pyTMD.earth._wgs84.GM
# universal gravitational constant [N*m^2/kg^2]
G = 6.67430e-11
# average radius of the Earth with same volume as ellipsoid [m]
rad_e = a_axis * np.power(1.0 - flat, 1.0 / 3.0)
# average mass of the Earth
m_e = GM / G
# scale to convert mass to uplift
scale = rad_e / m_e

# Load Love number dataset
lln = "wang-prem"
# read lookup table of unscaled Green's functions
# uses reference angular distances from Farrell (1972)
theta, Gu, Gv = pyTMD.earth.read_greens_function(lln=lln, reference="CE")
# normalization in Table A3 of Farrell
norm = 1e12 * rad_e * theta

# create figure
fig, ax = plt.subplots(figsize=(8, 4))
# plot normalized Green's Functions
ax.plot(np.degrees(theta), scale * norm * Gu, color="0.4")
ax.set_xlabel("Angular Distance [\u00b0]")
ax.set_ylabel("Green's Function (normalized)")
ax.set_xlim(-1, 181)
ax.xaxis.set_major_locator(MultipleLocator(20))
ax.xaxis.set_minor_locator(MultipleLocator(10))
fig.tight_layout()
plt.show()
