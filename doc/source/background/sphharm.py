import numpy as np
import pyTMD.math
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# latitude and longitude
dlon, dlat = 0.625, 0.5
lat = np.arange(-90 + dlat/2.0, 90 + dlat/2.0, dlat)
lon = np.arange(0 + dlon/2.0, 360 + dlon/2.0, dlon)
gridlon, gridlat = np.meshgrid(lon, lat)
# colatitude and longitude in radians
theta = xr.DataArray(np.radians(90.0 - gridlat), dims=('y','x'))
phi = xr.DataArray(np.radians(gridlon), dims=('y','x'))

# spherical harmonic degree
l = 2
# number of rows and columns for subplots
nrows = 2
ncols = l + 1
species = ["Long-Period", "Diurnal", "Semi-Diurnal"]
# projection for the plots
proj1 = ccrs.Orthographic(central_longitude=0.0, central_latitude=0.0)
proj2 = ccrs.Orthographic(central_longitude=180.0, central_latitude=0.0)

# plot spherical harmonics
fig = plt.figure(num=1, figsize=(8,6))
for tau in range(0, l+1):
    # setup subplots
    ax1 = fig.add_subplot(nrows, ncols, tau+1, projection=proj1)
    ax2 = fig.add_subplot(nrows, ncols, ncols+tau+1, projection=proj2)
    # calculate spherical harmonics (and derivatives w.r.t. colatitude)
    Y_lm, dY_lm = pyTMD.math.sph_harm(l, theta, phi, m=tau)
    # set the title
    ax1.set_title(f'{species[tau]}\n$l={l}, m={tau}$')
    # add projection text
    if tau == 0:
        proj_text = 'Projection centered on 0.0\u00b0E'
        ax1.text(0.0, -0.06, proj_text, fontsize=8, transform=ax1.transAxes)
        proj_text = 'Projection centered on 180.0\u00b0E'
        ax2.text(0.0, -0.06, proj_text, fontsize=8, transform=ax2.transAxes)
    # for each subplot axis
    for ax in (ax1, ax2):
        # plot the surface
        ax.pcolormesh(lon, lat, Y_lm.real,
            transform=ccrs.PlateCarree(),
            cmap='viridis', rasterized=True)
        # add coastlines and set global
        ax.coastlines()
        ax.set_global()
        # turn off the axis
        ax.set_axis_off()

# adjust spacing and show
plt.tight_layout()
plt.show()